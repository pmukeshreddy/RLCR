#!/usr/bin/env python3
"""Step 7: Scale to Qwen3-4B.

Swaps the model to a larger variant, retrains, and compares results:
  - Retrain DAPO with the larger model
  - Re-evaluate with the same cold-start protocol
  - Generate 1.7B vs 4B comparison data

Usage:
    python scripts/07_scale_4b.py [--config configs/default.yaml] [--team TEAM]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import load_config, set_seed
from src.data.team_simulator import TeamSimulator
from src.evaluation.cold_start import ColdStartEvaluator
from src.evaluation.metrics import compute_metrics
from src.models.scoring import ReviewScorer
from src.models.sglang_server import SGLangServer
from src.training.grpo import train_all_teams
from transformers import AutoTokenizer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Scale to larger model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--team", default=None)
    parser.add_argument("--eval-only", action="store_true", help="Skip training, evaluate only")
    parser.add_argument("--no-sglang", action="store_true", help="Force local HF inference instead of SGLang")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    use_sglang = cfg.model.sglang.get("enabled", True) and not args.no_sglang

    sglang_url = None
    if cfg.model.sglang.get("enabled", True) and not args.no_sglang:
        sglang_url = f"http://{cfg.model.sglang.host}:{cfg.model.sglang.port}"

    console.rule("[bold blue]Step 7: Scale to Larger Model[/bold blue]")

    large_model = cfg.model.large.name
    small_model = cfg.model.small.name
    console.print(f"Small model: {small_model}")
    console.print(f"Large model: {large_model}")

    teams_dir = Path(cfg.data.processed_dir) / "teams"
    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator.load(teams_dir, team_configs)

    teams_to_run = (
        {args.team: simulator.teams[args.team]} if args.team else simulator.teams
    )

    # --- Train with larger model ---
    if not args.eval_only:
        console.rule("[bold]Training with larger model[/bold]")
        console.print(f"[cyan]Model loaded once, LoRA swapped per team[/cyan]")

        dapo = cfg.training.dapo
        config_dict = {
            "model_name": large_model,
            "base_output_dir": "outputs/dapo_large",
            "lora_r": cfg.training.lora.r,
            "lora_alpha": cfg.training.lora.alpha,
            "lora_dropout": cfg.training.lora.get("dropout", 0.05),
            "lora_target_modules": list(cfg.training.lora.target_modules),
            "group_size": dapo.group_size,
            "learning_rate": dapo.learning_rate,
            "num_epochs": dapo.num_epochs,
            "per_device_batch_size": dapo.per_device_batch_size,
            "ppo_epochs": dapo.ppo_epochs,
            "clip_ratio_low": dapo.clip_ratio_low,
            "clip_ratio_high": dapo.clip_ratio_high,
            "dynamic_sampling": dapo.dynamic_sampling,
            "overlong_penalty": dapo.overlong_penalty,
            "overlong_buffer_len": dapo.overlong_buffer_len,
            "seed": cfg.project.seed,
            "sglang_url": sglang_url,
        }
        train_all_teams(teams_to_run, config_dict)

    # --- Evaluate both models ---
    console.rule("[bold]Comparing models[/bold]")

    eval_sglang_url = f"http://{cfg.model.sglang.host}:{cfg.model.sglang.port}"
    sglang_server = None
    if use_sglang:
        sglang_server = SGLangServer(
            model_name=small_model,
            host=cfg.model.sglang.host,
            port=cfg.model.sglang.port,
            mem_fraction=cfg.model.sglang.mem_fraction,
        )

    results = {"small": {}, "large": {}}
    for model_key, model_name in [("small", small_model), ("large", large_model)]:
        console.print(f"\n[bold]Evaluating {model_key}: {model_name}[/bold]")

        if sglang_server is not None:
            console.print(f"[cyan](Re)starting SGLang with {model_name}...[/cyan]")
            if not sglang_server.restart(model_name=model_name):
                console.print("[yellow]SGLang failed — falling back to local HF[/yellow]")
                sglang_server = None
                use_sglang = False

        try:
            scorer = ReviewScorer(
                model_name=model_name,
                device="auto",
                max_new_tokens=256,
                temperature=0.7,
                use_sglang=use_sglang,
                sglang_url=eval_sglang_url,
            )
        except Exception as e:
            console.print(f"[red]Could not load {model_name}: {e}[/red]")
            continue

        for team_name, team in teams_to_run.items():
            test_subset = team.test_samples[:100]
            labels = []
            predictions = []
            scores = []

            for sample in test_subset:
                output = scorer.score(
                    diff=sample.diff,
                    comment=sample.comment,
                    team_name=team_name,
                    team_description=team.description,
                    vote_history=team.vote_history,
                )
                labels.append(sample.label)
                predictions.append(output.binary_label)
                scores.append(output.score)

            metrics = compute_metrics(labels, predictions, scores)
            results[model_key][team_name] = metrics.to_dict()
            console.print(f"  {team_name}: {metrics.summary_str()}")

    if sglang_server is not None:
        console.print("\n[dim]Stopping managed SGLang server...[/dim]")
        sglang_server.stop()

    # --- Comparison table ---
    table = Table(title="Model Size Comparison (F1)")
    table.add_column("Team", style="cyan")
    table.add_column(f"Small ({small_model})", style="blue")
    table.add_column(f"Large ({large_model})", style="green")
    table.add_column("Δ", style="magenta")

    for team_name in teams_to_run:
        small_f1 = results.get("small", {}).get(team_name, {}).get("f1", 0)
        large_f1 = results.get("large", {}).get(team_name, {}).get("f1", 0)
        delta = large_f1 - small_f1
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        table.add_row(team_name.title(), f"{small_f1:.3f}", f"{large_f1:.3f}", delta_str)

    console.print(table)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "model_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"\n[bold green]✓ Step 7 complete![/bold green]")


if __name__ == "__main__":
    main()
