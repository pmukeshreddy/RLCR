#!/usr/bin/env python3
"""Step 6: Comprehensive Evaluation.

Runs:
  1. Cold-start curves at 0, 5, 10, 20, 50, 100 samples
  2. Per-team breakdown with full metrics
  3. Generalization test (train style → test thorough)
  4. 3 seeds minimum, reports mean ± std

SGLang is used by default for fast inference. If the server isn't running,
falls back to local HF inference automatically.

Usage:
    python scripts/06_evaluate.py [--config configs/default.yaml]
    python scripts/06_evaluate.py --no-sglang   # Force local HF inference
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import load_config, set_seed
from src.baselines.embedding_filter import EmbeddingFilter
from src.data.team_simulator import TeamSimulator
from src.evaluation.cold_start import ColdStartEvaluator, run_generalization_test
from src.evaluation.metrics import compute_metrics
from src.models.scoring import ReviewScorer
from src.models.sglang_server import SGLangServer
from src.training.grpo import build_training_dataset, train_all_teams, GRPORunConfig, RLCRTrainer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Full evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-sglang", action="store_true", help="Force local HF inference instead of SGLang")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL eval (baseline only)")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    use_sglang = cfg.model.sglang.get("enabled", True) and not args.no_sglang

    console.rule("[bold blue]Step 6: Evaluation[/bold blue]")

    teams_dir = Path(cfg.data.processed_dir) / "teams"
    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator.load(teams_dir, team_configs)

    cold_start = ColdStartEvaluator(
        steps=list(cfg.evaluation.cold_start_steps),
        n_seeds=cfg.evaluation.n_seeds,
        base_seed=cfg.project.seed,
    )

    # --- Baseline Cold-Start Evaluation ---
    console.rule("[bold]Embedding Baseline Cold-Start[/bold]")
    baseline_kwargs = {
        "model_name": cfg.baseline.embedding_model,
        "dim": cfg.baseline.embedding_dim,
        "upvote_weight": cfg.baseline.upvote_weight,
        "downvote_weight": cfg.baseline.downvote_weight,
        "min_votes": cfg.baseline.min_votes_for_prediction,
        "batch_size": cfg.baseline.batch_size,
    }

    for team_name, team in simulator.teams.items():
        console.print(f"\n[cyan]Evaluating baseline: {team_name}[/cyan]")
        cold_start.evaluate_embedding_baseline(
            baseline_cls=EmbeddingFilter,
            baseline_kwargs=baseline_kwargs,
            team_name=team_name,
            all_train_samples=team.train_samples,
            test_samples=team.test_samples,
        )

    # --- RL Cold-Start Evaluation ---
    rl_metrics_per_team = {}
    sglang_managed = None

    if not args.skip_rl:
        console.rule("[bold]RL Model Cold-Start[/bold]")

        model_name = cfg.model.large.name
        sglang_url = f"http://{cfg.model.sglang.host}:{cfg.model.sglang.port}"
        console.print(f"Model: {model_name}")
        console.print(f"SGLang: {'enabled' if use_sglang else 'disabled (local HF inference)'}")

        # If SGLang is requested but not already running, launch it as a subprocess
        if use_sglang:
            server = SGLangServer(
                model_name=model_name,
                host=cfg.model.sglang.host,
                port=cfg.model.sglang.port,
                mem_fraction=cfg.model.sglang.mem_fraction,
            )
            if not server.is_healthy:
                console.print("[cyan]Launching SGLang server as subprocess...[/cyan]")
                if server.start():
                    sglang_managed = server
                else:
                    console.print("[yellow]SGLang launch failed — falling back to local HF[/yellow]")
                    use_sglang = False

        scorer = ReviewScorer(
            model_name=model_name,
            device="auto",
            max_new_tokens=cfg.model.large.max_new_tokens,
            temperature=cfg.model.large.temperature,
            use_sglang=use_sglang,
            sglang_url=sglang_url,
        )

        dapo_cfg = cfg.training.dapo
        sglang_train_url = sglang_url if use_sglang else None

        def make_train_fn(t_name, t_desc, t_votes):
            """Create a train_fn closure that trains a fresh LoRA from N samples."""
            def train_fn(samples, seed=42):
                from transformers import AutoTokenizer as AT
                tok = AT.from_pretrained(model_name, trust_remote_code=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                sample_dicts = [s.to_dict() for s in samples]
                ds = build_training_dataset(sample_dicts, t_name, t_desc, t_votes, tok)
                rc = GRPORunConfig(
                    model_name=model_name,
                    output_dir=f"outputs/dapo_coldstart/{t_name}_n{len(samples)}_s{seed}",
                    team_name=t_name,
                    team_description=t_desc,
                    vote_history=t_votes,
                    lora_target_modules=list(cfg.training.lora.target_modules),
                    group_size=min(getattr(dapo_cfg, "group_size", 8), 4),
                    num_epochs=1,
                    per_device_batch_size=getattr(dapo_cfg, "per_device_batch_size", 2),
                    max_completion_length=getattr(dapo_cfg, "max_completion_length", 256),
                    ppo_epochs=1,
                    seed=seed,
                    sglang_url=sglang_train_url,
                )
                trainer = RLCRTrainer(rc)
                trainer.train(ds, None)
            return train_fn

        for team_name, team in simulator.teams.items():
            console.print(f"\n[cyan]Evaluating RL: {team_name}[/cyan]")

            test_subset = team.test_samples[:50]
            train_fn = make_train_fn(
                team_name, team.description, team.vote_history
            )

            cold_start.evaluate_rl_model(
                scorer=scorer,
                team_name=team_name,
                team_description=team.description,
                all_train_samples=team.train_samples,
                test_samples=test_subset,
                train_fn=train_fn,
            )

    # --- Aggregate Results ---
    console.rule("[bold]Results[/bold]")
    aggregated = cold_start.aggregate_results()

    table = Table(title="Cold-Start Performance (F1 at key points)")
    table.add_column("Method / Team", style="cyan")
    for step in cfg.evaluation.cold_start_steps:
        table.add_column(f"n={step}", style="green")

    for key, step_data in sorted(aggregated.items()):
        row = [key]
        for step in cfg.evaluation.cold_start_steps:
            if step in step_data and "f1" in step_data[step]:
                mean = step_data[step]["f1"]["mean"]
                std = step_data[step]["f1"]["std"]
                row.append(f"{mean:.3f}±{std:.3f}")
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)

    # --- Generalization Test ---
    if not args.skip_rl:
        console.rule("[bold]Generalization Test[/bold]")
        train_team = cfg.evaluation.generalization.train_pattern
        test_team = cfg.evaluation.generalization.test_pattern

        if train_team in simulator.teams and test_team in simulator.teams:
            gen_result = run_generalization_test(
                scorer=scorer,
                train_team=train_team,
                test_team=test_team,
                train_description=simulator.teams[train_team].description,
                test_description=simulator.teams[test_team].description,
                train_samples=simulator.teams[train_team].train_samples,
                test_samples=simulator.teams[test_team].test_samples[:50],
                n_seeds=cfg.evaluation.n_seeds,
                base_seed=cfg.project.seed,
            )

            console.print(
                f"\nGeneralization: Train on [bold]{train_team}[/bold] → "
                f"Test on [bold]{test_team}[/bold]"
            )
            for metric, vals in gen_result["metrics"].items():
                console.print(f"  {metric}: {vals['mean']:.3f} ± {vals['std']:.3f}")

            aggregated["generalization"] = gen_result

    # --- Save ---
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    cold_start.save_results(results_dir / "cold_start_results.json")

    with open(results_dir / "full_evaluation.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    # Clean up SGLang if we launched it
    if sglang_managed is not None:
        console.print("\n[dim]Stopping managed SGLang server...[/dim]")
        sglang_managed.stop()

    console.print(f"\n[bold green]✓ Step 6 complete![/bold green]")
    console.print(f"  Full results: {results_dir / 'full_evaluation.json'}")


if __name__ == "__main__":
    main()
