#!/usr/bin/env python3
"""Step 5: GRPO Training Loop.

Trains per-team scoring policies using Group Relative Policy Optimization:
  - Reward = address rate from CodeReviewer labels
  - LoRA adapters for memory efficiency
  - Ray distribution across teams (default)

Uses Qwen3-4B by default (cfg.model.large). Pass --small to use Qwen3-1.7B.

Usage:
    python scripts/05_grpo_train.py                    # All teams, Ray, Qwen3-4B
    python scripts/05_grpo_train.py --team security    # Single team
    python scripts/05_grpo_train.py --no-ray           # Sequential training
    python scripts/05_grpo_train.py --small            # Use Qwen3-1.7B instead
"""

from __future__ import annotations

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
from src.training.grpo import (
    RLCRTrainer,
    GRPORunConfig,
    build_training_dataset,
    train_all_teams_ray,
)
from transformers import AutoTokenizer

console = Console()


def train_single_team(team_name: str, team, model_name: str, sglang_url: str | None, cfg, tokenizer):
    """Train GRPO on a single team."""
    console.print(f"\n[bold cyan]Training team: {team_name}[/bold cyan]")

    train_dicts = [s.to_dict() for s in team.train_samples]
    eval_dicts = [s.to_dict() for s in team.test_samples[:50]]

    train_ds = build_training_dataset(
        train_dicts, team_name, team.description, team.vote_history, tokenizer
    )
    eval_ds = build_training_dataset(
        eval_dicts, team_name, team.description, team.vote_history, tokenizer
    )

    run_config = GRPORunConfig(
        model_name=model_name,
        output_dir=f"outputs/grpo/{team_name}",
        team_name=team_name,
        team_description=team.description,
        vote_history=team.vote_history,
        lora_r=cfg.training.lora.r,
        lora_alpha=cfg.training.lora.alpha,
        lora_dropout=cfg.training.lora.dropout,
        lora_target_modules=list(cfg.training.lora.target_modules),
        group_size=cfg.training.grpo.group_size,
        learning_rate=cfg.training.grpo.learning_rate,
        num_epochs=cfg.training.grpo.num_epochs,
        per_device_batch_size=cfg.training.grpo.per_device_batch_size,
        gradient_accumulation_steps=cfg.training.grpo.gradient_accumulation_steps,
        warmup_ratio=cfg.training.grpo.warmup_ratio,
        kl_coef=cfg.training.grpo.kl_coef,
        clip_range=cfg.training.grpo.clip_range,
        max_grad_norm=cfg.training.grpo.max_grad_norm,
        logging_steps=cfg.training.grpo.logging_steps,
        save_steps=cfg.training.grpo.save_steps,
        eval_steps=cfg.training.grpo.eval_steps,
        seed=cfg.project.seed,
        sglang_url=sglang_url,
    )

    trainer = RLCRTrainer(run_config)
    result = trainer.train(train_ds, eval_ds)
    return result


def main():
    parser = argparse.ArgumentParser(description="GRPO training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--team", default=None, help="Train specific team only")
    parser.add_argument("--no-ray", action="store_true", help="Disable Ray, train teams sequentially")
    parser.add_argument("--small", action="store_true", help="Use small model (Qwen3-1.7B) instead of default 4B")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    use_ray = cfg.training.ray.get("enabled", True) and not args.no_ray
    model_name = cfg.model.small.name if args.small else cfg.model.large.name

    sglang_url = None
    if cfg.model.sglang.get("enabled", True):
        sglang_url = f"http://{cfg.model.sglang.host}:{cfg.model.sglang.port}"

    console.rule("[bold blue]Step 5: GRPO Training[/bold blue]")

    teams_dir = Path(cfg.data.processed_dir) / "teams"
    if not teams_dir.exists():
        console.print("[red]Teams not found. Run step 02 first.[/red]")
        sys.exit(1)

    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator.load(teams_dir, team_configs)

    console.print(f"Model: {model_name}")
    console.print(f"Method: GRPO with LoRA (r={cfg.training.lora.r})")
    console.print(f"Group size: {cfg.training.grpo.group_size}")
    console.print(f"Epochs: {cfg.training.grpo.num_epochs}")
    console.print(f"Ray: {'enabled' if use_ray else 'disabled (sequential)'}")
    console.print(f"Rollout generation: {'SGLang' if sglang_url else 'local model'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_ray and not args.team:
        console.print("\n[bold]Distributing training across teams via Ray[/bold]")
        config_dict = {
            "model_name": model_name,
            "base_output_dir": "outputs/grpo",
            "lora_r": cfg.training.lora.r,
            "lora_alpha": cfg.training.lora.alpha,
            "lora_dropout": cfg.training.lora.dropout,
            "group_size": cfg.training.grpo.group_size,
            "learning_rate": cfg.training.grpo.learning_rate,
            "num_epochs": cfg.training.grpo.num_epochs,
            "per_device_batch_size": cfg.training.grpo.per_device_batch_size,
            "gradient_accumulation_steps": cfg.training.grpo.gradient_accumulation_steps,
            "seed": cfg.project.seed,
            "sglang_url": sglang_url,
        }
        results = train_all_teams_ray(
            simulator.teams,
            config_dict,
            num_cpus=cfg.training.ray.num_cpus,
            num_gpus=cfg.training.ray.num_gpus,
        )
    else:
        if args.team:
            console.print(f"\n[bold]Training single team: {args.team}[/bold]")
        teams_to_train = (
            {args.team: simulator.teams[args.team]}
            if args.team
            else simulator.teams
        )
        results = {}
        for name, team in teams_to_train.items():
            result = train_single_team(name, team, model_name, sglang_url, cfg, tokenizer)
            results[name] = result

    table = Table(title="Training Results")
    table.add_column("Team", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Output Dir", style="yellow")

    for name, result in results.items():
        status = "Error" if "error" in result else "Complete"
        style = "red" if "error" in result else "green"
        table.add_row(
            name.title(),
            f"[{style}]{status}[/{style}]",
            result.get("output_dir", "N/A"),
        )

    console.print(table)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: str(vv) for kk, vv in v.items()}
    with open(results_dir / "training_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    console.print(f"\n[bold green]✓ Step 5 complete![/bold green]")


if __name__ == "__main__":
    main()
