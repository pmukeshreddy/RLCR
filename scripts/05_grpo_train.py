#!/usr/bin/env python3
"""Step 5: DAPO Training Loop.

Trains per-team scoring policies using DAPO (Decoupled Clip and Dynamic
sAmpling Policy Optimization):
  - Reward = address rate from real code review labels
  - LoRA adapters for memory efficiency
  - Clip-Higher, token-level loss, dynamic sampling, no KL penalty
  - Loads base model once, swaps LoRA per team (efficient single-GPU default)
  - Optional: --ray for multi-GPU parallel training

Uses Qwen3-4B by default (cfg.model.large). Pass --small to use Qwen3-1.7B.

Usage:
    python scripts/05_grpo_train.py                    # All teams, Qwen3-4B
    python scripts/05_grpo_train.py --team security    # Single team
    python scripts/05_grpo_train.py --ray              # Multi-GPU via Ray
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
    train_all_teams,
    train_all_teams_ray,
)
from transformers import AutoTokenizer

console = Console()


def train_single_team(team_name: str, team, model_name: str, sglang_url: str | None, cfg, tokenizer):
    """Train DAPO on a single team (standalone, loads its own model)."""
    console.print(f"\n[bold cyan]Training team: {team_name}[/bold cyan]")

    train_dicts = [s.to_dict() for s in team.train_samples]
    eval_dicts = [s.to_dict() for s in team.test_samples[:50]]

    train_ds = build_training_dataset(
        train_dicts, team_name, team.description, team.vote_history, tokenizer
    )
    eval_ds = build_training_dataset(
        eval_dicts, team_name, team.description, team.vote_history, tokenizer
    )

    dapo = cfg.training.dapo
    run_config = GRPORunConfig(
        model_name=model_name,
        output_dir=f"outputs/dapo/{team_name}",
        team_name=team_name,
        team_description=team.description,
        vote_history=team.vote_history,
        lora_r=cfg.training.lora.r,
        lora_alpha=cfg.training.lora.alpha,
        lora_dropout=cfg.training.lora.dropout,
        lora_target_modules=list(cfg.training.lora.target_modules),
        group_size=dapo.group_size,
        learning_rate=dapo.learning_rate,
        num_epochs=dapo.num_epochs,
        per_device_batch_size=dapo.per_device_batch_size,
        ppo_epochs=dapo.ppo_epochs,
        warmup_ratio=dapo.warmup_ratio,
        clip_ratio_low=dapo.clip_ratio_low,
        clip_ratio_high=dapo.clip_ratio_high,
        dynamic_sampling=dapo.dynamic_sampling,
        overlong_penalty=dapo.overlong_penalty,
        overlong_buffer_len=dapo.overlong_buffer_len,
        max_completion_length=getattr(dapo, "max_completion_length", 128),
        max_grad_norm=dapo.max_grad_norm,
        logging_steps=dapo.logging_steps,
        save_steps=dapo.save_steps,
        eval_steps=dapo.eval_steps,
        seed=cfg.project.seed,
        sglang_url=sglang_url,
    )

    trainer = RLCRTrainer(run_config)
    result = trainer.train(train_ds, eval_ds)
    return result


def _build_config_dict(model_name: str, sglang_url: str | None, cfg) -> dict:
    """Build config dict for train_all_teams / train_all_teams_ray."""
    dapo = cfg.training.dapo
    return {
        "model_name": model_name,
        "base_output_dir": "outputs/dapo",
        "lora_r": cfg.training.lora.r,
        "lora_alpha": cfg.training.lora.alpha,
        "lora_dropout": cfg.training.lora.dropout,
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
        "max_completion_length": getattr(dapo, "max_completion_length", 128),
        "sglang_concurrent": getattr(dapo, "sglang_concurrent", 64),
        "seed": cfg.project.seed,
        "sglang_url": sglang_url,
    }


def main():
    parser = argparse.ArgumentParser(description="DAPO training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--team", default=None, help="Train specific team only")
    parser.add_argument("--ray", action="store_true", help="Multi-GPU parallel training via Ray")
    parser.add_argument("--small", action="store_true", help="Use small model (Qwen3-1.7B) instead of default 4B")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    model_name = cfg.model.small.name if args.small else cfg.model.large.name

    sglang_url = None
    if cfg.model.sglang.get("enabled", True):
        sglang_url = f"http://{cfg.model.sglang.host}:{cfg.model.sglang.port}"

    console.rule("[bold blue]Step 5: DAPO Training[/bold blue]")

    teams_dir = Path(cfg.data.processed_dir) / "teams"
    if not teams_dir.exists():
        console.print("[red]Teams not found. Run step 02 first.[/red]")
        sys.exit(1)

    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator.load(teams_dir, team_configs)

    dapo = cfg.training.dapo
    mode = "Ray (multi-GPU)" if args.ray else "single-GPU (model loaded once, LoRA swapped per team)"
    console.print(f"Model: {model_name}")
    console.print(f"Method: DAPO with LoRA (r={cfg.training.lora.r})")
    console.print(f"Group size: {dapo.group_size}")
    console.print(f"Clip-Higher: low={dapo.clip_ratio_low}, high={dapo.clip_ratio_high}")
    console.print(f"PPO inner epochs: {dapo.ppo_epochs}")
    console.print(f"Dynamic sampling: {dapo.dynamic_sampling}")
    console.print(f"Epochs: {dapo.num_epochs}")
    console.print(f"Training mode: {mode}")
    console.print(f"Rollout generation: {'SGLang' if sglang_url else 'local model'}")

    if args.team:
        console.print(f"\n[bold]Training single team: {args.team}[/bold]")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        team = simulator.teams[args.team]
        results = {args.team: train_single_team(args.team, team, model_name, sglang_url, cfg, tokenizer)}

    elif args.ray:
        console.print("\n[bold]Multi-GPU training via Ray[/bold]")
        config_dict = _build_config_dict(model_name, sglang_url, cfg)
        results = train_all_teams_ray(
            simulator.teams,
            config_dict,
            num_cpus=cfg.training.ray.num_cpus,
            num_gpus=cfg.training.ray.num_gpus,
        )

    else:
        console.print("\n[bold]Training all teams (base model loaded once, LoRA swapped per team)[/bold]")
        config_dict = _build_config_dict(model_name, sglang_url, cfg)
        results = train_all_teams(simulator.teams, config_dict)

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
