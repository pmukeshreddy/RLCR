#!/usr/bin/env python3
"""Step 10: Distillation — fine-tune sentence transformer from teacher labels.

Takes the soft labels from Step 9 and:
  1. Fine-tunes a MiniLM backbone with contrastive pairs so that the
     embedding space encodes RL-informed relevance (not just semantics).
  2. Trains per-team projection heads for direct score prediction.

The result is a lightweight, CPU-friendly model that serves at <10ms
per query with the RL teacher's understanding baked in.

Usage:
    python scripts/10_distill.py
    python scripts/10_distill.py --epochs 5   # More backbone training
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console

from src.config import load_config, set_seed
from src.distillation.teacher_labeler import load_labels
from src.distillation.distill_trainer import distill

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Distillation training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--labels-dir", default="data/distillation_labels")
    parser.add_argument("--output-dir", default="outputs/distilled_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    console.rule("[bold blue]Step 10: Distillation[/bold blue]")

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        console.print("[red]Teacher labels not found. Run step 09 first.[/red]")
        sys.exit(1)

    labels_by_team = load_labels(labels_dir)
    total = sum(len(v) for v in labels_by_team.values())
    console.print(f"Loaded {total} teacher-labeled samples across {len(labels_by_team)} teams")

    base_model = cfg.baseline.embedding_model
    console.print(f"Base model: {base_model}")
    console.print(f"Backbone epochs: {args.epochs}")
    console.print(f"Head epochs: {args.head_epochs}")
    console.print(f"Output: {args.output_dir}")

    backbone, heads = distill(
        labels_by_team=labels_by_team,
        base_model_name=base_model,
        output_dir=args.output_dir,
        backbone_epochs=args.epochs,
        head_epochs=args.head_epochs,
        batch_size=args.batch_size,
        seed=cfg.project.seed,
    )

    console.print(f"\n[bold green]✓ Step 10 complete![/bold green]")
    console.print(f"  Backbone: {args.output_dir}")
    console.print(f"  Heads: {args.output_dir}/heads/")
    for team_name in heads:
        console.print(f"    • {team_name}.pt")


if __name__ == "__main__":
    main()
