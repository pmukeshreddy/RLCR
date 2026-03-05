#!/usr/bin/env python3
"""Step 9: Teacher Labeling — score full corpus with trained DAPO models.

Runs the RL teacher (Qwen3-4B + per-team LoRA) over every sample to
produce soft labels (continuous scores 0.0-1.0). These labels encode
the teacher's nuanced understanding of per-team relevance and are used
as training targets for the distilled embedding model.

Uses SGLang for fast batch inference.

Usage:
    python scripts/09_teacher_label.py
    python scripts/09_teacher_label.py --small   # Use 1.7B teacher
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import load_config, set_seed
from src.data.team_simulator import TeamSimulator
from src.distillation.teacher_labeler import label_all_teams, save_labels

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Teacher labeling")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    model_name = cfg.model.small.name if args.small else cfg.model.large.name
    sglang_url = None
    if cfg.model.sglang.get("enabled", True):
        sglang_url = f"http://{cfg.model.sglang.host}:{cfg.model.sglang.port}"

    console.rule("[bold blue]Step 9: Teacher Labeling[/bold blue]")
    console.print(f"Model: {model_name}")
    console.print(f"SGLang: {'enabled' if sglang_url else 'disabled'}")

    teams_dir = Path(cfg.data.processed_dir) / "teams"
    if not teams_dir.exists():
        console.print("[red]Teams not found. Run step 02 first.[/red]")
        sys.exit(1)

    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator.load(teams_dir, team_configs)

    labels = label_all_teams(
        teams=simulator.teams,
        model_name=model_name,
        sglang_url=sglang_url,
        max_new_tokens=cfg.model.large.max_new_tokens,
    )

    output_dir = Path("data/distillation_labels")
    save_labels(labels, output_dir)

    table = Table(title="Teacher Labeling Summary")
    table.add_column("Team", style="cyan")
    table.add_column("Samples", style="green")
    table.add_column("Mean Score", style="yellow")
    table.add_column("Score Std", style="yellow")

    for team_name, records in labels.items():
        scores = [r["teacher_score"] for r in records]
        import numpy as np
        table.add_row(
            team_name.title(),
            str(len(records)),
            f"{np.mean(scores):.3f}",
            f"{np.std(scores):.3f}",
        )
    console.print(table)
    console.print(f"\n[bold green]✓ Step 9 complete![/bold green]")
    console.print(f"  Labels saved: {output_dir}")


if __name__ == "__main__":
    main()
