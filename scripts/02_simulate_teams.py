#!/usr/bin/env python3
"""Step 2: Simulate 5 development teams by clustering comment patterns.

Creates teams:
  - Security-focused: vulnerability detection, secure coding
  - Style-obsessed: naming conventions, formatting
  - Performance-focused: efficiency, complexity analysis
  - Pragmatic: quick, actionable, low-ceremony
  - Thorough: detailed, educational, principle-driven

Each team gets 20-50 training samples and 200+ test samples.

Usage:
    python scripts/02_simulate_teams.py [--config configs/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_from_disk
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import load_config, set_seed
from src.data.parser import CodeReviewSample
from src.data.team_simulator import TeamSimulator

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Simulate development teams")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    console.rule("[bold blue]Step 2: Team Simulation[/bold blue]")

    processed_dir = Path(cfg.data.processed_dir)
    all_samples = []
    for split_name in ["train", "validation", "test"]:
        split_path = processed_dir / split_name
        if split_path.exists():
            ds = load_from_disk(str(split_path))
            for row in ds:
                all_samples.append(CodeReviewSample(
                    diff=row["diff"],
                    comment=row["comment"],
                    label=row["label"],
                    diff_tokens=row.get("diff_tokens", 0),
                    comment_tokens=row.get("comment_tokens", 0),
                ))
            logger.info(f"Loaded {len(ds)} samples from {split_name}")

    if not all_samples:
        console.print("[red]No processed data found. Run step 01 first.[/red]")
        sys.exit(1)

    console.print(f"Total samples available: {len(all_samples)}")

    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator(team_configs, seed=cfg.project.seed)

    teams = simulator.assign_samples(
        all_samples,
        train_range=tuple(cfg.teams.train_samples_range),
        min_test=cfg.teams.min_test_samples,
        fallback_random=cfg.teams.fallback_random_assignment,
    )

    table = Table(title="Team Assignments")
    table.add_column("Team", style="cyan")
    table.add_column("Train", style="green")
    table.add_column("Test", style="yellow")
    table.add_column("Total", style="magenta")
    table.add_column("+Rate (train)", style="blue")
    table.add_column("+Rate (test)", style="blue")

    for name, team in teams.items():
        summary = team.summary()
        table.add_row(
            name.title(),
            str(summary["train_count"]),
            str(summary["test_count"]),
            str(summary["train_count"] + summary["test_count"]),
            f"{summary['train_positive_rate']:.3f}",
            f"{summary['test_positive_rate']:.3f}",
        )

    console.print(table)

    output_dir = processed_dir / "teams"
    simulator.save(output_dir)

    console.print(f"\n[bold green]✓ Step 2 complete![/bold green]")
    console.print(f"  Teams saved to: {output_dir}")


if __name__ == "__main__":
    main()
