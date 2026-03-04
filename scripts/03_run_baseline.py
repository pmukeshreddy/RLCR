#!/usr/bin/env python3
"""Step 3: Run the embedding baseline (Greptile's approach).

Implements Soohoon's cosine similarity filter:
  1. Encode comments with sentence-transformers/all-MiniLM-L6-v2
  2. Build per-team FAISS vector stores from training votes
  3. Tune thresholds to maximize F1 (fairest possible baseline)
  4. Evaluate on test sets

Usage:
    python scripts/03_run_baseline.py [--config configs/default.yaml]
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
from src.baselines.embedding_filter import EmbeddingFilter
from src.data.team_simulator import TeamSimulator

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Run embedding baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    console.rule("[bold blue]Step 3: Embedding Baseline[/bold blue]")

    teams_dir = Path(cfg.data.processed_dir) / "teams"
    if not teams_dir.exists():
        console.print("[red]Teams not found. Run step 02 first.[/red]")
        sys.exit(1)

    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator.load(teams_dir, team_configs)

    baseline = EmbeddingFilter(
        model_name=cfg.baseline.embedding_model,
        dim=cfg.baseline.embedding_dim,
        upvote_weight=cfg.baseline.upvote_weight,
        downvote_weight=cfg.baseline.downvote_weight,
        min_votes=cfg.baseline.min_votes_for_prediction,
        batch_size=cfg.baseline.batch_size,
    )

    console.rule("[bold]Building vector stores[/bold]")
    for name, team in simulator.teams.items():
        baseline.build_store(name, team.vote_history)

    console.rule("[bold]Tuning thresholds[/bold]")
    for name, team in simulator.teams.items():
        baseline.tune_threshold(
            name,
            team.train_samples,
            threshold_range=tuple(cfg.baseline.threshold_range),
            n_steps=cfg.baseline.threshold_steps,
        )

    console.rule("[bold]Evaluation[/bold]")
    results = {}
    table = Table(title="Embedding Baseline Results")
    table.add_column("Team", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Precision", style="yellow")
    table.add_column("Recall", style="yellow")
    table.add_column("F1", style="magenta")
    table.add_column("Action Rate", style="blue")
    table.add_column("Threshold", style="red")

    for name, team in simulator.teams.items():
        metrics = baseline.evaluate(name, team.test_samples)
        results[name] = metrics
        table.add_row(
            name.title(),
            f"{metrics['accuracy']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{metrics['action_rate']:.3f}",
            f"{metrics['threshold']:.3f}",
        )

    console.print(table)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    baseline.save(results_dir / "baseline_model")

    console.print(f"\n[bold green]✓ Step 3 complete![/bold green]")
    console.print(f"  Results saved to: {results_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
