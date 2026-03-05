#!/usr/bin/env python3
"""Step 6b: A/B Split Evaluation.

Within-team 50/50 A/B test: for each team, splits test data in half,
evaluates both DAPO RL and embedding baseline on both halves, repeated
across multiple seeds. Reports paired t-test with p-values.

Usage:
    python scripts/06b_ab_test.py [--config configs/default.yaml]
    python scripts/06b_ab_test.py --no-sglang   # Force local HF inference
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
from src.evaluation.ab_test import run_ab_test
from src.models.scoring import ReviewScorer
from src.models.sglang_server import SGLangServer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="A/B split evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-sglang", action="store_true")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    use_sglang = cfg.model.sglang.get("enabled", True) and not args.no_sglang

    console.rule("[bold blue]Step 6b: A/B Split Evaluation[/bold blue]")

    teams_dir = Path(cfg.data.processed_dir) / "teams"
    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator.load(teams_dir, team_configs)

    baseline_kwargs = {
        "model_name": cfg.baseline.embedding_model,
        "dim": cfg.baseline.embedding_dim,
        "upvote_weight": cfg.baseline.upvote_weight,
        "downvote_weight": cfg.baseline.downvote_weight,
        "min_votes": cfg.baseline.min_votes_for_prediction,
        "batch_size": cfg.baseline.batch_size,
    }

    model_name = cfg.model.large.name
    sglang_url = f"http://{cfg.model.sglang.host}:{cfg.model.sglang.port}"
    sglang_managed = None

    if use_sglang:
        server = SGLangServer(
            model_name=model_name,
            host=cfg.model.sglang.host,
            port=cfg.model.sglang.port,
            mem_fraction=cfg.model.sglang.mem_fraction,
        )
        if not server.is_healthy:
            console.print("[cyan]Launching SGLang server...[/cyan]")
            if server.start():
                sglang_managed = server
            else:
                console.print("[yellow]SGLang failed -- falling back to local HF[/yellow]")
                use_sglang = False

    scorer = ReviewScorer(
        model_name=model_name,
        device="auto",
        max_new_tokens=cfg.model.large.max_new_tokens,
        temperature=cfg.model.large.temperature,
        use_sglang=use_sglang,
        sglang_url=sglang_url,
    )

    all_results = {}
    for team_name, team in simulator.teams.items():
        console.print(f"\n[cyan]A/B test: {team_name}[/cyan]")
        result = run_ab_test(
            team_name=team_name,
            team_description=team.description,
            train_samples=team.train_samples,
            test_samples=team.test_samples,
            scorer=scorer,
            baseline_cls=EmbeddingFilter,
            baseline_kwargs=baseline_kwargs,
            n_seeds=args.seeds,
            base_seed=cfg.project.seed,
        )
        all_results[team_name] = result

    # Print results table
    console.print()
    table = Table(title="A/B Test Results (F1)")
    table.add_column("Team", style="cyan")
    table.add_column("RL F1", style="blue")
    table.add_column("Baseline F1", style="red")
    table.add_column("Delta", style="yellow")
    table.add_column("95% CI", style="dim")
    table.add_column("p-value", style="magenta")
    table.add_column("Winner", style="bold")

    for team_name, result in all_results.items():
        f1 = result["metrics"].get("f1", {})
        if not f1:
            continue
        rl_f1 = f1["rl_mean"]
        bl_f1 = f1["baseline_mean"]
        delta = f1["mean_delta"]
        ci = f1["ci_95"]
        pval = f1["p_value"]
        winner = f1["winner"]

        sig = ""
        if pval < 0.01:
            sig = " **"
        elif pval < 0.05:
            sig = " *"

        winner_str = f"{winner.upper()}{sig}" if winner != "tie" else "-"

        table.add_row(
            team_name.title(),
            f"{rl_f1:.3f}",
            f"{bl_f1:.3f}",
            f"{delta:+.3f}",
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            f"{pval:.4f}",
            winner_str,
        )

    console.print(table)

    # Summary line
    f1_deltas = []
    for r in all_results.values():
        f1 = r["metrics"].get("f1", {})
        if f1:
            f1_deltas.append(f1["mean_delta"])
    if f1_deltas:
        avg_delta = sum(f1_deltas) / len(f1_deltas)
        n_wins = sum(1 for r in all_results.values() if r["metrics"].get("f1", {}).get("winner") == "rl")
        console.print(
            f"\n  Average F1 delta (RL - Baseline): {avg_delta:+.3f} | "
            f"RL wins: {n_wins}/{len(all_results)} teams"
        )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "ab_test_results.json"

    serializable = {}
    for team, res in all_results.items():
        serializable[team] = {
            "team": res["team"],
            "n_pairs": res["n_pairs"],
            "metrics": {},
        }
        for m, vals in res["metrics"].items():
            serializable[team]["metrics"][m] = {
                k: v for k, v in vals.items()
                if k not in ("rl_values", "baseline_values")
            }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    console.print(f"\n  Results saved: {out_path}")

    if sglang_managed is not None:
        sglang_managed.stop()

    console.print(f"\n[bold green]Step 6b complete![/bold green]")


if __name__ == "__main__":
    main()
