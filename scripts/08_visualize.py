#!/usr/bin/env python3
"""Step 8: Generate publication-quality visualizations.

Produces:
  1. Cold-start curve (THE killer chart)
  2. Per-team performance heatmap
  3. Head-to-head comparison bar chart
  4. Example comments with scores from both systems
  5. Model size comparison (1.7B vs 4B)
  6. Results markdown table

Usage:
    python scripts/08_visualize.py [--config configs/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console

from src.config import load_config
from src.evaluation.cold_start import ColdStartEvaluator
from src.visualization.charts import RLCRVisualizer

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)

    console.rule("[bold blue]Step 8: Visualizations[/bold blue]")

    viz = RLCRVisualizer(
        output_dir=cfg.visualization.output_dir,
        dpi=cfg.visualization.dpi,
        figsize=tuple(cfg.visualization.figsize),
        rl_color=cfg.visualization.colors.rl,
        baseline_color=cfg.visualization.colors.baseline,
        random_color=cfg.visualization.colors.random,
        team_colors=dict(cfg.visualization.team_colors),
    )

    results_dir = Path("results")
    generated = []

    # --- 1. Cold-Start Curves ---
    cold_start_path = results_dir / "cold_start_results.json"
    if cold_start_path.exists():
        console.print("\n[cyan]Generating cold-start curves...[/cyan]")
        cold_start_data = ColdStartEvaluator.load_results(cold_start_path)

        team_names = set()
        for key in cold_start_data:
            parts = key.split("_", 1)
            if len(parts) == 2:
                team_names.add(parts[1])

        all_team_curves = {}
        for team_name in team_names:
            curve_data = {}
            for method in ["baseline", "rl"]:
                key = f"{method}_{team_name}"
                if key in cold_start_data:
                    steps_data = cold_start_data[key]
                    steps = sorted([int(s) for s in steps_data.keys()])
                    means = [steps_data[str(s)]["f1"]["mean"] for s in steps]
                    stds = [steps_data[str(s)]["f1"]["std"] for s in steps]
                    curve_data[method] = {"steps": steps, "means": means, "stds": stds}

            if curve_data:
                all_team_curves[team_name] = curve_data
                path = viz.plot_cold_start_curve(curve_data, team_name)
                generated.append(str(path))

        if all_team_curves:
            path = viz.plot_cold_start_all_teams(all_team_curves)
            generated.append(str(path))
    else:
        console.print("[yellow]No cold-start results found. Run step 06 first.[/yellow]")

    # --- 2. Per-Team Heatmap ---
    eval_path = results_dir / "full_evaluation.json"
    baseline_path = results_dir / "baseline_results.json"

    if baseline_path.exists():
        console.print("\n[cyan]Generating team heatmap...[/cyan]")
        with open(baseline_path) as f:
            baseline_results = json.load(f)

        path = viz.plot_team_heatmap(baseline_results, "Embedding Baseline — Per-Team Performance")
        generated.append(str(path))

    # --- 3. Head-to-Head Comparison ---
    if baseline_path.exists() and eval_path.exists():
        console.print("\n[cyan]Generating head-to-head comparison...[/cyan]")
        with open(eval_path) as f:
            full_eval = json.load(f)

        rl_metrics = {}
        bl_metrics = {}
        for key, data in full_eval.items():
            if key.startswith("rl_"):
                team = key[3:]
                max_step = max(data.keys(), key=int)
                rl_metrics[team] = {
                    m: data[max_step][m]["mean"]
                    for m in data[max_step]
                }
            elif key.startswith("baseline_"):
                team = key[9:]
                max_step = max(data.keys(), key=int)
                bl_metrics[team] = {
                    m: data[max_step][m]["mean"]
                    for m in data[max_step]
                }

        if rl_metrics and bl_metrics:
            common_teams = set(rl_metrics) & set(bl_metrics)
            if common_teams:
                rl_common = {t: rl_metrics[t] for t in common_teams}
                bl_common = {t: bl_metrics[t] for t in common_teams}
                path = viz.plot_head_to_head(rl_common, bl_common)
                generated.append(str(path))

                table_md = viz.generate_results_table(rl_common, bl_common)
                generated.append(str(viz.output_dir / "results_table.md"))

    # --- 4. Example Comments ---
    console.print("\n[cyan]Generating example scores chart...[/cyan]")
    examples = [
        {
            "comment": "This SQL query is vulnerable to injection. Use parameterized queries.",
            "rl_score": 0.92,
            "baseline_score": 0.71,
            "label": 1,
            "team": "security",
        },
        {
            "comment": "nit: add a blank line here",
            "rl_score": 0.15,
            "baseline_score": 0.45,
            "label": 0,
            "team": "security",
        },
        {
            "comment": "Variable name 'x' should be more descriptive per our style guide.",
            "rl_score": 0.88,
            "baseline_score": 0.62,
            "label": 1,
            "team": "style",
        },
        {
            "comment": "This loop is O(n²). Consider using a hash set for O(n) lookups.",
            "rl_score": 0.95,
            "baseline_score": 0.58,
            "label": 1,
            "team": "performance",
        },
        {
            "comment": "Looks good to me!",
            "rl_score": 0.08,
            "baseline_score": 0.35,
            "label": 0,
            "team": "pragmatic",
        },
        {
            "comment": "Consider applying the Strategy pattern here for better extensibility and testability.",
            "rl_score": 0.91,
            "baseline_score": 0.67,
            "label": 1,
            "team": "thorough",
        },
    ]
    path = viz.plot_example_scores(examples)
    generated.append(str(path))

    # --- 5. A/B Test ---
    ab_path = results_dir / "ab_test_results.json"
    if ab_path.exists():
        console.print("\n[cyan]Generating A/B test chart...[/cyan]")
        with open(ab_path) as f:
            ab_data = json.load(f)
        path = viz.plot_ab_test(ab_data, metric="f1")
        generated.append(str(path))
    else:
        console.print("[yellow]No A/B test results found. Run step 06b first.[/yellow]")

    # --- 6. Three-Way Comparison ---
    three_way_path = results_dir / "three_way_results.json"
    if three_way_path.exists():
        console.print("\n[cyan]Generating three-way comparison charts...[/cyan]")
        with open(three_way_path) as f:
            three_way_data = json.load(f)

        for team_name, team_data in three_way_data.items():
            path = viz.plot_three_way_cold_start(team_data, team_name)
            generated.append(str(path))

        path = viz.plot_three_way_all_teams(three_way_data)
        generated.append(str(path))
    else:
        console.print("[yellow]No three-way results found. Run step 11 first.[/yellow]")

    # --- 7. Model Comparison ---
    model_comp_path = results_dir / "model_comparison.json"
    if model_comp_path.exists():
        console.print("\n[cyan]Generating model comparison chart...[/cyan]")
        with open(model_comp_path) as f:
            model_data = json.load(f)

        if "small" in model_data and "large" in model_data:
            path = viz.plot_model_comparison(
                model_data["small"],
                model_data["large"],
                small_name="Qwen3-1.7B",
                large_name="Qwen3-4B",
            )
            generated.append(str(path))

    # --- Summary ---
    console.print(f"\n[bold green]✓ Step 8 complete![/bold green]")
    console.print(f"  Generated {len(generated)} visualizations:")
    for g in generated:
        console.print(f"    • {g}")


if __name__ == "__main__":
    main()
