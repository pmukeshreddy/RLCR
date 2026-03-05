"""Publication-quality visualizations for RLCR results.

Generates:
  1. Cold-start curve (THE killer chart)
  2. Per-team performance heatmap
  3. Head-to-head comparison table
  4. Example comments with scores
  5. Calibration plots
  6. Model size comparison (1.7B vs 4B)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from loguru import logger


class RLCRVisualizer:
    """Generate all publication-quality figures for RLCR."""

    def __init__(
        self,
        output_dir: str = "./results/figures",
        dpi: int = 300,
        figsize: tuple[int, int] = (10, 6),
        style: str = "seaborn-v0_8-whitegrid",
        rl_color: str = "#2196F3",
        baseline_color: str = "#FF5722",
        random_color: str = "#9E9E9E",
        team_colors: dict[str, str] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        self.rl_color = rl_color
        self.baseline_color = baseline_color
        self.random_color = random_color
        self.team_colors = team_colors or {
            "security": "#E53935",
            "style": "#8E24AA",
            "performance": "#FF8F00",
            "pragmatic": "#43A047",
            "thorough": "#1E88E5",
        }

        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-v0_8")

    def plot_cold_start_curve(
        self,
        curve_data: dict[str, dict],
        team_name: str = "all",
        metric: str = "f1",
        title: str | None = None,
    ) -> Path:
        """THE killer chart: cold-start performance curve.

        Shows how quickly RL adapts vs embedding baseline as training
        samples increase from 0 to 100.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for method, data in curve_data.items():
            steps = data["steps"]
            means = np.array(data["means"])
            stds = np.array(data["stds"])

            color = self.rl_color if method == "rl" else self.baseline_color
            label = "DAPO (Ours)" if method == "rl" else "Embedding Baseline"
            marker = "o" if method == "rl" else "s"

            ax.plot(steps, means, color=color, marker=marker, linewidth=2.5,
                    markersize=8, label=label, zorder=3)
            ax.fill_between(steps, means - stds, means + stds,
                            alpha=0.15, color=color, zorder=2)

        random_baseline = 0.5
        ax.axhline(y=random_baseline, color=self.random_color, linestyle="--",
                    linewidth=1.5, label="Random Baseline", alpha=0.7)

        ax.set_xlabel("Number of Training Samples (Team Feedback)", fontsize=13)
        ax.set_ylabel(f"{metric.upper()} Score", fontsize=13)
        ax.set_title(
            title or f"Cold-Start Learning Curve — {team_name.title()} Team",
            fontsize=15, fontweight="bold", pad=15,
        )
        ax.legend(fontsize=11, loc="lower right", framealpha=0.9)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3)

        ax.annotate(
            "RL adapts faster\nwith few samples",
            xy=(10, curve_data.get("rl", {}).get("means", [0.6])[2] if "rl" in curve_data else 0.6),
            xytext=(30, 0.45),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#333"),
            color="#333",
            style="italic",
        )

        plt.tight_layout()
        path = self.output_dir / f"cold_start_{team_name}_{metric}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_cold_start_all_teams(
        self,
        all_curve_data: dict[str, dict[str, dict]],
        metric: str = "f1",
    ) -> Path:
        """Cold-start curves for all teams in a single multi-panel figure."""
        teams = list(all_curve_data.keys())
        n_teams = len(teams)
        n_cols = min(3, n_teams)
        n_rows = (n_teams + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, team_name in enumerate(teams):
            ax = axes[idx]
            curve = all_curve_data[team_name]
            team_color = self.team_colors.get(team_name, "#333")

            for method, data in curve.items():
                steps = data["steps"]
                means = np.array(data["means"])
                stds = np.array(data["stds"])

                if method == "rl":
                    ax.plot(steps, means, color=self.rl_color, marker="o",
                            linewidth=2, markersize=5, label="DAPO")
                    ax.fill_between(steps, means - stds, means + stds,
                                    alpha=0.15, color=self.rl_color)
                else:
                    ax.plot(steps, means, color=self.baseline_color, marker="s",
                            linewidth=2, markersize=5, label="Embedding")
                    ax.fill_between(steps, means - stds, means + stds,
                                    alpha=0.15, color=self.baseline_color)

            ax.axhline(y=0.5, color=self.random_color, linestyle="--",
                        linewidth=1, alpha=0.5)
            ax.set_title(team_name.title(), fontsize=12, fontweight="bold",
                         color=team_color)
            ax.set_ylim(0.3, 1.0)
            ax.grid(True, alpha=0.2)
            if idx == 0:
                ax.legend(fontsize=8)

        for idx in range(n_teams, len(axes)):
            axes[idx].set_visible(False)

        fig.supxlabel("Training Samples", fontsize=13, y=0.02)
        fig.supylabel(f"{metric.upper()}", fontsize=13, x=0.02)
        fig.suptitle("Cold-Start Performance Across Teams",
                     fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()

        path = self.output_dir / f"cold_start_all_teams_{metric}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_team_heatmap(
        self,
        team_metrics: dict[str, dict[str, float]],
        title: str = "Per-Team Performance Heatmap",
    ) -> Path:
        """Heatmap showing per-team, per-metric performance."""
        teams = list(team_metrics.keys())
        metrics = ["accuracy", "precision", "recall", "f1", "auroc", "action_rate"]
        available_metrics = [m for m in metrics if m in list(team_metrics.values())[0]]

        data = np.zeros((len(teams), len(available_metrics)))
        for i, team in enumerate(teams):
            for j, metric in enumerate(available_metrics):
                data[i, j] = team_metrics[team].get(metric, 0)

        fig, ax = plt.subplots(figsize=(max(8, len(available_metrics) * 1.5), max(5, len(teams) * 0.8)))
        sns.heatmap(
            data,
            annot=True,
            fmt=".3f",
            xticklabels=[m.upper() for m in available_metrics],
            yticklabels=[t.title() for t in teams],
            cmap="RdYlGn",
            vmin=0.3,
            vmax=1.0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Score"},
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()

        path = self.output_dir / "team_heatmap.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_head_to_head(
        self,
        rl_metrics: dict[str, dict[str, float]],
        baseline_metrics: dict[str, dict[str, float]],
        metric: str = "f1",
    ) -> Path:
        """Bar chart comparing RL vs baseline per team."""
        teams = list(rl_metrics.keys())
        x = np.arange(len(teams))
        width = 0.35

        rl_vals = [rl_metrics[t].get(metric, 0) for t in teams]
        bl_vals = [baseline_metrics[t].get(metric, 0) for t in teams]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars1 = ax.bar(x - width / 2, rl_vals, width, label="DAPO (Ours)",
                       color=self.rl_color, alpha=0.85, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, bl_vals, width, label="Embedding Baseline",
                       color=self.baseline_color, alpha=0.85, edgecolor="white", linewidth=0.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                )

        ax.set_ylabel(f"{metric.upper()} Score", fontsize=13)
        ax.set_title(f"Head-to-Head: DAPO vs Embedding Baseline ({metric.upper()})",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([t.title() for t in teams], fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / f"head_to_head_{metric}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_example_scores(
        self,
        examples: list[dict],
        title: str = "Example Comments with Scores",
    ) -> Path:
        """Visual comparison of RL vs baseline scores on example comments."""
        fig, ax = plt.subplots(figsize=(12, max(4, len(examples) * 0.8)))

        y_positions = np.arange(len(examples))
        height = 0.3

        for i, ex in enumerate(examples):
            comment = ex["comment"][:80] + ("..." if len(ex["comment"]) > 80 else "")
            rl_score = ex.get("rl_score", 0.5)
            bl_score = ex.get("baseline_score", 0.5)
            label = ex.get("label", -1)

            ax.barh(i + height / 2, rl_score, height, color=self.rl_color,
                    alpha=0.85, label="DAPO" if i == 0 else "")
            ax.barh(i - height / 2, bl_score, height, color=self.baseline_color,
                    alpha=0.85, label="Embedding" if i == 0 else "")

            bg_color = "#E8F5E9" if label == 1 else "#FFEBEE" if label == 0 else "#FFF"
            ax.annotate(
                comment,
                xy=(0, i),
                xytext=(-0.02, i),
                fontsize=8,
                ha="right",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.8),
            )

        ax.set_xlabel("Relevance Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlim(-0.5, 1.1)
        ax.set_yticks([])
        ax.legend(fontsize=10, loc="lower right")
        ax.axvline(x=0.5, color=self.random_color, linestyle="--", alpha=0.5)
        ax.grid(True, axis="x", alpha=0.3)

        legend_elements = [
            mpatches.Patch(facecolor="#E8F5E9", label="Addressed (ground truth)"),
            mpatches.Patch(facecolor="#FFEBEE", label="Ignored (ground truth)"),
        ]
        ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][:2],
                  labels=["Addressed", "Ignored", "DAPO", "Embedding"],
                  fontsize=9, loc="lower right")

        plt.tight_layout()
        path = self.output_dir / "example_scores.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_model_comparison(
        self,
        small_metrics: dict[str, dict[str, float]],
        large_metrics: dict[str, dict[str, float]],
        small_name: str = "Qwen3-1.7B",
        large_name: str = "Qwen3-4B",
    ) -> Path:
        """Compare model sizes (1.7B vs 4B)."""
        teams = list(small_metrics.keys())
        metrics = ["f1", "auroc", "accuracy"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            x = np.arange(len(teams))
            width = 0.35
            small_vals = [small_metrics[t].get(metric, 0) for t in teams]
            large_vals = [large_metrics[t].get(metric, 0) for t in teams]

            ax.bar(x - width / 2, small_vals, width, label=small_name,
                   color="#64B5F6", alpha=0.85)
            ax.bar(x + width / 2, large_vals, width, label=large_name,
                   color="#1565C0", alpha=0.85)

            ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([t.title() for t in teams], rotation=45, fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle("Model Size Comparison", fontsize=15, fontweight="bold")
        plt.tight_layout()

        path = self.output_dir / "model_comparison.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def generate_results_table(
        self,
        rl_metrics: dict[str, dict[str, float]],
        baseline_metrics: dict[str, dict[str, float]],
    ) -> str:
        """Generate a markdown comparison table."""
        teams = list(rl_metrics.keys())
        metrics = ["accuracy", "f1", "auroc", "action_rate"]

        header = "| Team | " + " | ".join(
            f"RL {m.upper()} | BL {m.upper()}" for m in metrics
        ) + " |"
        sep = "|" + "|".join(["---"] * (1 + 2 * len(metrics))) + "|"

        rows = [header, sep]
        for team in teams:
            row = f"| {team.title()} |"
            for metric in metrics:
                rl_val = rl_metrics[team].get(metric, 0)
                bl_val = baseline_metrics[team].get(metric, 0)
                rl_str = f" **{rl_val:.3f}**" if rl_val > bl_val else f" {rl_val:.3f}"
                bl_str = f" **{bl_val:.3f}**" if bl_val > rl_val else f" {bl_val:.3f}"
                row += f"{rl_str} |{bl_str} |"
            rows.append(row)

        table = "\n".join(rows)

        table_path = self.output_dir / "results_table.md"
        table_path.write_text(table)
        logger.info(f"Saved: {table_path}")
        return table

    def plot_ab_test(
        self,
        ab_results: dict[str, dict],
        metric: str = "f1",
    ) -> Path:
        """Bar chart of A/B test results with error bars and significance stars.

        Args:
            ab_results: {team_name: {"metrics": {metric: {rl_mean, baseline_mean, ...}}}}
        """
        plt.style.use(self.style if self.style != "seaborn-whitegrid" else "default")

        teams = list(ab_results.keys())
        x = np.arange(len(teams))
        width = 0.35

        rl_means = []
        bl_means = []
        rl_cis = []
        bl_cis = []
        p_values = []

        for t in teams:
            m = ab_results[t]["metrics"].get(metric, {})
            rl_means.append(m.get("rl_mean", 0))
            bl_means.append(m.get("baseline_mean", 0))
            rl_cis.append(m.get("rl_std", 0))
            bl_cis.append(m.get("baseline_std", 0))
            p_values.append(m.get("p_value", 1.0))

        fig, ax = plt.subplots(figsize=self.figsize)
        bars1 = ax.bar(
            x - width / 2, rl_means, width,
            yerr=rl_cis, capsize=4,
            label="DAPO (Ours)", color=self.rl_color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2, bl_means, width,
            yerr=bl_cis, capsize=4,
            label="Embedding Baseline", color=self.baseline_color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

        for i, pval in enumerate(p_values):
            top = max(rl_means[i] + rl_cis[i], bl_means[i] + bl_cis[i])
            if pval < 0.01:
                ax.text(x[i], top + 0.02, "**", ha="center", fontsize=14, fontweight="bold")
            elif pval < 0.05:
                ax.text(x[i], top + 0.02, "*", ha="center", fontsize=14, fontweight="bold")

        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(
            f"A/B Test: DAPO vs Embedding Baseline ({metric.upper()})",
            fontsize=14, fontweight="bold", pad=15,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([t.title() for t in teams], fontsize=11)
        ax.set_ylim(0, min(max(max(rl_means), max(bl_means)) + 0.15, 1.1))
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

        sig_patch = mpatches.Patch(color="none", label="* p<0.05  ** p<0.01")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(sig_patch)
        ax.legend(handles=handles, fontsize=10, loc="upper right")

        plt.tight_layout()
        path = self.output_dir / f"ab_test_{metric}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path
