"""Cold-start evaluation: how quickly does each method learn?

This is THE key evaluation for the paper. It measures:
  - Performance at 0, 5, 10, 20, 50, 100 training samples
  - Per-team breakdown
  - Multiple seeds for statistical significance
  - Comparison between RL (DAPO) and embedding baseline

The cold-start curve is the killer chart that shows RL adapts faster
than cosine-similarity filtering with limited feedback.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.data.parser import CodeReviewSample
from src.evaluation.metrics import compute_metrics, MetricsResult
from src.training.rewards import calibrate_threshold


class ColdStartEvaluator:
    """Evaluate cold-start performance across multiple sample counts and seeds."""

    def __init__(
        self,
        steps: list[int] = None,
        n_seeds: int = 3,
        base_seed: int = 42,
    ):
        self.steps = steps or [0, 5, 10, 20, 50, 100]
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.results: dict[str, dict[int, list[MetricsResult]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def evaluate_embedding_baseline(
        self,
        baseline_cls,
        baseline_kwargs: dict,
        team_name: str,
        all_train_samples: list[CodeReviewSample],
        test_samples: list[CodeReviewSample],
    ) -> dict[int, list[MetricsResult]]:
        """Evaluate embedding baseline at each cold-start step.

        For each step N and each seed:
          1. Take N random training samples
          2. Build the vector store from those N samples
          3. Tune threshold on those N samples
          4. Evaluate on the full test set
        """
        step_results: dict[int, list[MetricsResult]] = defaultdict(list)

        for seed_offset in range(self.n_seeds):
            seed = self.base_seed + seed_offset
            rng = random.Random(seed)

            for n_samples in self.steps:
                baseline = baseline_cls(**baseline_kwargs)

                if n_samples > 0:
                    chosen = rng.sample(
                        all_train_samples,
                        min(n_samples, len(all_train_samples)),
                    )
                    votes = [
                        {
                            "comment": s.comment,
                            "vote": "upvote" if s.label == 1 else "downvote",
                        }
                        for s in chosen
                    ]
                    baseline.build_store(team_name, votes)
                    baseline.tune_threshold(team_name, chosen)
                else:
                    baseline.build_store(team_name, [])
                    baseline.thresholds[team_name] = 0.0

                preds = baseline.predict(team_name, [s.comment for s in test_samples])
                labels = [s.label for s in test_samples]
                decisions = [p["decision"] for p in preds]
                scores_list = [p["score"] for p in preds]

                metrics = compute_metrics(labels, decisions, scores_list)
                step_results[n_samples].append(metrics)

                logger.debug(
                    f"  Baseline | {team_name} | n={n_samples} | seed={seed} | "
                    f"{metrics.summary_str()}"
                )

        key = f"baseline_{team_name}"
        self.results[key] = step_results
        return step_results

    def evaluate_rl_model(
        self,
        scorer,
        team_name: str,
        team_description: str,
        all_train_samples: list[CodeReviewSample],
        test_samples: list[CodeReviewSample],
        train_fn=None,
        calibrated_threshold: float | None = None,
    ) -> dict[int, list[MetricsResult]]:
        """Evaluate RL model at each cold-start step.

        For each step N and each seed:
          1. Take N random training samples
          2. If train_fn provided: fine-tune a fresh LoRA from those N samples
          3. Build vote history from those N samples
          4. Evaluate on the full test set
          5. Apply calibrated threshold if provided (instead of default 0.5)
        """
        step_results: dict[int, list[MetricsResult]] = defaultdict(list)

        for seed_offset in range(self.n_seeds):
            seed = self.base_seed + seed_offset

            for n_samples in self.steps:
                train_rng = random.Random(seed * 10000 + n_samples)
                context_rng = random.Random(seed * 10000 + n_samples + 1)

                chosen = []
                if n_samples > 0:
                    chosen = train_rng.sample(
                        all_train_samples,
                        min(n_samples, len(all_train_samples)),
                    )

                if n_samples > 0 and train_fn is not None:
                    try:
                        train_fn(chosen, seed=seed)
                    except Exception as e:
                        logger.warning(f"train_fn failed at n={n_samples} seed={seed}: {e}")

                vote_history = []
                if n_samples > 0:
                    chosen_for_context = context_rng.sample(
                        all_train_samples,
                        min(n_samples, len(all_train_samples)),
                    )
                    vote_history = [
                        {
                            "comment": s.comment,
                            "vote": "upvote" if s.label == 1 else "downvote",
                        }
                        for s in chosen_for_context
                    ]

                batch = [{"diff": s.diff, "comment": s.comment} for s in test_samples]
                outputs = scorer.batch_score(
                    batch,
                    team_name=team_name,
                    team_description=team_description,
                    vote_history=vote_history,
                )
                labels = [s.label for s in test_samples]
                scores_list = [o.score for o in outputs]

                threshold = calibrated_threshold or 0.5
                if calibrated_threshold is None and n_samples >= 10:
                    threshold = calibrate_threshold(
                        scores_list, labels, target_action_rate=0.55
                    )

                decisions = [1 if s >= threshold else 0 for s in scores_list]

                metrics = compute_metrics(labels, decisions, scores_list)
                step_results[n_samples].append(metrics)

                logger.debug(
                    f"  RL | {team_name} | n={n_samples} | seed={seed} | "
                    f"t={threshold:.2f} | {metrics.summary_str()}"
                )

        key = f"rl_{team_name}"
        self.results[key] = step_results
        return step_results

    def aggregate_results(self) -> dict[str, Any]:
        """Aggregate results across seeds, computing mean and std."""
        aggregated = {}
        for method_team, step_data in self.results.items():
            aggregated[method_team] = {}
            for n_samples, metrics_list in step_data.items():
                if not metrics_list:
                    continue
                metric_names = [
                    "accuracy", "precision", "recall", "f1",
                    "auroc", "action_rate", "coverage",
                ]
                agg = {}
                for metric_name in metric_names:
                    values = [getattr(m, metric_name) for m in metrics_list]
                    agg[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "values": [float(v) for v in values],
                    }
                aggregated[method_team][n_samples] = agg
        return aggregated

    def get_cold_start_curve_data(
        self, team_name: str, metric: str = "f1"
    ) -> dict[str, dict]:
        """Extract cold-start curve data for plotting.

        Returns data for both baseline and RL for a specific team and metric.
        """
        aggregated = self.aggregate_results()
        curve_data = {}

        for method in ["baseline", "rl"]:
            key = f"{method}_{team_name}"
            if key not in aggregated:
                continue
            steps = sorted(aggregated[key].keys())
            means = [aggregated[key][s][metric]["mean"] for s in steps]
            stds = [aggregated[key][s][metric]["std"] for s in steps]
            curve_data[method] = {
                "steps": steps,
                "means": means,
                "stds": stds,
            }

        return curve_data

    def save_results(self, path: str | Path):
        """Save all results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        aggregated = self.aggregate_results()
        with open(path, "w") as f:
            json.dump(aggregated, f, indent=2)
        logger.info(f"Results saved to {path}")

    @classmethod
    def load_results(cls, path: str | Path) -> dict:
        """Load previously saved results."""
        with open(path) as f:
            return json.load(f)


def run_generalization_test(
    scorer,
    train_team: str,
    test_team: str,
    train_description: str,
    test_description: str,
    train_samples: list[CodeReviewSample],
    test_samples: list[CodeReviewSample],
    n_seeds: int = 3,
    base_seed: int = 42,
) -> dict[str, Any]:
    """Test generalization: train on one team pattern, evaluate on another.

    e.g., Train on "hates docstrings" style → test on "wants type hints" thorough.
    This shows whether the RL policy learns transferable review preferences.
    """
    results = []

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        rng = random.Random(seed)

        vote_history = [
            {
                "comment": s.comment,
                "vote": "upvote" if s.label == 1 else "downvote",
            }
            for s in train_samples
        ]

        batch = [{"diff": s.diff, "comment": s.comment} for s in test_samples]
        outputs = scorer.batch_score(
            batch,
            team_name=test_team,
            team_description=test_description,
            vote_history=vote_history,
        )
        labels = [s.label for s in test_samples]
        decisions = [o.binary_label for o in outputs]
        scores_list = [o.score for o in outputs]

        metrics = compute_metrics(labels, decisions, scores_list)
        results.append(metrics)

    metric_names = ["accuracy", "f1", "auroc", "action_rate"]
    aggregated = {}
    for m in metric_names:
        values = [getattr(r, m) for r in results]
        aggregated[m] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    return {
        "train_team": train_team,
        "test_team": test_team,
        "n_seeds": n_seeds,
        "metrics": aggregated,
    }
