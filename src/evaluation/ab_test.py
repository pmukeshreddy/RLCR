"""A/B split evaluation: within-team 50/50 paired comparison.

For each team, the test set is randomly split into two halves. Both
methods (DAPO RL and embedding baseline) are evaluated on both halves.
This is repeated across N random seeds, yielding 2*N paired observations
per team -- enough for a paired t-test with confidence intervals.

This is the fairest head-to-head comparison: same data distribution,
same label balance, statistical significance via paired testing.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from loguru import logger
from scipy import stats

from src.data.parser import CodeReviewSample
from src.evaluation.metrics import compute_metrics, MetricsResult


def _split_samples(
    samples: list[CodeReviewSample], seed: int
) -> tuple[list[CodeReviewSample], list[CodeReviewSample]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]


def _eval_baseline(
    baseline, team_name: str, train_samples: list[CodeReviewSample],
    test_samples: list[CodeReviewSample],
) -> MetricsResult:
    votes = [
        {"comment": s.comment, "vote": "upvote" if s.label == 1 else "downvote"}
        for s in train_samples
    ]
    baseline.build_store(team_name, votes)
    baseline.tune_threshold(team_name, train_samples)

    preds = baseline.predict(team_name, [s.comment for s in test_samples])
    return compute_metrics(
        labels=[s.label for s in test_samples],
        predictions=[p["decision"] for p in preds],
        scores=[p["score"] for p in preds],
    )


def _eval_rl(
    scorer, team_name: str, team_description: str,
    train_samples: list[CodeReviewSample],
    test_samples: list[CodeReviewSample],
) -> MetricsResult:
    vote_history = [
        {"comment": s.comment, "vote": "upvote" if s.label == 1 else "downvote"}
        for s in train_samples
    ]
    batch = [{"diff": s.diff, "comment": s.comment} for s in test_samples]
    outputs = scorer.batch_score(
        batch,
        team_name=team_name,
        team_description=team_description,
        vote_history=vote_history,
    )
    return compute_metrics(
        labels=[s.label for s in test_samples],
        predictions=[o.binary_label for o in outputs],
        scores=[o.score for o in outputs],
    )


def run_ab_test(
    team_name: str,
    team_description: str,
    train_samples: list[CodeReviewSample],
    test_samples: list[CodeReviewSample],
    scorer,
    baseline_cls,
    baseline_kwargs: dict,
    n_seeds: int = 5,
    base_seed: int = 42,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Run within-team A/B split evaluation.

    For each seed, splits test_samples 50/50 into halves A and B.
    Both RL and baseline are evaluated on both halves, giving 2 paired
    observations per seed (2 * n_seeds total).

    Returns a dict with per-metric results including:
      - rl_mean, baseline_mean, mean_delta
      - 95% confidence interval on the delta
      - p-value from a paired t-test
      - winner ("rl", "baseline", or "tie" at p < 0.05)
    """
    metrics = metrics or ["accuracy", "precision", "recall", "f1", "auroc"]

    rl_scores: dict[str, list[float]] = {m: [] for m in metrics}
    bl_scores: dict[str, list[float]] = {m: [] for m in metrics}

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        half_a, half_b = _split_samples(test_samples, seed)

        for half_label, half in [("A", half_a), ("B", half_b)]:
            if len(half) < 4:
                continue

            baseline = baseline_cls(**baseline_kwargs)
            bl_result = _eval_baseline(baseline, team_name, train_samples, half)

            rl_result = _eval_rl(
                scorer, team_name, team_description, train_samples, half
            )

            for m in metrics:
                rl_scores[m].append(getattr(rl_result, m))
                bl_scores[m].append(getattr(bl_result, m))

            logger.debug(
                f"  {team_name} seed={seed} half={half_label} | "
                f"RL F1={rl_result.f1:.3f} BL F1={bl_result.f1:.3f}"
            )

    results: dict[str, Any] = {"team": team_name, "n_pairs": len(rl_scores.get("f1", [])), "metrics": {}}

    for m in metrics:
        rl_arr = np.array(rl_scores[m])
        bl_arr = np.array(bl_scores[m])
        deltas = rl_arr - bl_arr

        n = len(deltas)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1)) if n > 1 else 0.0

        if n >= 2 and std_delta > 1e-12:
            t_stat, p_value = stats.ttest_rel(rl_arr, bl_arr)
            t_crit = stats.t.ppf(0.975, df=n - 1)
            se = std_delta / np.sqrt(n)
            ci_low = mean_delta - t_crit * se
            ci_high = mean_delta + t_crit * se
        else:
            p_value = 1.0
            ci_low = mean_delta
            ci_high = mean_delta

        if p_value < 0.05:
            winner = "rl" if mean_delta > 0 else "baseline"
        else:
            winner = "tie"

        results["metrics"][m] = {
            "rl_mean": float(np.mean(rl_arr)),
            "rl_std": float(np.std(rl_arr)),
            "baseline_mean": float(np.mean(bl_arr)),
            "baseline_std": float(np.std(bl_arr)),
            "mean_delta": mean_delta,
            "ci_95": [float(ci_low), float(ci_high)],
            "p_value": float(p_value),
            "winner": winner,
            "rl_values": rl_arr.tolist(),
            "baseline_values": bl_arr.tolist(),
        }

    return results
