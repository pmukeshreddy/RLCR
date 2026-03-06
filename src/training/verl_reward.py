"""veRL-compatible reward function wrapping CodeReviewReward.

veRL reward functions receive:
    data_source: str — dataset identifier
    solution_str: str — decoded model response
    ground_truth: str — ground truth value (we store label here)
    extra_info: dict — optional metadata

Returns a float score. veRL's RewardManager calls this per-sample.
"""
from __future__ import annotations

from src.training.rewards import CodeReviewReward

_scorer = None


def _get_scorer() -> CodeReviewReward:
    global _scorer
    if _scorer is None:
        _scorer = CodeReviewReward(
            overlong_penalty=1.0,
            overlong_buffer_len=32,
            max_completion_length=128,
        )
    return _scorer


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Reward function with veRL's expected signature.

    ground_truth encodes the binary label as "0" or "1".
    extra_info may contain overlong_penalty / max_completion_length overrides.
    """
    scorer = _get_scorer()
    label = int(float(ground_truth))
    completion = [{"role": "assistant", "content": solution_str}]
    rewards = scorer(completions=[completion], label=[label])
    return rewards[0]
