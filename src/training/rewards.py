"""Reward functions for DAPO training on code review filtering.

Two-component reward:
  1. Shaped correctness (weight 0.8): distance-based, rewards calibration.
     reward = (1 - |score - label|) * 2 - 1  → maps to [-1, +1]
  2. Format compliance (weight 0.2): did the model produce parseable
     <think>/<score>/<decision> tags? +1 for clean output, 0 for garbage.

Plus DAPO overlong reward shaping for completions exceeding expected length.
"""

from __future__ import annotations

import re

from loguru import logger


_SCORE_RE = re.compile(r"<score>\s*([\d.]+)\s*</score>")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DECISION_RE = re.compile(r"<decision>\s*(SURFACE|FILTER)\s*</decision>", re.IGNORECASE)


def _extract_score(text: str) -> float:
    """Extract a numeric score from model output.

    Tries: <score> tags first, then bare float, then any 0-1 number.
    Returns 0.5 on failure.
    """
    m = _SCORE_RE.search(text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass

    text_stripped = text.strip()
    try:
        val = float(text_stripped)
        return max(0.0, min(1.0, val))
    except ValueError:
        pass

    m = re.search(r"(\d\.\d+|\d+\.\d*|[01])", text)
    if m:
        try:
            val = float(m.group(1))
            if 0.0 <= val <= 1.0:
                return val
        except ValueError:
            pass

    return 0.5


def _check_format(text: str) -> float:
    """Check if output has proper structured format.

    Returns 1.0 if all three tags present (<think>, <score>, <decision>),
    0.5 if at least <score> is present, 0.0 otherwise.
    """
    has_think = bool(_THINK_RE.search(text))
    has_score = bool(_SCORE_RE.search(text))
    has_decision = bool(_DECISION_RE.search(text))

    if has_think and has_score and has_decision:
        return 1.0
    if has_score:
        return 0.5
    return 0.0


class CodeReviewReward:
    """Two-component reward for code review filtering.

    Shaped correctness (0.8): rewards calibrated predictions.
    Format compliance (0.2): rewards structured output.
    Plus DAPO overlong penalty.
    """

    CORRECTNESS_WEIGHT = 0.8
    FORMAT_WEIGHT = 0.2

    def __init__(
        self,
        overlong_penalty: float = 1.0,
        overlong_buffer_len: int = 64,
        max_completion_length: int = 256,
        action_rate_target: float = 0.55,
        action_rate_penalty: float = 0.3,
    ):
        self.overlong_penalty = overlong_penalty
        self.overlong_buffer_len = overlong_buffer_len
        self.max_completion_length = max_completion_length
        self.action_rate_target = action_rate_target
        self.action_rate_penalty = action_rate_penalty
        self._recent_scores: list[float] = []

    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[float]:
        """Compute rewards for a batch of completions.

        Compatible with TRL's GRPOTrainer reward function signature.
        """
        label = kwargs.get("label", [0] * len(completions))
        rewards = []
        batch_scores = []
        for completion, lbl in zip(completions, label):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            r, s = self._score_single(text, lbl)
            rewards.append(r)
            batch_scores.append(s)

        if self.action_rate_penalty > 0 and batch_scores:
            mean_score = sum(batch_scores) / len(batch_scores)
            drift = max(0.0, mean_score - self.action_rate_target)
            penalty = drift * self.action_rate_penalty
            rewards = [r - penalty for r in rewards]

        return rewards

    def _score_single(self, text: str, label: int) -> tuple[float, float]:
        """Returns (reward, extracted_score)."""
        score = _extract_score(text)

        correctness = (1.0 - abs(score - float(label))) * 2.0 - 1.0

        fmt = _check_format(text)
        format_reward = fmt * 2.0 - 1.0

        reward = (
            self.CORRECTNESS_WEIGHT * correctness
            + self.FORMAT_WEIGHT * format_reward
        )

        if self.overlong_penalty > 0 and self.overlong_buffer_len > 0:
            text_len = len(text.split())
            threshold = self.max_completion_length - self.overlong_buffer_len
            if threshold > 0 and text_len > threshold:
                exceed = text_len - threshold
                penalty = min(exceed / self.overlong_buffer_len, 1.0) * self.overlong_penalty
                reward -= penalty

        return reward, score


def calibrate_threshold(
    scores: list[float],
    labels: list[int],
    target_action_rate: float = 0.55,
) -> float:
    """Find the decision threshold that produces a target action rate.

    Searches over candidate thresholds and picks the one closest to
    the target action rate. Used as a post-training calibration step.
    """
    import numpy as np
    scores_arr = np.array(scores)
    best_t = 0.5
    best_diff = float("inf")
    for t in np.linspace(0.05, 0.95, 91):
        rate = float(np.mean(scores_arr >= t))
        diff = abs(rate - target_action_rate)
        if diff < best_diff:
            best_diff = diff
            best_t = float(t)
    return best_t


def correctness_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Shaped correctness reward (standalone, no format component)."""
    label = kwargs.get("label", [0] * len(completions))
    rewards = []
    for completion, lbl in zip(completions, label):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        score = _extract_score(text)
        rewards.append((1.0 - abs(score - float(lbl))) * 2.0 - 1.0)
    return rewards


def format_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Reward for outputting proper structured format."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        rewards.append(_check_format(text) * 2.0 - 1.0)
    return rewards
