"""Reward functions for DAPO training on code review filtering.

Pure binary reward: +1 if the model's prediction matches ground truth,
-1 otherwise. With DAPO overlong reward shaping for completions that
exceed the expected length.

The model outputs a score (0.0 to 1.0). Prediction = SURFACE if > 0.5.
"""

from __future__ import annotations

import re

from loguru import logger


def _extract_score(text: str) -> float:
    """Extract a numeric score from model output.

    Tries patterns in order: bare float, <score> tags, any number 0-1.
    Returns 0.5 on failure (maps to FILTER at the boundary).
    """
    text = text.strip()
    try:
        val = float(text)
        return max(0.0, min(1.0, val))
    except ValueError:
        pass

    m = re.search(r"<score>\s*([\d.]+)\s*</score>", text)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1))))
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


class CodeReviewReward:
    """Pure binary reward for code review filtering.

    +1 if predicted label matches ground truth, -1 otherwise.
    Plus DAPO overlong reward shaping.
    """

    def __init__(
        self,
        overlong_penalty: float = 1.0,
        overlong_buffer_len: int = 16,
        max_completion_length: int = 32,
    ):
        self.overlong_penalty = overlong_penalty
        self.overlong_buffer_len = overlong_buffer_len
        self.max_completion_length = max_completion_length

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
        for completion, lbl in zip(completions, label):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            rewards.append(self._score_single(text, lbl))
        return rewards

    def _score_single(self, text: str, label: int) -> float:
        score = _extract_score(text)
        predicted = 1 if score > 0.5 else 0
        reward = 1.0 if predicted == label else -1.0

        if self.overlong_penalty > 0 and self.overlong_buffer_len > 0:
            text_len = len(text.split())
            threshold = self.max_completion_length - self.overlong_buffer_len
            if threshold > 0 and text_len > threshold:
                exceed = text_len - threshold
                penalty = min(exceed / self.overlong_buffer_len, 1.0) * self.overlong_penalty
                reward -= penalty

        return reward


def correctness_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Binary correctness reward (same as CodeReviewReward but standalone)."""
    label = kwargs.get("label", [0] * len(completions))
    rewards = []
    for completion, lbl in zip(completions, label):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        score = _extract_score(text)
        predicted = 1 if score > 0.5 else 0
        rewards.append(1.0 if predicted == lbl else -1.0)
    return rewards


def format_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Reward for outputting a clean numeric score."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        text = text.strip()
        try:
            val = float(text)
            rewards.append(1.0 if 0.0 <= val <= 1.0 else 0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards
