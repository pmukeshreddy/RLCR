"""Reward functions for GRPO training on code review filtering.

The reward signal combines:
  1. Binary correctness: did the model's decision match ground truth?
  2. Calibration bonus: is the score well-calibrated with the outcome?
  3. Format compliance: did the model follow the output format?

This multi-signal reward encourages the model to not just get the decision
right, but to provide well-reasoned, well-calibrated scores.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger


class CodeReviewReward:
    """Reward computation for code review filtering.

    Used as the reward function in GRPO training. Receives model completions
    and ground-truth labels, returns per-completion reward scores.
    """

    def __init__(
        self,
        correctness_weight: float = 1.0,
        calibration_weight: float = 0.3,
        format_weight: float = 0.2,
        wrong_penalty: float = -0.5,
    ):
        self.correctness_weight = correctness_weight
        self.calibration_weight = calibration_weight
        self.format_weight = format_weight
        self.wrong_penalty = wrong_penalty

    def __call__(
        self,
        completions: list[list[dict[str, str]]],
        **kwargs,
    ) -> list[float]:
        """Compute rewards for a batch of completions.

        Compatible with TRL's GRPOTrainer reward function signature.
        TRL passes extra dataset columns (like 'label') through **kwargs.

        Args:
            completions: List of completion message lists from the model.
                         Each element is [{"role": "assistant", "content": "..."}]
            **kwargs: Must contain 'label' — ground-truth list (1=addressed, 0=ignored).

        Returns:
            List of reward floats, one per completion.
        """
        label = kwargs.get("label", [0] * len(completions))
        rewards = []
        for completion, lbl in zip(completions, label):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            reward = self._score_single(text, lbl)
            rewards.append(reward)
        return rewards

    def _score_single(self, text: str, label: int) -> float:
        """Compute reward for a single completion."""
        score, decision, has_format = self._parse_output(text)

        reward = 0.0

        predicted_label = 1 if decision == "SURFACE" else 0
        if predicted_label == label:
            reward += self.correctness_weight
        else:
            reward += self.wrong_penalty

        if label == 1:
            reward += self.calibration_weight * score
        else:
            reward += self.calibration_weight * (1.0 - score)

        if has_format:
            reward += self.format_weight

        return reward

    def _parse_output(self, text: str) -> tuple[float, str, bool]:
        """Extract score, decision, and format compliance from model output."""
        has_format = True

        score = 0.5
        score_match = re.search(r"<score>\s*([\d.]+)\s*</score>", text)
        if score_match:
            try:
                score = max(0.0, min(1.0, float(score_match.group(1))))
            except ValueError:
                has_format = False
        else:
            has_format = False
            num_match = re.search(r"([\d.]+)", text)
            if num_match:
                try:
                    val = float(num_match.group(1))
                    if 0 <= val <= 1:
                        score = val
                except ValueError:
                    pass

        decision = "SURFACE" if score > 0.5 else "FILTER"
        dec_match = re.search(r"<decision>\s*(SURFACE|FILTER)\s*</decision>", text, re.IGNORECASE)
        if dec_match:
            decision = dec_match.group(1).upper()
        else:
            has_format = False

        return score, decision, has_format


def correctness_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Simple binary correctness reward (for ablation studies)."""
    label = kwargs.get("label", [0] * len(completions))
    rewards = []
    for completion, lbl in zip(completions, label):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        score_match = re.search(r"<score>\s*([\d.]+)\s*</score>", text)
        if score_match:
            score = float(score_match.group(1))
            predicted = 1 if score > 0.5 else 0
        else:
            predicted = 1 if "SURFACE" in text.upper() else 0
        rewards.append(1.0 if predicted == lbl else -1.0)
    return rewards


def calibration_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Reward based on score calibration only (for ablation studies)."""
    label = kwargs.get("label", [0] * len(completions))
    rewards = []
    for completion, lbl in zip(completions, label):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        score_match = re.search(r"<score>\s*([\d.]+)\s*</score>", text)
        if score_match:
            score = max(0.0, min(1.0, float(score_match.group(1))))
        else:
            score = 0.5
        if lbl == 1:
            rewards.append(score)
        else:
            rewards.append(1.0 - score)
    return rewards


def format_reward(
    completions: list[list[dict[str, str]]],
    **kwargs,
) -> list[float]:
    """Reward for following the output format (for use as auxiliary reward)."""
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        r = 0.0
        if re.search(r"<think>.*?</think>", text, re.DOTALL):
            r += 0.33
        if re.search(r"<score>\s*[\d.]+\s*</score>", text):
            r += 0.33
        if re.search(r"<decision>\s*(SURFACE|FILTER)\s*</decision>", text, re.IGNORECASE):
            r += 0.34
        rewards.append(r)
    return rewards
