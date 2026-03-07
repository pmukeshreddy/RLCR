"""Reward functions for DAPO training on code review filtering.

Three-component reward (RLBFF-style: continuous + binary verifiable + format):
  1. Shaped correctness (weight 0.5): distance-based, smooth gradients.
     reward = (1 - |score - label|) * 2 - 1  → maps to [-1, +1]
  2. Binary decision reward (weight 0.3): uses <decision>SURFACE/FILTER</decision>
     as a verifiable binary signal (RLBFF). Asymmetric penalties:
       True positive  (SURFACE, label=1): +0.5
       True negative  (FILTER,  label=0): +0.5
       False positive (SURFACE, label=0): -1.5  ← strong: reduces noise
       False negative (FILTER,  label=1): -0.3  ← mild: don't miss too much
  3. Format compliance (weight 0.2): +1 for all three tags, -1 for garbage.

Unparseable output gets graduated penalties (not flat -1.0) based on how
close the response got to valid format, plus a length-shaping term that
ensures reward variance within every GRPO group — achieving the effect of
DAPO dynamic sampling without patching the advantage computation.

Plus DAPO overlong reward shaping for completions exceeding expected length.

Design rationale (Greptile use case):
  Greptile's value prop is reducing review noise (4x faster merges).
  False positives (surfacing irrelevant comments) destroy developer trust.
  False negatives (missing relevant comments) are recoverable.
  Asymmetric penalty: FP = 5x more penalized than FN.
  Ref: RLBFF (Wang et al., 2025) — binary verifiable rewards for nuanced quality.
"""

from __future__ import annotations

import re

from loguru import logger


_SCORE_RE = re.compile(r"<score>\s*([\d.]+)\s*</score>")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DECISION_RE = re.compile(r"<decision>\s*(SURFACE|FILTER)\s*</decision>", re.IGNORECASE)


def _extract_score(text: str) -> float | None:
    """Extract a numeric score from model output.

    Tries: <score> tags first, then bare float, then any 0-1 number.
    Returns None on failure so callers can apply a garbage penalty.
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

    return None


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
    """Three-component reward for code review filtering (RLBFF-style).

    Shaped correctness (0.5): smooth gradient signal.
    Binary decision reward (0.3): precision-focused verifiable signal.
    Format compliance (0.2): structured output incentive.
    Plus DAPO overlong penalty.
    """

    CORRECTNESS_WEIGHT = 0.5
    DECISION_WEIGHT = 0.3
    FORMAT_WEIGHT = 0.2

    # Asymmetric decision penalties (FP >> FN for noise reduction)
    TP_BONUS = 0.5   # SURFACE, label=1 — correct surface
    TN_BONUS = 0.5   # FILTER,  label=0 — correct filter
    FP_PENALTY = -1.5  # SURFACE, label=0 — noisy comment surfaced
    FN_PENALTY = -0.3  # FILTER,  label=1 — relevant comment missed

    def __init__(
        self,
        overlong_penalty: float = 1.0,
        overlong_buffer_len: int = 32,
        max_completion_length: int = 128,
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
            r, _ = self._score_single(text, lbl)
            rewards.append(r)
        return rewards

    def _score_single(self, text: str, label: int) -> tuple[float, float]:
        """Returns (reward, extracted_score)."""
        score = _extract_score(text)

        if score is None:
            return self._unparseable_reward(text), 0.5

        # Component 1: shaped correctness (smooth gradients)
        correctness = (1.0 - abs(score - float(label))) * 2.0 - 1.0

        # Component 2: binary decision reward (RLBFF verifiable signal)
        # Use <decision> tag if present, else fall back to score threshold
        decision_match = _DECISION_RE.search(text)
        if decision_match:
            predict = 1 if decision_match.group(1).upper() == "SURFACE" else 0
        else:
            predict = 1 if score >= 0.5 else 0

        if predict == 1 and label == 1:
            decision_reward = self.TP_BONUS
        elif predict == 0 and label == 0:
            decision_reward = self.TN_BONUS
        elif predict == 1 and label == 0:
            decision_reward = self.FP_PENALTY
        else:
            decision_reward = self.FN_PENALTY

        # Component 3: format compliance
        fmt = _check_format(text)
        format_reward = fmt * 2.0 - 1.0

        reward = (
            self.CORRECTNESS_WEIGHT * correctness
            + self.DECISION_WEIGHT * decision_reward
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

    def _unparseable_reward(self, text: str) -> float:
        """Graduated penalty for responses without a parseable <score> tag.

        Instead of a flat -1.0, assigns a reward based on structural progress
        plus a character-level length penalty that guarantees reward variance
        within every GRPO group — achieving DAPO dynamic sampling at the
        reward level (100% action rate, no zero-variance groups).

        Variance budget (for 8 truncated responses in the same structural tier):
          - Character counts naturally differ by 200-400 chars across responses
          - With coefficient 0.3 and ~4 chars/token, this yields ~0.02 reward
            spread per group — enough for meaningful advantages with
            norm_adv_by_std_in_grpo=False
        """
        text_lower = text.lower()
        has_think_open = "<think>" in text_lower
        has_think_close = "</think>" in text_lower

        think_word_count = 0
        if has_think_open:
            start = text_lower.find("<think>") + len("<think>")
            end = text_lower.find("</think>") if has_think_close else len(text)
            think_word_count = len(text[start:end].strip().split())

        if has_think_open and has_think_close:
            base = -0.3
        elif has_think_open and think_word_count > 10:
            base = -0.5
        elif has_think_open:
            base = -0.7
        else:
            base = -1.0

        max_chars = self.max_completion_length * 4
        char_ratio = min(len(text) / max(max_chars, 1), 1.0)
        length_penalty = 0.3 * char_ratio

        return base - length_penalty


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
        if score is None:
            rewards.append(-1.0)
        else:
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
