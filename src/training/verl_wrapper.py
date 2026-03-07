"""Wrapper that applies DAPO dynamic sampling before launching veRL.

DAPO filters out groups where all responses get identical rewards —
these groups provide zero learning signal and dilute the loss denominator.
Without this, 99% of the training batch can be wasted on zero-advantage
tokens, making the effective learning rate negligible.

Usage: replace `python3 -m verl.trainer.main_ppo` with
       `python3 src/training/verl_wrapper.py` (same CLI args).
"""
from __future__ import annotations

from collections import defaultdict

import torch


def _patched_compute_advantage(
    data,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """compute_advantage with DAPO dynamic sampling for GRPO.

    After computing group-normalized advantages, masks out response tokens
    from groups where all rewards are identical (zero variance). This
    concentrates the loss on informative groups — the core of DAPO's
    dynamic sampling.
    """
    result = _original_compute_advantage(
        data, adv_estimator, gamma, lam, num_repeat,
        norm_adv_by_std_in_grpo, config,
    )

    if "uid" not in data.non_tensor_batch:
        return result

    scores = data.batch["token_level_rewards"].sum(dim=-1)
    uid = data.non_tensor_batch["uid"]

    group_scores: dict[str, list[float]] = defaultdict(list)
    for i in range(len(uid)):
        group_scores[uid[i]].append(scores[i].item())

    active = torch.ones(len(uid), 1, device=data.batch["response_mask"].device)
    n_filtered = 0
    for i in range(len(uid)):
        g = group_scores[uid[i]]
        if len(g) > 1 and max(g) - min(g) < 1e-4:
            active[i] = 0.0
            n_filtered += 1

    n_total = len(uid)
    if 0 < n_filtered < n_total:
        data.batch["response_mask"] = data.batch["response_mask"] * active
        n_active_groups = sum(
            1 for g in group_scores.values() if max(g) - min(g) >= 1e-4
        )
        print(
            f"[DAPO] Dynamic sampling: kept {n_active_groups}/{len(group_scores)} "
            f"groups ({n_total - n_filtered}/{n_total} samples)"
        )

    return result


_original_compute_advantage = None


def _apply_patch():
    global _original_compute_advantage
    import verl.trainer.ppo.ray_trainer as rt

    _original_compute_advantage = rt.compute_advantage
    rt.compute_advantage = _patched_compute_advantage
    print("[DAPO] Dynamic sampling patch applied")


if __name__ == "__main__":
    _apply_patch()
    from verl.trainer.main_ppo import main
    main()
