"""veRL-based DAPO training for RLCR.

Uses veRL's RayPPOTrainer with FSDP across all GPUs.
Rollout uses veRL's built-in engine (controlled by ROLLOUT_ENGINE constant).

Each team gets a fresh RayPPOTrainer with its own LoRA adapter.
Base model loads per-team (14B loads in ~6s on H100 NVMe — acceptable).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from transformers import AutoTokenizer

from src.models.scoring import format_prompt_text

ROLLOUT_ENGINE = "sglang"


def _make_parquet(
    samples: list[dict],
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    tokenizer,
    output_path: str,
) -> str:
    """Convert RLCR samples to a veRL-compatible parquet file.

    veRL expects parquet with columns:
      - prompt: list of chat messages (as JSON string or list-of-dicts)
      - data_source: dataset name
      - ground_truth: label for reward function
      - extra_info: optional dict (JSON string)
    """
    records = []
    for s in samples:
        diff = s["diff"] if isinstance(s, dict) else s.diff
        comment = s["comment"] if isinstance(s, dict) else s.comment
        label = s["label"] if isinstance(s, dict) else s.label

        prompt_text = format_prompt_text(
            diff=diff,
            comment=comment,
            team_name=team_name,
            team_description=team_description,
            vote_history=vote_history,
            tokenizer=tokenizer,
        )

        messages = [{"role": "user", "content": prompt_text}]

        records.append({
            "prompt": json.dumps(messages),
            "data_source": f"rlcr_{team_name}",
            "ground_truth": str(int(label)),
            "extra_info": json.dumps({"team": team_name}),
        })

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(records)} samples to {output_path}")
    return output_path


def build_verl_command(
    team_name: str,
    train_parquet: str,
    val_parquet: str,
    config_dict: dict,
    output_dir: str,
    n_train_samples: int = 0,
) -> list[str]:
    """Generate the veRL CLI command for a single team's GRPO training.

    Returns a list of args for subprocess.run().
    """
    model_name = config_dict["model_name"]
    lora_r = config_dict.get("lora_r", 32)
    lora_alpha = config_dict.get("lora_alpha", 64)
    lr = config_dict.get("learning_rate", 2e-5)
    num_epochs = config_dict.get("num_epochs", 5)
    group_size = config_dict.get("group_size", 8)
    batch_size = config_dict.get("per_device_batch_size", 8) * group_size
    if n_train_samples > 0:
        batch_size = min(batch_size, n_train_samples)
    ppo_epochs = config_dict.get("ppo_epochs", 2)
    max_completion = config_dict.get("max_completion_length", 128)
    n_gpus = config_dict.get("n_gpus", 2)
    clip_low = config_dict.get("clip_ratio_low", 0.2)
    clip_high = config_dict.get("clip_ratio_high", 0.28)

    reward_fn_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "verl_reward.py")
    )

    args = [
        "python3", "-m", "verl.trainer.main_ppo",
        f"algorithm.adv_estimator=grpo",
        f"data.train_files={os.path.abspath(train_parquet)}",
        f"data.val_files={os.path.abspath(val_parquet)}",
        f"data.train_batch_size={batch_size}",
        f"data.max_prompt_length=1024",
        f"data.max_response_length={max_completion}",
        f"data.filter_overlong_prompts=True",
        f"data.truncation=error",
        f"actor_rollout_ref.model.path={model_name}",
        f"actor_rollout_ref.model.lora_rank={lora_r}",
        f"actor_rollout_ref.model.lora_alpha={lora_alpha}",
        f"actor_rollout_ref.model.use_remove_padding=True",
        f"actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.optim.lr={lr}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        f"actor_rollout_ref.actor.ppo_epochs={ppo_epochs}",
        f"actor_rollout_ref.actor.use_kl_loss=False",
        f"actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.clip_ratio_low={clip_low}",
        f"actor_rollout_ref.actor.clip_ratio_high={clip_high}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload=False",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.rollout.name={ROLLOUT_ENGINE}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization=0.80",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.n={group_size}",
        f"actor_rollout_ref.rollout.temperature=1.2",
        f"actor_rollout_ref.rollout.top_p=0.95",
        f"actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.rollout.layered_summon=True",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        f"actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"algorithm.use_kl_in_reward=False",
        f"algorithm.kl_ctrl.kl_coef=0.0",
        f"custom_reward_function.path={reward_fn_path}",
        f"custom_reward_function.name=compute_score",
        f"trainer.val_before_train=False",
        f"trainer.critic_warmup=0",
        "trainer.logger=[console]",
        f"trainer.project_name=rlcr",
        f"trainer.experiment_name=rlcr_{team_name}",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.nnodes=1",
        f"trainer.save_freq=50",
        f"trainer.test_freq=10",
        f"trainer.total_epochs={num_epochs}",
        f"trainer.default_local_dir={os.path.abspath(output_dir)}",
    ]

    return args


def train_team_verl(
    team_name: str,
    team,
    config_dict: dict,
    tokenizer,
) -> dict:
    """Train a single team using veRL's GRPO trainer."""
    output_dir = os.path.join(config_dict["base_output_dir"], team_name)
    data_dir = os.path.join("data", "verl", team_name)

    train_dicts = [s.to_dict() for s in team.train_samples]
    eval_dicts = [s.to_dict() for s in team.test_samples[:50]]

    train_parquet = _make_parquet(
        train_dicts, team_name, team.description, team.vote_history,
        tokenizer, os.path.join(data_dir, "train.parquet"),
    )
    val_parquet = _make_parquet(
        eval_dicts, team_name, team.description, team.vote_history,
        tokenizer, os.path.join(data_dir, "val.parquet"),
    )

    cmd = build_verl_command(
        team_name=team_name,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        config_dict=config_dict,
        output_dir=output_dir,
        n_train_samples=len(train_dicts),
    )

    logger.info(f"[veRL] Training team: {team_name}")
    logger.info(f"[veRL] Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["RLCR_REWARD_OVERLONG_PENALTY"] = str(config_dict.get("overlong_penalty", 1.0))
    env["RLCR_REWARD_OVERLONG_BUFFER_LEN"] = str(config_dict.get("overlong_buffer_len", 32))
    env["RLCR_REWARD_MAX_COMPLETION_LENGTH"] = str(config_dict.get("max_completion_length", 128))

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"[veRL] Team {team_name} training failed (exit {e.returncode})")
        return {"error": f"exit_code={e.returncode}", "team": team_name}

    logger.success(f"[veRL] Team {team_name} training complete")
    return {
        "team": team_name,
        "output_dir": output_dir,
        "best_reward": "see_verl_logs",
    }


def train_all_teams_verl(
    teams: dict,
    config_dict: dict,
) -> dict[str, dict]:
    """Train all teams sequentially using veRL.

    Each team gets a fresh veRL trainer invocation. The base model loads
    per-team (~6s on H100 NVMe for 14B). veRL manages FSDP + rollout internally.
    """
    model_name = config_dict["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    config_dict["n_gpus"] = n_gpus
    logger.info(f"[veRL] Training {len(teams)} teams with {n_gpus} GPUs")
    logger.info(f"[veRL] Model: {model_name}")

    results = {}
    for team_name, team in teams.items():
        result = train_team_verl(team_name, team, config_dict, tokenizer)
        results[team_name] = result

    return results
