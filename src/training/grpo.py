"""GRPO (Group Relative Policy Optimization) training for code review scoring.

Architecture for rollout generation:

  SGLang Server (fast batched generation, no gradients)
       │
       │  concurrent requests via ThreadPool
       │  prompt × group_size completions
       ▼
  Completion Text  ──►  Reward Function  ──►  Advantages
       │
       │  tokenize completions
       ▼
  Local Model + LoRA (forward pass WITH gradients)
       │
       │  log_softmax → gather token log-probs
       ▼
  Policy Gradient Loss  ──►  backward()  ──►  optimizer.step()

SGLang handles the expensive generation (continuous batching, KV-cache reuse).
The local model only runs forward passes for differentiable log-probs.
If SGLang is unavailable, falls back to local generation (slower but functional).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.scoring import format_prompt_text
from src.training.rewards import CodeReviewReward, format_reward


@dataclass
class GRPORunConfig:
    """Configuration for a single GRPO training run."""

    model_name: str
    output_dir: str
    team_name: str
    team_description: str
    vote_history: list[dict]
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    group_size: int = 8
    learning_rate: float = 5e-6
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    kl_coef: float = 0.04
    clip_range: float = 0.2
    max_grad_norm: float = 0.5
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50
    max_completion_length: int = 256
    max_prompt_length: int = 1024
    seed: int = 42
    sglang_url: str | None = None
    sglang_concurrent: int = 16


def build_training_dataset(
    samples: list[dict],
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    tokenizer=None,
) -> Dataset:
    """Convert code review samples into a GRPO-ready dataset.

    Each row has:
      - prompt: formatted text prompt
      - label: ground truth (for reward computation)
      - team: team name (for reward context)
    """
    records = []
    for s in samples:
        prompt = format_prompt_text(
            diff=s["diff"] if isinstance(s, dict) else s.diff,
            comment=s["comment"] if isinstance(s, dict) else s.comment,
            team_name=team_name,
            team_description=team_description,
            vote_history=vote_history,
            tokenizer=tokenizer,
        )
        label = s["label"] if isinstance(s, dict) else s.label
        records.append({
            "prompt": prompt,
            "label": int(label),
            "team": team_name,
        })
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# SGLang rollout helpers
# ---------------------------------------------------------------------------

def _probe_sglang(url: str) -> bool:
    """Check if an SGLang server is healthy."""
    try:
        import requests
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except Exception:
        return False


def _sglang_generate_one(args: tuple) -> str:
    """Generate a single completion via SGLang. Designed for ThreadPoolExecutor."""
    import requests

    sglang_url, prompt_text, model_name, max_tokens, temperature = args
    resp = requests.post(
        f"{sglang_url}/v1/completions",
        json={
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def rollout_sglang(
    prompts: list[str],
    group_size: int,
    sglang_url: str,
    model_name: str,
    tokenizer,
    max_tokens: int = 256,
    temperature: float = 0.8,
    max_workers: int = 16,
    max_prompt_length: int = 1024,
) -> tuple[list[list[list[dict]]], list[torch.Tensor], list[list[torch.Tensor]]]:
    """Generate completions for all prompts × group_size via SGLang.

    Sends concurrent requests — SGLang's continuous batching handles the rest.
    Returns (completions_text, prompt_ids, gen_ids_per_prompt).
    """
    tasks = []
    for prompt in prompts:
        for _ in range(group_size):
            tasks.append((sglang_url, prompt, model_name, max_tokens, temperature))

    results: list[str] = [""] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_sglang_generate_one, t): i for i, t in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"SGLang generation failed for request {idx}: {e}")
                results[idx] = ""

    all_completions = []
    all_prompt_ids = []
    all_gen_ids = []
    idx = 0
    for prompt in prompts:
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).input_ids[0]
        all_prompt_ids.append(prompt_ids)

        group_comps = []
        group_ids = []
        for _ in range(group_size):
            text = results[idx]
            group_comps.append([{"role": "assistant", "content": text}])
            gen_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
            group_ids.append(gen_ids)
            idx += 1
        all_completions.append(group_comps)
        all_gen_ids.append(group_ids)

    return all_completions, all_prompt_ids, all_gen_ids


def rollout_local(
    prompts: list[str],
    group_size: int,
    model,
    tokenizer,
    device,
    max_tokens: int = 256,
    max_prompt_length: int = 1024,
) -> tuple[list[list[list[dict]]], list[torch.Tensor], list[list[torch.Tensor]]]:
    """Fallback: generate completions with the local model (sequential, slower)."""
    all_completions = []
    all_prompt_ids = []
    all_gen_ids = []

    for prompt in prompts:
        prompt_enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).to(device)
        all_prompt_ids.append(prompt_enc["input_ids"][0])

        group_comps = []
        group_ids = []
        for _ in range(group_size):
            with torch.no_grad():
                outputs = model.generate(
                    **prompt_enc,
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = outputs.sequences[0][prompt_enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            group_comps.append([{"role": "assistant", "content": text}])
            group_ids.append(gen_ids)

        all_completions.append(group_comps)
        all_gen_ids.append(group_ids)

    return all_completions, all_prompt_ids, all_gen_ids


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RLCRTrainer:
    """Orchestrates GRPO training for code review scoring.

    Supports two modes:
      1. Single-team training: Train a LoRA adapter for one team
      2. Multi-team training via Ray: Parallel training across all teams

    Rollout generation uses SGLang by default (fast concurrent batched
    inference). Falls back to local model.generate() if SGLang is unavailable.
    """

    def __init__(self, config: GRPORunConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self):
        """Load model, apply LoRA, and prepare for training."""
        logger.info(f"Setting up GRPO trainer for team: {self.config.team_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules or ["q_proj", "v_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train(self, train_dataset: Dataset, eval_dataset: Dataset | None = None) -> dict:
        """Run GRPO training.

        Attempts to use TRL's GRPOTrainer if available, otherwise falls back
        to a custom training loop.
        """
        self.setup()

        try:
            return self._train_with_trl(train_dataset, eval_dataset)
        except (ImportError, Exception) as e:
            logger.warning(f"TRL GRPOTrainer not available ({e}), using custom loop")
            return self._train_custom(train_dataset, eval_dataset)

    def _train_with_trl(self, train_dataset: Dataset, eval_dataset: Dataset | None) -> dict:
        """Train using TRL's GRPOTrainer."""
        from trl import GRPOTrainer, GRPOConfig

        reward_fn = CodeReviewReward(
            correctness_weight=1.0,
            calibration_weight=0.3,
            format_weight=0.2,
        )

        trl_kwargs = dict(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            num_generations=self.config.group_size,
            max_completion_length=self.config.max_completion_length,
            max_prompt_length=self.config.max_prompt_length,
            kl_coef=self.config.kl_coef,
            seed=self.config.seed,
            bf16=torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
        )
        if eval_dataset:
            trl_kwargs["eval_steps"] = self.config.eval_steps
            import inspect
            sig = inspect.signature(GRPOConfig.__init__)
            if "eval_strategy" in sig.parameters:
                trl_kwargs["eval_strategy"] = "steps"
            else:
                trl_kwargs["evaluation_strategy"] = "steps"
        training_args = GRPOConfig(**trl_kwargs)

        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=[reward_fn, format_reward],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting GRPO training with TRL...")
        result = self.trainer.train()

        self.trainer.save_model(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        logger.success(f"Model saved to {self.config.output_dir}")

        return {
            "train_loss": result.training_loss if hasattr(result, "training_loss") else None,
            "team": self.config.team_name,
            "output_dir": self.config.output_dir,
        }

    def _train_custom(self, train_dataset: Dataset, eval_dataset: Dataset | None) -> dict:
        """Custom GRPO training loop with SGLang-accelerated rollouts.

        Phase 1 — Rollout generation (SGLang or local fallback):
          Send all prompt × group_size requests concurrently to SGLang.
          SGLang's continuous batching + KV-cache reuse makes this fast.
          No gradients needed — just text completions.

        Phase 2 — Reward computation:
          Parse scores/decisions from completion text, compare to labels.

        Phase 3 — Policy gradient (local model with gradients):
          Forward pass through the LoRA model on (prompt + completion).
          Extract differentiable log-probs, multiply by advantages, backprop.
        """
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        device = next(self.model.parameters()).device
        reward_fn = CodeReviewReward()
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )
        total_steps = (
            len(train_dataset)
            // self.config.per_device_batch_size
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )

        use_sglang = (
            self.config.sglang_url is not None
            and _probe_sglang(self.config.sglang_url)
        )
        if use_sglang:
            logger.info(
                f"Rollout generation via SGLang at {self.config.sglang_url} "
                f"(concurrency={self.config.sglang_concurrent})"
            )
        else:
            logger.info("Rollout generation via local model (no SGLang)")

        logger.info(f"Custom GRPO training: {total_steps} steps, {self.config.num_epochs} epochs")

        global_step = 0
        total_reward = 0.0
        best_reward = float("-inf")
        optimizer.zero_grad()

        for epoch in range(self.config.num_epochs):
            self.model.train()
            indices = torch.randperm(len(train_dataset))

            for batch_start in range(
                0, len(train_dataset), self.config.per_device_batch_size
            ):
                batch_idx = indices[
                    batch_start : batch_start + self.config.per_device_batch_size
                ]
                batch = [train_dataset[int(i)] for i in batch_idx]
                prompts = [b["prompt"] for b in batch]
                labels = [b["label"] for b in batch]

                # Phase 1: Generate completions (no gradients)
                if use_sglang:
                    all_completions, all_prompt_ids, all_gen_ids = rollout_sglang(
                        prompts=prompts,
                        group_size=self.config.group_size,
                        sglang_url=self.config.sglang_url,
                        model_name=self.config.model_name,
                        tokenizer=self.tokenizer,
                        max_tokens=self.config.max_completion_length,
                        max_workers=self.config.sglang_concurrent,
                        max_prompt_length=self.config.max_prompt_length,
                    )
                else:
                    all_completions, all_prompt_ids, all_gen_ids = rollout_local(
                        prompts=prompts,
                        group_size=self.config.group_size,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=device,
                        max_tokens=self.config.max_completion_length,
                        max_prompt_length=self.config.max_prompt_length,
                    )

                # Phase 2: Compute rewards from completion text
                batch_rewards = []
                for group_comps, lbl in zip(all_completions, labels):
                    group_rewards = reward_fn(
                        group_comps, label=[lbl] * len(group_comps)
                    )
                    batch_rewards.append(group_rewards)

                # Phase 3: Forward pass WITH gradients for differentiable log-probs
                batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

                for i, (prompt_ids_cpu, group_gen_ids, group_rewards) in enumerate(
                    zip(all_prompt_ids, all_gen_ids, batch_rewards)
                ):
                    prompt_ids = prompt_ids_cpu.to(device)
                    rewards_t = torch.tensor(group_rewards, dtype=torch.float32, device=device)
                    mean_r = rewards_t.mean()
                    std_r = rewards_t.std() + 1e-8
                    advantages = ((rewards_t - mean_r) / std_r).detach()
                    total_reward += mean_r.item()

                    group_log_probs = []
                    for gen_ids in group_gen_ids:
                        if gen_ids.numel() == 0:
                            group_log_probs.append(torch.tensor(0.0, device=device, requires_grad=True))
                            continue

                        full_ids = torch.cat([prompt_ids, gen_ids.to(device)]).unsqueeze(0)
                        attention_mask = torch.ones_like(full_ids)

                        logits = self.model(
                            input_ids=full_ids,
                            attention_mask=attention_mask,
                        ).logits

                        prompt_len = prompt_ids.shape[0]
                        gen_logits = logits[0, prompt_len - 1 : -1, :]
                        gen_targets = gen_ids.to(device)
                        min_len = min(gen_logits.shape[0], gen_targets.shape[0])
                        gen_logits = gen_logits[:min_len]
                        gen_targets = gen_targets[:min_len]

                        log_probs = torch.log_softmax(gen_logits, dim=-1)
                        token_log_probs = log_probs.gather(
                            1, gen_targets.unsqueeze(1)
                        ).squeeze(1)
                        avg_log_prob = token_log_probs.mean()
                        group_log_probs.append(avg_log_prob)

                    log_probs_t = torch.stack(group_log_probs)
                    policy_loss = -(advantages * log_probs_t).mean()
                    batch_loss = batch_loss + policy_loss / (
                        len(all_prompt_ids) * self.config.gradient_accumulation_steps
                    )

                batch_loss.backward()

                if (global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.config.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                global_step += 1

                if global_step % self.config.logging_steps == 0:
                    avg_reward = total_reward / (
                        self.config.logging_steps * len(prompts)
                    )
                    logger.info(
                        f"  Step {global_step}/{total_steps} | "
                        f"Loss: {batch_loss.item():.4f} | "
                        f"Reward: {avg_reward:.3f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                    total_reward = 0.0

                if global_step % self.config.save_steps == 0:
                    save_path = Path(self.config.output_dir) / f"checkpoint-{global_step}"
                    self.model.save_pretrained(str(save_path))
                    self.tokenizer.save_pretrained(str(save_path))

            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} complete")

        final_path = Path(self.config.output_dir)
        final_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        logger.success(f"Training complete. Best reward: {best_reward:.3f}")

        return {
            "best_reward": best_reward,
            "total_steps": global_step,
            "team": self.config.team_name,
            "output_dir": self.config.output_dir,
        }


# ---------------------------------------------------------------------------
# Ray distribution
# ---------------------------------------------------------------------------

def train_team_worker(
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    train_samples: list[dict],
    eval_samples: list[dict] | None,
    config_dict: dict,
) -> dict:
    """Ray-compatible worker function for per-team training."""
    run_config = GRPORunConfig(
        model_name=config_dict["model_name"],
        output_dir=os.path.join(config_dict["base_output_dir"], team_name),
        team_name=team_name,
        team_description=team_description,
        vote_history=vote_history,
        lora_r=config_dict.get("lora_r", 16),
        lora_alpha=config_dict.get("lora_alpha", 32),
        lora_dropout=config_dict.get("lora_dropout", 0.05),
        group_size=config_dict.get("group_size", 8),
        learning_rate=config_dict.get("learning_rate", 5e-6),
        num_epochs=config_dict.get("num_epochs", 3),
        per_device_batch_size=config_dict.get("per_device_batch_size", 2),
        gradient_accumulation_steps=config_dict.get("gradient_accumulation_steps", 4),
        seed=config_dict.get("seed", 42),
        sglang_url=config_dict.get("sglang_url"),
        sglang_concurrent=config_dict.get("sglang_concurrent", 16),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config_dict["model_name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_training_dataset(
        train_samples, team_name, team_description, vote_history, tokenizer
    )
    eval_ds = None
    if eval_samples:
        eval_ds = build_training_dataset(
            eval_samples, team_name, team_description, vote_history, tokenizer
        )

    trainer = RLCRTrainer(run_config)
    return trainer.train(train_ds, eval_ds)


def train_all_teams_ray(
    teams: dict,
    config_dict: dict,
    num_cpus: int = 4,
    num_gpus: int = 1,
) -> dict[str, dict]:
    """Launch per-team GRPO training via Ray.

    Each worker requests 1 full GPU. On multi-GPU (e.g., 4×A100), Ray runs
    up to num_gpus teams in parallel. On single GPU, Ray queues teams
    automatically — no OOM from concurrent model copies.
    """
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)

    gpu_per_worker = 1

    @ray.remote(num_gpus=gpu_per_worker)
    def _train_remote(team_name, team_desc, votes, train_s, eval_s, cfg):
        return train_team_worker(team_name, team_desc, votes, train_s, eval_s, cfg)

    futures = {}
    for team_name, team in teams.items():
        train_dicts = [s.to_dict() for s in team.train_samples]
        eval_dicts = [s.to_dict() for s in team.test_samples[:50]]
        futures[team_name] = _train_remote.remote(
            team_name,
            team.description,
            team.vote_history,
            train_dicts,
            eval_dicts,
            config_dict,
        )

    results = {}
    for team_name, future in futures.items():
        try:
            results[team_name] = ray.get(future)
            logger.success(f"Team {team_name} training complete")
        except Exception as e:
            logger.error(f"Team {team_name} training failed: {e}")
            results[team_name] = {"error": str(e)}

    return results
