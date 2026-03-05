"""DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) training
for code review scoring.

DAPO extends GRPO with four techniques:
  1. Clip-Higher: asymmetric clipping (e_low=0.2, e_high=0.28)
  2. Token-level loss: normalize by total active tokens, not per-sequence
  3. Dynamic sampling: skip groups where all completions have identical rewards
  4. No KL penalty

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
  DAPO Policy Gradient Loss  ──►  backward()  ──►  optimizer.step()

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
from src.training.rewards import CodeReviewReward


@dataclass
class GRPORunConfig:
    """Configuration for a DAPO training run.

    DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) extends
    GRPO with: Clip-Higher, token-level loss, dynamic sampling, no KL.
    """

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
    gradient_accumulation_steps: int = 1  # only used by TRL path
    ppo_epochs: int = 2
    warmup_ratio: float = 0.1
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    dynamic_sampling: bool = True
    overlong_penalty: float = 1.0
    overlong_buffer_len: int = 64
    max_grad_norm: float = 0.5
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 25
    max_completion_length: int = 256
    max_prompt_length: int = 1024
    seed: int = 42
    sglang_url: str | None = None
    sglang_concurrent: int = 32


def build_training_dataset(
    samples: list[dict],
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    tokenizer=None,
) -> Dataset:
    """Convert code review samples into a training-ready dataset.

    Uses the SAME structured prompt as evaluation (think/score/decision format)
    so DAPO can explore diverse reasoning strategies.
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
    """Check if an SGLang server is healthy. Tries multiple endpoints."""
    import requests
    for endpoint in ["/health", "/v1/models"]:
        try:
            resp = requests.get(f"{url}{endpoint}", timeout=10)
            if resp.status_code == 200:
                logger.info(f"SGLang healthy at {url}{endpoint}")
                return True
        except Exception:
            continue
    logger.warning(f"SGLang not reachable at {url} (tried /health and /v1/models)")
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
            "top_p": 0.95,
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
    max_tokens: int = 32,
    temperature: float = 1.2,
    max_workers: int = 16,
    max_prompt_length: int = 1024,
) -> tuple[list[list[list[dict]]], list[torch.Tensor], list[list[torch.Tensor]]]:
    """Generate completions for all prompts x group_size via SGLang.

    Sends concurrent requests -- SGLang's continuous batching handles the rest.
    Temperature 1.2 for high diversity (DAPO benefits from exploration).
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
    max_tokens: int = 32,
    max_prompt_length: int = 1024,
    temperature: float = 0.9,
) -> tuple[list[list[list[dict]]], list[torch.Tensor], list[list[torch.Tensor]]]:
    """On-policy generation with the LoRA-adapted model.

    Batches all group_size completions per prompt in a single
    model.generate() call for ~group_size× speedup over sequential.
    """
    all_completions = []
    all_prompt_ids = []
    all_gen_ids = []

    for prompt in prompts:
        prompt_enc = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length
        ).to(device)
        prompt_len = prompt_enc["input_ids"].shape[1]
        all_prompt_ids.append(prompt_enc["input_ids"][0])

        batch_ids = prompt_enc["input_ids"].expand(group_size, -1)
        batch_mask = prompt_enc["attention_mask"].expand(group_size, -1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        group_comps = []
        group_ids = []
        for j in range(group_size):
            gen_ids = outputs.sequences[j][prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            group_comps.append([{"role": "assistant", "content": text}])
            group_ids.append(gen_ids)

        all_completions.append(group_comps)
        all_gen_ids.append(group_ids)
        del outputs

    return all_completions, all_prompt_ids, all_gen_ids


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RLCRTrainer:
    """Orchestrates DAPO training for code review scoring.

    Supports two modes:
      1. Single-team training: Train a LoRA adapter for one team
      2. Multi-team training via Ray: Parallel training across all teams

    DAPO techniques applied in the custom loop:
      - Clip-Higher: asymmetric clipping for importance ratios
      - Token-level loss: normalize by total active tokens across the batch
      - Dynamic sampling: skip groups with zero reward variance
      - No KL penalty

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
        logger.info(f"Setting up DAPO trainer for team: {self.config.team_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        load_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, attn_implementation="flash_attention_2", **load_kwargs
            )
            logger.info("Using Flash Attention 2")
        except (ValueError, ImportError):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **load_kwargs
            )
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train(self, train_dataset: Dataset, eval_dataset: Dataset | None = None) -> dict:
        """Run DAPO training.

        Attempts to use TRL's GRPOTrainer (with loss_type='dapo') if available,
        otherwise falls back to a custom training loop.
        """
        self.setup()

        try:
            return self._train_with_trl(train_dataset, eval_dataset)
        except (ImportError, Exception) as e:
            logger.warning(f"TRL GRPOTrainer not available ({e}), using custom DAPO loop")
            return self._train_custom(train_dataset, eval_dataset)

    def _train_with_trl(self, train_dataset: Dataset, eval_dataset: Dataset | None) -> dict:
        """Train using TRL's GRPOTrainer with DAPO loss type."""
        from trl import GRPOTrainer, GRPOConfig

        reward_fn = CodeReviewReward(
            overlong_penalty=self.config.overlong_penalty,
            overlong_buffer_len=self.config.overlong_buffer_len,
            max_completion_length=self.config.max_completion_length,
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
            loss_type="dapo",
            beta=0.0,
            epsilon_high=self.config.clip_ratio_high,
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
            reward_funcs=[reward_fn],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )

        logger.info("Starting DAPO training with TRL (loss_type='dapo')...")
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
        """Custom DAPO training loop with proper importance ratio + clipping.

        Phase 1 -- Rollout generation (SGLang or local fallback):
          Generate group_size completions per prompt.

        Phase 2 -- Reward computation + dynamic sampling:
          Binary reward (+1/-1). Skip groups with zero reward variance.

        Phase 3 -- Old log-probs (no_grad batched forward pass):
          Compute token-level log-probs under the CURRENT policy before
          any updates. These serve as the reference for importance ratios.

        Phase 4 -- PPO inner loop with DAPO Clip-Higher:
          For ppo_epochs iterations on the SAME rollouts:
            - Forward pass → new token log-probs
            - ratio = exp(new_lp - old_lp)
            - Asymmetric clip: clamp(ratio, 1-ε_low, 1+ε_high)
            - loss = -min(ratio·A, clipped_ratio·A)  (token-level, summed)
            - Normalize by total active tokens
            - backward → clip_grad → optimizer.step()

          ε_high > ε_low allows the policy to increase probability of
          good actions more aggressively than it decreases bad ones.
        """
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        device = next(self.model.parameters()).device
        reward_fn = CodeReviewReward(
            overlong_penalty=self.config.overlong_penalty,
            overlong_buffer_len=self.config.overlong_buffer_len,
            max_completion_length=self.config.max_completion_length,
        )
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

        batches_per_epoch = max(len(train_dataset) // self.config.per_device_batch_size, 1)
        total_optim_steps = batches_per_epoch * self.config.num_epochs * self.config.ppo_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_optim_steps * self.config.warmup_ratio),
            num_training_steps=total_optim_steps,
        )

        # Always use on-policy local generation during training so that
        # rollouts reflect the current LoRA policy, not a frozen base model.
        use_sglang = False
        logger.info("Rollout via local LoRA model (on-policy)")

        eps_low = self.config.clip_ratio_low
        eps_high = self.config.clip_ratio_high
        total_batches = batches_per_epoch * self.config.num_epochs
        logger.info(
            f"DAPO training: {total_batches} batches × {self.config.ppo_epochs} PPO epochs "
            f"= {total_optim_steps} optimizer steps | "
            f"clip=[{1-eps_low:.2f}, {1+eps_high:.2f}], "
            f"dynamic_sampling={self.config.dynamic_sampling}"
        )

        optim_step = 0
        global_step = 0
        total_reward = 0.0
        best_reward = float("-inf")
        total_groups_skipped = 0
        total_groups_seen = 0
        n_reward_samples = 0

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
                torch.cuda.empty_cache()

                # Phase 2: Rewards + dynamic sampling filter
                batch_rewards = []
                active_mask = []
                for group_comps, lbl in zip(all_completions, labels):
                    group_rewards = reward_fn(
                        group_comps, label=[lbl] * len(group_comps)
                    )
                    batch_rewards.append(group_rewards)
                    total_groups_seen += 1

                    reward_std = torch.tensor(group_rewards).std().item()
                    if self.config.dynamic_sampling and reward_std < 1e-8:
                        active_mask.append(False)
                        total_groups_skipped += 1
                    else:
                        active_mask.append(True)

                # Collect active sequences + advantages
                sequences = []
                for i, (prompt_ids_cpu, group_gen_ids, group_rewards) in enumerate(
                    zip(all_prompt_ids, all_gen_ids, batch_rewards)
                ):
                    if not active_mask[i]:
                        continue

                    rewards_t = torch.tensor(group_rewards, dtype=torch.float32)
                    mean_r = rewards_t.mean()
                    std_r = rewards_t.std() + 1e-8
                    advantages = (rewards_t - mean_r) / std_r
                    total_reward += mean_r.item()
                    n_reward_samples += 1

                    for j, gen_ids in enumerate(group_gen_ids):
                        if gen_ids.numel() == 0:
                            continue
                        full_ids = torch.cat([prompt_ids_cpu, gen_ids])
                        sequences.append({
                            "full_ids": full_ids,
                            "prompt_len": prompt_ids_cpu.shape[0],
                            "gen_len": gen_ids.shape[0],
                            "advantage": advantages[j].item(),
                        })

                global_step += 1

                if not sequences:
                    continue

                # Build padded batch (shared across old_lp + PPO steps)
                max_len = max(s["full_ids"].shape[0] for s in sequences)
                n_seqs = len(sequences)
                pad_id = self.tokenizer.pad_token_id or 0

                batch_ids = torch.full(
                    (n_seqs, max_len), pad_id, dtype=torch.long, device=device
                )
                batch_mask = torch.zeros(
                    (n_seqs, max_len), dtype=torch.long, device=device
                )
                for k, s in enumerate(sequences):
                    seq_len = s["full_ids"].shape[0]
                    batch_ids[k, :seq_len] = s["full_ids"].to(device)
                    batch_mask[k, :seq_len] = 1

                # Micro-batch forward passes to avoid materialising
                # a single (n_seqs, max_len, vocab) logits tensor (~6 GiB).
                _FWD_MB = 4

                # Phase 3: Compute old log-probs (reference, no gradient)
                old_token_lps = []
                with torch.no_grad():
                    for mb_s in range(0, n_seqs, _FWD_MB):
                        mb_e = min(mb_s + _FWD_MB, n_seqs)
                        mb_logits = self.model(
                            input_ids=batch_ids[mb_s:mb_e],
                            attention_mask=batch_mask[mb_s:mb_e],
                        ).logits
                        for k_local, k_global in enumerate(range(mb_s, mb_e)):
                            s = sequences[k_global]
                            pl, gl = s["prompt_len"], s["gen_len"]
                            glgts = mb_logits[k_local, pl - 1 : pl - 1 + gl, :]
                            targets = batch_ids[k_global, pl : pl + gl]
                            lp = torch.log_softmax(glgts, dim=-1)
                            old_token_lps.append(
                                lp.gather(1, targets.unsqueeze(1)).squeeze(1)
                            )
                        del mb_logits

                # Phase 4: PPO inner loop with DAPO Clip-Higher
                # Pre-compute total tokens for correct loss normalisation
                # across micro-batched backward passes.
                total_tokens_all = sum(s["gen_len"] for s in sequences)
                step_loss = 0.0

                for ppo_step in range(self.config.ppo_epochs):
                    optimizer.zero_grad()
                    total_loss_val = 0.0
                    clip_frac = 0.0
                    n_clip_tokens = 0

                    for mb_s in range(0, n_seqs, _FWD_MB):
                        mb_e = min(mb_s + _FWD_MB, n_seqs)
                        logits = self.model(
                            input_ids=batch_ids[mb_s:mb_e],
                            attention_mask=batch_mask[mb_s:mb_e],
                        ).logits

                        mb_loss = torch.tensor(0.0, device=device)
                        for k_local, k_global in enumerate(range(mb_s, mb_e)):
                            s = sequences[k_global]
                            pl, gl = s["prompt_len"], s["gen_len"]
                            gen_logits = logits[k_local, pl - 1 : pl - 1 + gl, :]
                            gen_targets = batch_ids[k_global, pl : pl + gl]

                            new_lp = torch.log_softmax(gen_logits, dim=-1)
                            new_token_lps = new_lp.gather(
                                1, gen_targets.unsqueeze(1)
                            ).squeeze(1)

                            ratio = torch.exp(new_token_lps - old_token_lps[k_global])
                            clipped_ratio = torch.clamp(
                                ratio, 1.0 - eps_low, 1.0 + eps_high
                            )

                            adv = s["advantage"]
                            surr1 = ratio * adv
                            surr2 = clipped_ratio * adv
                            mb_loss += -torch.min(surr1, surr2).sum()

                            with torch.no_grad():
                                clip_frac += (ratio != clipped_ratio).float().sum().item()
                                n_clip_tokens += gl

                        if total_tokens_all > 0:
                            (mb_loss / total_tokens_all).backward()
                            total_loss_val += mb_loss.item()
                        del logits, mb_loss

                    if total_tokens_all > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.config.max_grad_norm,
                        )
                        optimizer.step()
                        scheduler.step()
                        step_loss = total_loss_val / total_tokens_all
                        optim_step += 1

                del batch_ids, batch_mask, old_token_lps
                torch.cuda.empty_cache()

                log_every = min(self.config.logging_steps, max(total_batches // 3, 1))
                if global_step % log_every == 0 or global_step == total_batches:
                    frac_skipped = total_groups_skipped / max(total_groups_seen, 1)
                    avg_reward = total_reward / max(n_reward_samples, 1)
                    cf = clip_frac / max(n_clip_tokens, 1)
                    logger.info(
                        f"  Batch {global_step}/{total_batches} "
                        f"(optim {optim_step}/{total_optim_steps}) | "
                        f"Loss: {step_loss:.4f} | "
                        f"Reward: {avg_reward:.3f} | "
                        f"Clip%: {cf:.1%} | "
                        f"Skip%: {frac_skipped:.1%} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    if avg_reward > best_reward:
                        best_reward = avg_reward
                    total_reward = 0.0
                    n_reward_samples = 0

                if global_step % self.config.save_steps == 0:
                    save_path = Path(self.config.output_dir) / f"checkpoint-{global_step}"
                    self.model.save_pretrained(str(save_path))
                    self.tokenizer.save_pretrained(str(save_path))

            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} complete")

        final_path = Path(self.config.output_dir)
        final_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        frac_skipped = total_groups_skipped / max(total_groups_seen, 1)
        logger.success(
            f"DAPO training complete. Best reward: {best_reward:.3f} | "
            f"Dynamic sampling skipped: {frac_skipped:.1%} of groups"
        )

        return {
            "best_reward": best_reward,
            "total_steps": global_step,
            "team": self.config.team_name,
            "output_dir": self.config.output_dir,
        }


# ---------------------------------------------------------------------------
# Single-GPU: load model once, swap LoRA per team
# ---------------------------------------------------------------------------

def train_all_teams(
    teams: dict,
    config_dict: dict,
) -> dict[str, dict]:
    """Train all teams on a single GPU with DAPO. Loads the base model ONCE.

    For each team: attach fresh LoRA -> train -> save adapter -> unload LoRA.
    Base model stays resident -- zero redundant loads, minimal GPU churn.
    """
    model_name = config_dict["model_name"]
    logger.info(f"Loading base model once: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="flash_attention_2", **load_kwargs
        )
        logger.info("Using Flash Attention 2")
    except (ValueError, ImportError):
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if torch.cuda.is_available():
        base_model.to("cuda")
    base_model.gradient_checkpointing_enable()

    lora_target = config_dict.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    sglang_url = config_dict.get("sglang_url")
    team_names = list(teams.keys())

    results = {}
    for idx, team_name in enumerate(team_names):
        team = teams[team_name]
        logger.info(f"[{idx+1}/{len(team_names)}] Training team: {team_name}")

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config_dict.get("lora_r", 16),
            lora_alpha=config_dict.get("lora_alpha", 32),
            lora_dropout=config_dict.get("lora_dropout", 0.05),
            target_modules=lora_target,
        )
        peft_model = get_peft_model(base_model, lora_cfg)

        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        logger.info(f"  LoRA params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        train_dicts = [s.to_dict() for s in team.train_samples]
        eval_dicts = [s.to_dict() for s in team.test_samples[:50]]

        train_ds = build_training_dataset(
            train_dicts, team_name, team.description, team.vote_history, tokenizer
        )
        eval_ds = build_training_dataset(
            eval_dicts, team_name, team.description, team.vote_history, tokenizer
        )

        run_config = GRPORunConfig(
            model_name=model_name,
            output_dir=os.path.join(config_dict["base_output_dir"], team_name),
            team_name=team_name,
            team_description=team.description,
            vote_history=team.vote_history,
            group_size=config_dict.get("group_size", 4),
            learning_rate=config_dict.get("learning_rate", 5e-6),
            num_epochs=config_dict.get("num_epochs", 3),
            per_device_batch_size=config_dict.get("per_device_batch_size", 4),
            ppo_epochs=config_dict.get("ppo_epochs", 2),
            clip_ratio_low=config_dict.get("clip_ratio_low", 0.2),
            clip_ratio_high=config_dict.get("clip_ratio_high", 0.28),
            dynamic_sampling=config_dict.get("dynamic_sampling", True),
            overlong_penalty=config_dict.get("overlong_penalty", 1.0),
            overlong_buffer_len=config_dict.get("overlong_buffer_len", 64),
            max_completion_length=config_dict.get("max_completion_length", 256),
            seed=config_dict.get("seed", 42),
            sglang_url=sglang_url,
            sglang_concurrent=config_dict.get("sglang_concurrent", 16),
        )

        trainer = RLCRTrainer(run_config)
        trainer.model = peft_model
        trainer.tokenizer = tokenizer

        try:
            result = trainer._train_custom(train_ds, eval_ds)
            results[team_name] = result
            logger.success(f"Team {team_name} complete")
        except Exception as e:
            logger.error(f"Team {team_name} failed: {e}")
            results[team_name] = {"error": str(e)}

        try:
            base_model = peft_model.unload()
        except Exception:
            base_model = peft_model.base_model.model
        if hasattr(base_model, "peft_config"):
            del base_model.peft_config
        del peft_model, trainer
        torch.cuda.empty_cache()

    del base_model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Ray: multi-GPU parallel training
# ---------------------------------------------------------------------------

def train_team_worker(
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    train_samples: list[dict],
    eval_samples: list[dict] | None,
    config_dict: dict,
) -> dict:
    """Ray-compatible worker function for per-team DAPO training."""
    run_config = GRPORunConfig(
        model_name=config_dict["model_name"],
        output_dir=os.path.join(config_dict["base_output_dir"], team_name),
        team_name=team_name,
        team_description=team_description,
        vote_history=vote_history,
        lora_r=config_dict.get("lora_r", 16),
        lora_alpha=config_dict.get("lora_alpha", 32),
        lora_dropout=config_dict.get("lora_dropout", 0.05),
        group_size=config_dict.get("group_size", 4),
        learning_rate=config_dict.get("learning_rate", 5e-6),
        num_epochs=config_dict.get("num_epochs", 3),
        per_device_batch_size=config_dict.get("per_device_batch_size", 4),
        ppo_epochs=config_dict.get("ppo_epochs", 2),
        clip_ratio_low=config_dict.get("clip_ratio_low", 0.2),
        clip_ratio_high=config_dict.get("clip_ratio_high", 0.28),
        dynamic_sampling=config_dict.get("dynamic_sampling", True),
        overlong_penalty=config_dict.get("overlong_penalty", 1.0),
        overlong_buffer_len=config_dict.get("overlong_buffer_len", 64),
        max_completion_length=config_dict.get("max_completion_length", 256),
        seed=config_dict.get("seed", 42),
        sglang_url=config_dict.get("sglang_url"),
        sglang_concurrent=config_dict.get("sglang_concurrent", 32),
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
    """Multi-GPU parallel DAPO training via Ray. Each worker gets 1 full GPU."""
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)

    @ray.remote(num_gpus=1)
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
