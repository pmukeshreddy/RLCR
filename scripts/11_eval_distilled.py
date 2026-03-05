#!/usr/bin/env python3
"""Step 11: Three-Way Evaluation — vanilla embeddings vs RL teacher vs distilled.

The killer comparison chart. Shows:
  1. Vanilla MiniLM embeddings (Greptile's current approach)
  2. DAPO RL teacher (with trained LoRA adapters, local inference)
  3. Distilled embeddings (fast, CPU-friendly, RL-informed)

The RL teacher line uses the actual trained LoRA adapters loaded
per team — NOT the base model with in-context votes.

Usage:
    python scripts/11_eval_distilled.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from loguru import logger
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config, set_seed
from src.baselines.embedding_filter import EmbeddingFilter
from src.baselines.distilled_filter import DistilledFilter
from src.data.team_simulator import TeamSimulator
from src.evaluation.metrics import compute_metrics
from src.models.scoring import format_prompt_text, parse_model_output

console = Console()


def eval_filter_cold_start(
    filter_cls,
    filter_kwargs: dict,
    team_name: str,
    train_samples,
    test_samples,
    steps: list[int],
    n_seeds: int = 3,
    base_seed: int = 42,
) -> dict[int, dict]:
    """Evaluate an embedding filter at each cold-start step."""
    results = defaultdict(list)

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        rng = random.Random(seed)

        for n_samples in steps:
            filt = filter_cls(**filter_kwargs)

            if n_samples > 0:
                chosen = rng.sample(train_samples, min(n_samples, len(train_samples)))
                votes = [
                    {"comment": s.comment, "vote": "upvote" if s.label == 1 else "downvote"}
                    for s in chosen
                ]
                filt.build_store(team_name, votes)
                filt.tune_threshold(team_name, chosen)
            else:
                filt.build_store(team_name, [])
                filt.thresholds[team_name] = 0.0

            preds = filt.predict(team_name, [s.comment for s in test_samples])
            labels = [s.label for s in test_samples]
            decisions = [p["decision"] for p in preds]
            scores = [p["score"] for p in preds]

            metrics = compute_metrics(labels, decisions, scores)
            results[n_samples].append(metrics)

    aggregated = {}
    for n_samples, metrics_list in results.items():
        agg = {}
        for metric_name in ["accuracy", "precision", "recall", "f1", "auroc", "action_rate"]:
            values = [getattr(m, metric_name) for m in metrics_list]
            agg[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        aggregated[n_samples] = agg

    return aggregated


def eval_rl_teacher_cold_start(
    base_model,
    tokenizer,
    team_name: str,
    team_description: str,
    train_samples,
    test_samples,
    lora_dir: str,
    steps: list[int],
    n_seeds: int = 3,
    base_seed: int = 42,
    max_new_tokens: int = 256,
) -> dict[int, dict]:
    """Evaluate the RL teacher (with LoRA) at each cold-start step.

    Loads the trained LoRA adapter for this team and scores test samples.
    Cold-start variation comes from varying the vote history context.
    """
    device = next(base_model.parameters()).device
    adapter_path = Path(lora_dir) / team_name

    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        peft_model.eval()
        model = peft_model
        has_lora = True
    else:
        logger.warning(f"No LoRA for {team_name}, using base model")
        model = base_model
        has_lora = False

    results = defaultdict(list)

    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset

        for n_samples in steps:
            rng = random.Random(seed * 10000 + n_samples + 1)
            vote_history = []
            if n_samples > 0:
                chosen = rng.sample(train_samples, min(n_samples, len(train_samples)))
                vote_history = [
                    {"comment": s.comment, "vote": "upvote" if s.label == 1 else "downvote"}
                    for s in chosen
                ]

            all_scores = []
            all_decisions = []
            for s in test_samples:
                prompt = format_prompt_text(
                    diff=s.diff, comment=s.comment,
                    team_name=team_name, team_description=team_description,
                    vote_history=vote_history, tokenizer=tokenizer,
                )
                inputs = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generated = outputs[0][inputs["input_ids"].shape[1]:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                parsed = parse_model_output(text)
                all_scores.append(parsed.score)
                all_decisions.append(parsed.binary_label)

            labels = [s.label for s in test_samples]
            metrics = compute_metrics(labels, all_decisions, all_scores)
            results[n_samples].append(metrics)

    if has_lora:
        try:
            restored = peft_model.unload()
        except Exception:
            restored = peft_model.base_model.model
        if hasattr(restored, "peft_config"):
            del restored.peft_config
        del peft_model
        torch.cuda.empty_cache()

    aggregated = {}
    for n_samples, metrics_list in results.items():
        agg = {}
        for mn in ["accuracy", "precision", "recall", "f1", "auroc", "action_rate"]:
            values = [getattr(m, mn) for m in metrics_list]
            agg[mn] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        aggregated[n_samples] = agg

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Three-way evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--distilled-model", default="outputs/distilled_model")
    parser.add_argument("--lora-dir", default="outputs/dapo")
    parser.add_argument("--no-sglang", action="store_true")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    console.rule("[bold blue]Step 11: Three-Way Evaluation[/bold blue]")

    teams_dir = Path(cfg.data.processed_dir) / "teams"
    team_configs = list(cfg.teams.types)
    simulator = TeamSimulator.load(teams_dir, team_configs)

    steps = list(cfg.evaluation.cold_start_steps)
    n_seeds = cfg.evaluation.n_seeds
    model_name = cfg.model.large.name

    vanilla_kwargs = {
        "model_name": cfg.baseline.embedding_model,
        "dim": cfg.baseline.embedding_dim,
        "upvote_weight": cfg.baseline.upvote_weight,
        "downvote_weight": cfg.baseline.downvote_weight,
        "min_votes": cfg.baseline.min_votes_for_prediction,
        "batch_size": cfg.baseline.batch_size,
    }

    distilled_path = Path(args.distilled_model)
    distilled_kwargs = {
        "model_path": str(distilled_path),
        "dim": cfg.baseline.embedding_dim,
        "upvote_weight": cfg.baseline.upvote_weight,
        "downvote_weight": cfg.baseline.downvote_weight,
        "min_votes": cfg.baseline.min_votes_for_prediction,
        "batch_size": cfg.baseline.batch_size,
    }

    # Load base model once for RL teacher evaluation (LoRA swapped per team)
    console.print(f"Loading base model for RL teacher eval: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    load_kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation="flash_attention_2", **load_kwargs
        )
    except (ValueError, ImportError):
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if torch.cuda.is_available():
        base_model.to("cuda")
    base_model.eval()

    all_results = {}

    for team_name, team in simulator.teams.items():
        console.print(f"\n[bold cyan]{team_name.title()}[/bold cyan]")
        test_subset = team.test_samples[:50]

        console.print("  Evaluating vanilla embeddings...")
        vanilla_results = eval_filter_cold_start(
            EmbeddingFilter, vanilla_kwargs,
            team_name, team.train_samples, test_subset,
            steps, n_seeds, cfg.project.seed,
        )

        distilled_results = {}
        if distilled_path.exists():
            console.print("  Evaluating distilled embeddings...")
            distilled_results = eval_filter_cold_start(
                DistilledFilter, distilled_kwargs,
                team_name, team.train_samples, test_subset,
                steps, n_seeds, cfg.project.seed,
            )
        else:
            console.print("  [yellow]Distilled model not found, skipping[/yellow]")

        console.print("  Evaluating RL teacher (with LoRA)...")
        rl_results = eval_rl_teacher_cold_start(
            base_model=base_model,
            tokenizer=tokenizer,
            team_name=team_name,
            team_description=team.description,
            train_samples=team.train_samples,
            test_samples=test_subset,
            lora_dir=args.lora_dir,
            steps=steps,
            n_seeds=n_seeds,
            base_seed=cfg.project.seed,
            max_new_tokens=cfg.model.large.max_new_tokens,
        )

        all_results[team_name] = {
            "vanilla": vanilla_results,
            "distilled": distilled_results,
            "rl_teacher": rl_results,
        }

    del base_model
    torch.cuda.empty_cache()

    # --- Print results table ---
    console.rule("[bold]Three-Way Comparison (F1)[/bold]")
    table = Table(title="Cold-Start F1 at Key Points")
    table.add_column("Team / Method", style="cyan")
    for step in steps:
        table.add_column(f"n={step}", style="green")

    for team_name, team_results in all_results.items():
        for method, label in [("vanilla", "Vanilla Emb"), ("distilled", "Distilled"), ("rl_teacher", "RL Teacher")]:
            row = [f"{team_name.title()} ({label})"]
            data = team_results.get(method, {})
            for step in steps:
                if step in data and "f1" in data[step]:
                    m = data[step]["f1"]["mean"]
                    s = data[step]["f1"]["std"]
                    row.append(f"{m:.3f}±{s:.3f}")
                else:
                    row.append("—")
            table.add_row(*row)

    console.print(table)

    # --- Save ---
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for team_name, team_data in all_results.items():
        serializable[team_name] = {}
        for method, data in team_data.items():
            serializable[team_name][method] = {
                str(k): v for k, v in data.items()
            }

    with open(results_dir / "three_way_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    console.print(f"\n[bold green]✓ Step 11 complete![/bold green]")
    console.print(f"  Results: results/three_way_results.json")


if __name__ == "__main__":
    main()
