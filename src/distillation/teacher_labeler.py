"""Run the trained DAPO teacher over the corpus to generate soft labels.

For each team, loads the base model + team's LoRA adapter from
outputs/dapo/{team}/ and scores every sample using local inference.

SGLang CANNOT be used here because it serves the base model without LoRA.
The whole point is to score with the TRAINED policy, not the base model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.parser import CodeReviewSample
from src.models.scoring import format_prompt_text, parse_model_output


def _score_batch_local(
    model,
    tokenizer,
    samples: list[CodeReviewSample],
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    """Score samples using local model (with LoRA already loaded)."""
    device = next(model.parameters()).device
    results = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        prompts = [
            format_prompt_text(
                diff=s.diff, comment=s.comment,
                team_name=team_name, team_description=team_description,
                vote_history=vote_history, tokenizer=tokenizer,
            )
            for s in batch
        ]

        for prompt, sample in zip(prompts, batch):
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            parsed = parse_model_output(text)

            results.append({
                "diff": sample.diff[:2000],
                "comment": sample.comment[:500],
                "team": team_name,
                "teacher_score": parsed.score,
                "teacher_decision": parsed.decision,
                "ground_truth": sample.label,
            })

        if (start + batch_size) % 50 == 0 or start + batch_size >= len(samples):
            logger.info(f"    Scored {min(start + batch_size, len(samples))}/{len(samples)}")

    return results


def label_all_teams(
    teams: dict,
    model_name: str,
    lora_base_dir: str = "outputs/dapo",
    max_new_tokens: int = 256,
    **_kwargs,
) -> dict[str, list[dict]]:
    """Score the full corpus for all teams using trained LoRA adapters.

    Loads the base model ONCE, then for each team:
      1. Load team's LoRA adapter from {lora_base_dir}/{team_name}/
      2. Score all train + test samples
      3. Unload LoRA, move to next team

    This is local inference only — SGLang can't serve per-team LoRA.
    """
    logger.info(f"Loading base model: {model_name}")
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

    lora_dir = Path(lora_base_dir)
    all_labels = {}

    for team_name, team in teams.items():
        all_samples = list(team.train_samples) + list(team.test_samples)
        logger.info(f"Labeling team: {team_name} ({len(all_samples)} samples)")

        adapter_path = lora_dir / team_name
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            logger.info(f"  Loading LoRA adapter from {adapter_path}")
            peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
            peft_model.eval()
            scoring_model = peft_model
        else:
            logger.warning(
                f"  No LoRA adapter found at {adapter_path}, "
                f"using base model (results will be weaker)"
            )
            scoring_model = base_model

        labels = _score_batch_local(
            model=scoring_model,
            tokenizer=tokenizer,
            samples=all_samples,
            team_name=team_name,
            team_description=team.description,
            vote_history=team.vote_history,
            max_new_tokens=max_new_tokens,
        )
        all_labels[team_name] = labels

        scores = [r["teacher_score"] for r in labels]
        logger.info(
            f"  {team_name}: {len(labels)} samples | "
            f"mean={sum(scores)/max(len(scores),1):.3f} | "
            f"std={torch.tensor(scores).std().item():.3f}"
        )

        if scoring_model is not base_model:
            try:
                base_model = peft_model.unload()
            except Exception:
                base_model = peft_model.base_model.model
            if hasattr(base_model, "peft_config"):
                del base_model.peft_config
            del peft_model
            torch.cuda.empty_cache()

    del base_model
    torch.cuda.empty_cache()
    return all_labels


def save_labels(labels: dict[str, list[dict]], output_dir: str | Path):
    """Save teacher labels to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for team_name, records in labels.items():
        all_records.extend(records)
        team_path = output_dir / f"{team_name}.json"
        with open(team_path, "w") as f:
            json.dump(records, f, indent=2)

    combined_path = output_dir / "all_labels.json"
    with open(combined_path, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.success(f"Saved {len(all_records)} labels to {output_dir}")
    return combined_path


def load_labels(label_dir: str | Path) -> dict[str, list[dict]]:
    """Load teacher labels from disk."""
    label_dir = Path(label_dir)
    labels = {}
    for path in label_dir.glob("*.json"):
        if path.name == "all_labels.json":
            continue
        team_name = path.stem
        with open(path) as f:
            labels[team_name] = json.load(f)
    return labels
