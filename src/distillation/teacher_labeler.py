"""Run the trained DAPO teacher over the corpus to generate soft labels.

For each team, loads the base model + team's LoRA adapter and scores
every sample. The output is a dataset of (diff, comment, team, teacher_score)
tuples that the distillation trainer uses as training targets.

SGLang is used for fast batch inference when available.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from src.data.parser import CodeReviewSample
from src.models.scoring import ReviewScorer, parse_model_output


def label_team(
    scorer: ReviewScorer,
    team_name: str,
    team_description: str,
    vote_history: list[dict],
    samples: list[CodeReviewSample],
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Score all samples for a single team using the RL teacher.

    Returns list of dicts: {diff, comment, team, teacher_score, ground_truth}.
    """
    batch = [{"diff": s.diff, "comment": s.comment} for s in samples]
    outputs = scorer.batch_score(
        batch,
        team_name=team_name,
        team_description=team_description,
        vote_history=vote_history,
        max_workers=batch_size,
    )

    labeled = []
    for sample, output in zip(samples, outputs):
        labeled.append({
            "diff": sample.diff[:2000],
            "comment": sample.comment[:500],
            "team": team_name,
            "teacher_score": output.score,
            "teacher_decision": output.decision,
            "ground_truth": sample.label,
        })

    logger.info(
        f"  {team_name}: labeled {len(labeled)} samples | "
        f"mean_score={sum(r['teacher_score'] for r in labeled)/max(len(labeled),1):.3f}"
    )
    return labeled


def label_all_teams(
    teams: dict,
    model_name: str,
    sglang_url: str | None = None,
    max_new_tokens: int = 256,
) -> dict[str, list[dict]]:
    """Score the full corpus for all teams.

    Returns {team_name: [labeled_samples]}.
    """
    scorer = ReviewScorer(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        use_sglang=sglang_url is not None,
        sglang_url=sglang_url or "http://127.0.0.1:30000",
    )

    all_labels = {}
    for team_name, team in teams.items():
        logger.info(f"Labeling team: {team_name} ({len(team.train_samples) + len(team.test_samples)} samples)")
        all_samples = list(team.train_samples) + list(team.test_samples)
        all_labels[team_name] = label_team(
            scorer=scorer,
            team_name=team_name,
            team_description=team.description,
            vote_history=team.vote_history,
            samples=all_samples,
        )

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
