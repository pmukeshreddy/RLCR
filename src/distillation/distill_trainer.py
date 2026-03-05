"""Distill the RL teacher's knowledge into a fast sentence transformer.

Training approach:
  1. Create contrastive pairs from teacher scores. Pairs where the teacher
     gave similar scores should be close in embedding space; pairs with
     divergent scores should be far apart.
  2. Fine-tune MiniLM backbone with CosineSimilarityLoss.
  3. Train per-team linear projection heads for direct score prediction.

The result is:
  - A shared embedding backbone that encodes RL-informed similarity
  - Per-team heads that predict the teacher's score directly
  - Both are tiny and CPU-friendly for production serving
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def _format_input(record: dict) -> str:
    """Format a single record into the text fed to the sentence transformer."""
    team = record.get("team", "")
    diff = record.get("diff", "")[:500]
    comment = record.get("comment", "")[:300]
    return f"[{team}] {diff} [SEP] {comment}"


def _create_contrastive_pairs(
    records: list[dict],
    pairs_per_sample: int = 5,
    seed: int = 42,
) -> list[InputExample]:
    """Create contrastive training pairs from teacher-labeled data.

    For each sample, pair with `pairs_per_sample` other samples.
    Target similarity = 1 - |score_a - score_b| (both high/low → similar).
    """
    rng = random.Random(seed)
    pairs = []

    texts = [_format_input(r) for r in records]
    scores = [r["teacher_score"] for r in records]
    n = len(records)

    for i in range(n):
        partners = rng.sample(range(n), min(pairs_per_sample, n - 1))
        for j in partners:
            if i == j:
                continue
            label = 1.0 - abs(scores[i] - scores[j])
            pairs.append(InputExample(
                texts=[texts[i], texts[j]],
                label=float(label),
            ))

    logger.info(f"Created {len(pairs)} contrastive pairs from {n} samples")
    return pairs


class TeamProjectionHead(nn.Module):
    """Per-team linear head: embedding → score prediction."""

    def __init__(self, input_dim: int = 384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


def train_backbone(
    all_records: list[dict],
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "outputs/distilled_model",
    epochs: int = 3,
    batch_size: int = 64,
    warmup_frac: float = 0.1,
    pairs_per_sample: int = 5,
    seed: int = 42,
) -> SentenceTransformer:
    """Fine-tune the sentence transformer backbone on contrastive pairs.

    This shapes the embedding space so that comments the RL teacher scored
    similarly are close together, regardless of surface-level text similarity.
    """
    logger.info(f"Training backbone from {base_model_name} on {len(all_records)} records")

    model = SentenceTransformer(base_model_name)
    pairs = _create_contrastive_pairs(all_records, pairs_per_sample, seed)
    dataloader = DataLoader(pairs, shuffle=True, batch_size=batch_size)
    loss_fn = losses.CosineSimilarityLoss(model)

    warmup_steps = int(len(dataloader) * epochs * warmup_frac)

    model.fit(
        train_objectives=[(dataloader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
    )

    logger.success(f"Backbone saved to {output_dir}")
    return model


def train_projection_heads(
    model: SentenceTransformer,
    labels_by_team: dict[str, list[dict]],
    output_dir: str = "outputs/distilled_model/heads",
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> dict[str, TeamProjectionHead]:
    """Train per-team projection heads on top of the frozen backbone.

    Each head learns: embedding → teacher_score (MSE loss).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dim = model.get_sentence_embedding_dimension()

    heads = {}
    for team_name, records in labels_by_team.items():
        logger.info(f"Training head for {team_name} ({len(records)} samples)")

        texts = [_format_input(r) for r in records]
        targets = torch.tensor([r["teacher_score"] for r in records], dtype=torch.float32)

        with torch.no_grad():
            embeddings = torch.tensor(
                model.encode(texts, batch_size=256, show_progress_bar=False),
                dtype=torch.float32,
            )

        head = TeamProjectionHead(input_dim=dim)
        optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        criterion = nn.MSELoss()

        n = len(embeddings)
        head.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                pred = head(embeddings[idx])
                loss = criterion(pred, targets[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                avg = total_loss / max(n_batches, 1)
                logger.debug(f"  {team_name} epoch {epoch+1}/{epochs}: loss={avg:.4f}")

        head_path = output_path / f"{team_name}.pt"
        torch.save(head.state_dict(), head_path)
        heads[team_name] = head
        logger.info(f"  Head saved: {head_path}")

    return heads


def distill(
    labels_by_team: dict[str, list[dict]],
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "outputs/distilled_model",
    backbone_epochs: int = 3,
    head_epochs: int = 20,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[SentenceTransformer, dict[str, TeamProjectionHead]]:
    """Full distillation pipeline: backbone + per-team heads."""
    all_records = []
    for records in labels_by_team.values():
        all_records.extend(records)

    backbone = train_backbone(
        all_records,
        base_model_name=base_model_name,
        output_dir=output_dir,
        epochs=backbone_epochs,
        batch_size=batch_size,
        seed=seed,
    )

    heads = train_projection_heads(
        backbone,
        labels_by_team,
        output_dir=os.path.join(output_dir, "heads"),
        epochs=head_epochs,
    )

    return backbone, heads
