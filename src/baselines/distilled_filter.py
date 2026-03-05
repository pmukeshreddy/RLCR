"""Production-ready filter using distilled embeddings.

Architecturally identical to EmbeddingFilter (FAISS + cosine similarity)
but uses the distilled sentence transformer whose embedding space was
shaped by the RL teacher's understanding of per-team relevance.

Two scoring modes:
  1. FAISS similarity (default): same as Greptile's approach but with
     better embeddings. Interpretable via nearest-neighbor explanations.
  2. Direct prediction via per-team projection heads: the head directly
     outputs the teacher's predicted score. Simpler, no FAISS needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.baselines.embedding_filter import EmbeddingFilter, TeamVectorStore
from src.data.parser import CodeReviewSample
from src.distillation.distill_trainer import TeamProjectionHead


class DistilledFilter(EmbeddingFilter):
    """Drop-in replacement for EmbeddingFilter using distilled embeddings.

    Inherits all FAISS/threshold logic. Only difference: the encoder
    is a fine-tuned model whose embedding space encodes RL-informed
    relevance, not generic semantic similarity.
    """

    def __init__(
        self,
        model_path: str = "outputs/distilled_model",
        head_dir: str | None = None,
        dim: int = 384,
        upvote_weight: float = 1.0,
        downvote_weight: float = 0.8,
        min_votes: int = 3,
        batch_size: int = 256,
        use_heads: bool = False,
    ):
        self.dim = dim
        self.upvote_weight = upvote_weight
        self.downvote_weight = downvote_weight
        self.min_votes = min_votes
        self.batch_size = batch_size
        self.stores: dict[str, TeamVectorStore] = {}
        self.thresholds: dict[str, float] = {}
        self.use_heads = use_heads
        self.heads: dict[str, TeamProjectionHead] = {}

        logger.info(f"Loading distilled model from {model_path}")
        self.encoder = SentenceTransformer(model_path)

        if use_heads and head_dir:
            self._load_heads(head_dir)

    def _load_heads(self, head_dir: str):
        """Load per-team projection heads."""
        head_path = Path(head_dir)
        dim = self.encoder.get_sentence_embedding_dimension()
        for pt_file in head_path.glob("*.pt"):
            team_name = pt_file.stem
            head = TeamProjectionHead(input_dim=dim)
            head.load_state_dict(torch.load(pt_file, map_location="cpu", weights_only=True))
            head.eval()
            self.heads[team_name] = head
            logger.info(f"  Loaded head: {team_name}")

    def predict_with_heads(
        self, team_name: str, comments: list[str], diffs: list[str] | None = None,
    ) -> list[dict[str, float]]:
        """Score using per-team projection heads (direct prediction mode).

        Bypasses FAISS entirely. The head directly predicts the teacher's score.
        """
        if team_name not in self.heads:
            logger.warning(f"No head for {team_name}, falling back to FAISS")
            return self.predict(team_name, comments)

        texts = []
        for i, comment in enumerate(comments):
            diff = diffs[i][:500] if diffs and i < len(diffs) else ""
            texts.append(f"[{team_name}] {diff} [SEP] {comment}")

        embeddings = self.encoder.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False
        )
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)

        head = self.heads[team_name]
        with torch.no_grad():
            scores = head(emb_tensor).numpy()

        threshold = self.thresholds.get(team_name, 0.5)
        results = []
        for score in scores:
            s = float(score)
            results.append({
                "score": s,
                "decision": 1 if s > threshold else 0,
                "confidence": abs(s - threshold),
            })
        return results

    def predict(
        self, team_name: str, comments: list[str]
    ) -> list[dict[str, float]]:
        """Score via FAISS similarity (same API as EmbeddingFilter).

        Uses the distilled encoder instead of vanilla MiniLM.
        """
        if self.use_heads and team_name in self.heads:
            return self.predict_with_heads(team_name, comments)
        return super().predict(team_name, comments)
