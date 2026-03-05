"""Production-ready filter using distilled embeddings.

Architecturally identical to EmbeddingFilter (FAISS + cosine similarity)
but uses the distilled sentence transformer whose embedding space was
shaped by the RL teacher's understanding of per-team relevance.

CRITICAL: The distilled model was trained on inputs formatted as
  "[{team}] {diff} [SEP] {comment}"
All encode paths MUST use this same format. The inherited EmbeddingFilter
methods that encode bare comments are overridden to include team context.

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


def _format_for_distilled(team_name: str, comment: str, diff: str = "") -> str:
    """Format input to match distillation training format."""
    return f"[{team_name}] {diff[:1500]} [SEP] {comment[:500]}"


class DistilledFilter(EmbeddingFilter):
    """Drop-in replacement for EmbeddingFilter using distilled embeddings.

    Overrides all encoding paths to format inputs as [team] diff [SEP] comment,
    matching the format used during distillation training.
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

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Batch-encode texts (already formatted) into normalized embeddings."""
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def build_store(self, team_name: str, vote_history: list[dict]):
        """Build vector store with team-formatted inputs.

        vote_history entries may have 'diff' key for richer context.
        Falls back to comment-only if diff isn't available.
        """
        store = TeamVectorStore(dim=self.dim)
        if not vote_history:
            self.stores[team_name] = store
            return

        texts = [
            _format_for_distilled(
                team_name,
                v["comment"],
                v.get("diff", ""),
            )
            for v in vote_history
        ]
        embeddings = self._encode(texts)

        for emb, vote in zip(embeddings, vote_history):
            is_up = vote["vote"] == "upvote"
            store.add_vote(emb, vote["comment"], is_up)

        self.stores[team_name] = store
        logger.info(
            f"  {team_name}: {store.n_upvotes} upvotes, "
            f"{store.n_downvotes} downvotes"
        )

    def tune_threshold(
        self,
        team_name: str,
        samples: list[CodeReviewSample],
        threshold_range: tuple[float, float] = (0.05, 0.95),
        n_steps: int = 50,
    ) -> float:
        """Find optimal threshold using team-formatted inputs."""
        if team_name not in self.stores:
            self.thresholds[team_name] = 0.0
            return 0.0

        store = self.stores[team_name]
        if store.total_votes < self.min_votes:
            self.thresholds[team_name] = 0.0
            return 0.0

        texts = [
            _format_for_distilled(team_name, s.comment, s.diff)
            for s in samples
        ]
        labels = np.array([s.label for s in samples])
        embeddings = self._encode(texts)

        scores = np.array([
            store.query(emb, upvote_weight=self.upvote_weight, downvote_weight=self.downvote_weight)
            for emb in embeddings
        ])

        best_threshold = 0.0
        best_f1 = -1.0
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)

        for t in thresholds:
            preds = (scores > t).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        self.thresholds[team_name] = best_threshold
        logger.info(f"  {team_name}: threshold={best_threshold:.3f}, F1={best_f1:.3f}")
        return best_threshold

    def predict(
        self, team_name: str, comments: list[str], diffs: list[str] | None = None,
    ) -> list[dict[str, float]]:
        """Score with team-formatted inputs.

        Args:
            team_name: Team to score for.
            comments: Comment texts.
            diffs: Optional diff texts (improves accuracy if available).
        """
        if self.use_heads and team_name in self.heads:
            return self.predict_with_heads(team_name, comments, diffs)

        store = self.stores.get(team_name)
        threshold = self.thresholds.get(team_name, 0.0)

        if store is None or store.total_votes < self.min_votes:
            return [{"score": 0.5, "decision": 1, "confidence": 0.0} for _ in comments]

        texts = [
            _format_for_distilled(
                team_name, c, diffs[i] if diffs and i < len(diffs) else ""
            )
            for i, c in enumerate(comments)
        ]
        embeddings = self._encode(texts)

        results = []
        for emb in embeddings:
            score = store.query(
                emb, upvote_weight=self.upvote_weight, downvote_weight=self.downvote_weight
            )
            decision = 1 if score > threshold else 0
            confidence = abs(score - threshold) / max(abs(threshold) + 0.1, 0.1)
            results.append({
                "score": float(score),
                "decision": decision,
                "confidence": min(float(confidence), 1.0),
            })
        return results

    def predict_with_heads(
        self, team_name: str, comments: list[str], diffs: list[str] | None = None,
    ) -> list[dict[str, float]]:
        """Score using per-team projection heads (direct prediction mode)."""
        if team_name not in self.heads:
            logger.warning(f"No head for {team_name}, falling back to FAISS")
            return self.predict(team_name, comments, diffs)

        texts = [
            _format_for_distilled(
                team_name, c, diffs[i] if diffs and i < len(diffs) else ""
            )
            for i, c in enumerate(comments)
        ]

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
