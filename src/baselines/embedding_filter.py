"""Embedding-based code review filter — Greptile/Soohoon's approach.

Implements the cosine-similarity filtering strategy described in Soohoon Park's
blog post about Greptile's code review system. The core idea:

  1. Embed all historical review comments using a sentence transformer
  2. Build a per-team vector store with upvote/downvote labels
  3. For a new comment, compute weighted similarity:
       score = sim(comment, upvoted_comments) - λ * sim(comment, downvoted_comments)
  4. Surface if score > threshold, filter otherwise
  5. Threshold is tuned per-team on the training set

This is a strong baseline when there's enough historical data, but struggles
with cold-start (few/no votes) and can't learn team-specific nuances beyond
surface-level similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.data.parser import CodeReviewSample


@dataclass
class VoteRecord:
    embedding: np.ndarray
    text: str
    vote: float  # +1 for upvote, -1 for downvote


class TeamVectorStore:
    """Per-team FAISS index with upvote/downvote tracking."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.upvote_index = faiss.IndexFlatIP(dim)
        self.downvote_index = faiss.IndexFlatIP(dim)
        self.upvote_texts: list[str] = []
        self.downvote_texts: list[str] = []
        self.n_upvotes = 0
        self.n_downvotes = 0

    def add_vote(self, embedding: np.ndarray, text: str, is_upvote: bool):
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        if is_upvote:
            self.upvote_index.add(vec)
            self.upvote_texts.append(text)
            self.n_upvotes += 1
        else:
            self.downvote_index.add(vec)
            self.downvote_texts.append(text)
            self.n_downvotes += 1

    def query(
        self,
        embedding: np.ndarray,
        k: int = 5,
        upvote_weight: float = 1.0,
        downvote_weight: float = 0.8,
        exclude_self: bool = False,
    ) -> float:
        """Compute weighted similarity score for a query embedding.

        When exclude_self=True, drops the top result if its similarity > 0.99
        (likely the query vector itself in the index). Used during threshold
        tuning to avoid train-on-train leakage.
        """
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)

        extra = 1 if exclude_self else 0

        up_score = 0.0
        if self.n_upvotes > 0:
            k_fetch = min(k + extra, self.n_upvotes)
            distances, _ = self.upvote_index.search(vec, k_fetch)
            dists = distances[0, :k_fetch]
            if exclude_self and len(dists) > 0 and dists[0] > 0.99:
                dists = dists[1:]
            dists = dists[:k]
            if len(dists) > 0:
                up_score = float(np.mean(dists))

        down_score = 0.0
        if self.n_downvotes > 0:
            k_fetch = min(k + extra, self.n_downvotes)
            distances, _ = self.downvote_index.search(vec, k_fetch)
            dists = distances[0, :k_fetch]
            if exclude_self and len(dists) > 0 and dists[0] > 0.99:
                dists = dists[1:]
            dists = dists[:k]
            if len(dists) > 0:
                down_score = float(np.mean(dists))

        return upvote_weight * up_score - downvote_weight * down_score

    @property
    def total_votes(self) -> int:
        return self.n_upvotes + self.n_downvotes


class EmbeddingFilter:
    """Full embedding-based filtering pipeline.

    Faithfully implements Soohoon's approach:
      - Sentence-transformer embeddings
      - Per-team vector stores
      - Cosine similarity scoring
      - Threshold tuning on training data
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        upvote_weight: float = 1.0,
        downvote_weight: float = 0.8,
        min_votes: int = 3,
        batch_size: int = 256,
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.dim = dim
        self.upvote_weight = upvote_weight
        self.downvote_weight = downvote_weight
        self.min_votes = min_votes
        self.batch_size = batch_size
        self.stores: dict[str, TeamVectorStore] = {}
        self.thresholds: dict[str, float] = {}

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Batch-encode texts into normalized embeddings."""
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def build_store(self, team_name: str, vote_history: list[dict]):
        """Build the vector store for a team from its vote history.

        Each vote is a dict with keys: 'comment', 'vote' ('upvote'/'downvote').
        """
        store = TeamVectorStore(dim=self.dim)
        if not vote_history:
            self.stores[team_name] = store
            return

        texts = [v["comment"] for v in vote_history]
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
        """Find optimal threshold via grid search on training samples.

        Uses exclude_self=True so that each training sample's score is
        computed WITHOUT itself in the FAISS index. This prevents the
        self-similarity leak (cosine sim = 1.0 to itself) that would
        give artificially inflated scores and a useless threshold.
        """
        if team_name not in self.stores:
            self.thresholds[team_name] = 0.0
            return 0.0

        store = self.stores[team_name]
        if store.total_votes < self.min_votes:
            self.thresholds[team_name] = 0.0
            return 0.0

        texts = [s.comment for s in samples]
        labels = np.array([s.label for s in samples])
        embeddings = self._encode(texts)

        scores = np.array([
            store.query(
                emb, upvote_weight=self.upvote_weight,
                downvote_weight=self.downvote_weight,
                exclude_self=True,
            )
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
        self, team_name: str, comments: list[str]
    ) -> list[dict[str, float]]:
        """Score comments for a given team.

        Returns list of dicts with 'score' and 'decision' (1=surface, 0=filter).
        """
        store = self.stores.get(team_name)
        threshold = self.thresholds.get(team_name, 0.0)

        if store is None or store.total_votes < self.min_votes:
            return [{"score": 0.5, "decision": 1, "confidence": 0.0} for _ in comments]

        embeddings = self._encode(comments)
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

    def evaluate(
        self, team_name: str, samples: list[CodeReviewSample]
    ) -> dict[str, float]:
        """Evaluate the filter on a set of labeled samples."""
        comments = [s.comment for s in samples]
        labels = np.array([s.label for s in samples])
        predictions = self.predict(team_name, comments)
        pred_decisions = np.array([p["decision"] for p in predictions])
        pred_scores = np.array([p["score"] for p in predictions])

        tp = np.sum((pred_decisions == 1) & (labels == 1))
        fp = np.sum((pred_decisions == 1) & (labels == 0))
        fn = np.sum((pred_decisions == 0) & (labels == 1))
        tn = np.sum((pred_decisions == 0) & (labels == 0))

        accuracy = (tp + tn) / max(len(labels), 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        action_rate = np.mean(pred_decisions)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "action_rate": float(action_rate),
            "threshold": float(self.thresholds.get(team_name, 0.0)),
            "n_samples": len(labels),
            "n_votes": self.stores.get(team_name, TeamVectorStore()).total_votes,
        }

    def incremental_update(
        self, team_name: str, comment: str, was_addressed: bool
    ):
        """Add a single new vote to the store (simulates online learning)."""
        if team_name not in self.stores:
            self.stores[team_name] = TeamVectorStore(dim=self.dim)
        embedding = self._encode([comment])[0]
        self.stores[team_name].add_vote(embedding, comment, was_addressed)

    def save(self, path: str | Path):
        """Persist filter state to disk."""
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "thresholds": self.thresholds,
            "upvote_weight": self.upvote_weight,
            "downvote_weight": self.downvote_weight,
            "min_votes": self.min_votes,
        }
        with open(path / "filter_state.pkl", "wb") as f:
            pickle.dump(state, f)

        for team_name, store in self.stores.items():
            team_path = path / team_name
            team_path.mkdir(exist_ok=True)
            if store.n_upvotes > 0:
                up_vecs = faiss.rev_swig_ptr(
                    store.upvote_index.get_xb(), store.n_upvotes * self.dim
                ).reshape(store.n_upvotes, self.dim)
                np.save(team_path / "upvotes.npy", up_vecs)
            if store.n_downvotes > 0:
                down_vecs = faiss.rev_swig_ptr(
                    store.downvote_index.get_xb(), store.n_downvotes * self.dim
                ).reshape(store.n_downvotes, self.dim)
                np.save(team_path / "downvotes.npy", down_vecs)

        logger.info(f"Filter saved to {path}")
