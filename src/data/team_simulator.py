"""Simulate 5 development teams by clustering code review comments.

Teams are defined by their comment preferences:
  - security: Prioritizes vulnerability detection and secure coding
  - style: Enforces naming conventions, formatting, consistency
  - performance: Focuses on algorithmic efficiency and resource usage
  - pragmatic: Values quick, actionable, low-ceremony feedback
  - thorough: Provides detailed, educational, principle-driven reviews

Clustering uses keyword matching with TF-IDF weighting, plus comment-length
heuristics for pragmatic (short) vs thorough (long) differentiation.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.parser import CodeReviewSample


@dataclass
class Team:
    """A simulated development team with its own review preferences."""

    name: str
    description: str
    keywords: list[str]
    train_samples: list[CodeReviewSample] = field(default_factory=list)
    test_samples: list[CodeReviewSample] = field(default_factory=list)
    vote_history: list[dict] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.train_samples) + len(self.test_samples)

    def summary(self) -> dict:
        train_labels = [s.label for s in self.train_samples]
        test_labels = [s.label for s in self.test_samples]
        return {
            "name": self.name,
            "train_count": len(self.train_samples),
            "test_count": len(self.test_samples),
            "train_positive_rate": np.mean(train_labels) if train_labels else 0,
            "test_positive_rate": np.mean(test_labels) if test_labels else 0,
        }


class TeamSimulator:
    """Assigns code review samples to simulated teams and manages splits."""

    def __init__(self, team_configs: list[dict], seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.teams: dict[str, Team] = {}
        for tc in team_configs:
            self.teams[tc["name"]] = Team(
                name=tc["name"],
                description=tc["description"],
                keywords=[kw.lower() for kw in tc["keywords"]],
            )
        self._build_keyword_index()

    def _build_keyword_index(self):
        """Pre-compile keyword patterns for fast matching."""
        import re
        self._patterns: dict[str, list[re.Pattern]] = {}
        for name, team in self.teams.items():
            self._patterns[name] = [
                re.compile(re.escape(kw), re.IGNORECASE) for kw in team.keywords
            ]

    def assign_samples(
        self,
        samples: list[CodeReviewSample],
        train_range: tuple[int, int] = (20, 50),
        min_test: int = 200,
        fallback_random: bool = True,
    ) -> dict[str, Team]:
        """Assign samples to teams based on comment content analysis.

        Uses a multi-signal scoring approach:
          1. Keyword match count (weighted by TF-IDF rarity)
          2. Comment length (pragmatic=short, thorough=long)
          3. Sentiment/tone patterns
        """
        logger.info(f"Assigning {len(samples)} samples to {len(self.teams)} teams")

        team_assignments: dict[str, list[CodeReviewSample]] = defaultdict(list)
        unassigned = []

        keyword_scores = self._compute_keyword_scores(samples)
        length_scores = self._compute_length_scores(samples)

        for i, sample in enumerate(samples):
            combined = {}
            for team_name in self.teams:
                kw = keyword_scores.get(team_name, {}).get(i, 0.0)
                ln = length_scores.get(team_name, {}).get(i, 0.0)
                combined[team_name] = 0.7 * kw + 0.3 * ln

            best_team = max(combined, key=combined.get)
            best_score = combined[best_team]

            if best_score > 0.1:
                team_assignments[best_team].append(sample)
            else:
                unassigned.append(sample)

        if fallback_random and unassigned:
            logger.info(f"Randomly assigning {len(unassigned)} unmatched samples")
            self.rng.shuffle(unassigned)
            team_names = list(self.teams.keys())
            for i, sample in enumerate(unassigned):
                team_assignments[team_names[i % len(team_names)]].append(sample)

        self._rebalance_if_needed(team_assignments, min_test)
        self._create_splits(team_assignments, train_range, min_test)
        self._build_vote_histories()

        for name, team in self.teams.items():
            logger.info(
                f"  {name}: {len(team.train_samples)} train, "
                f"{len(team.test_samples)} test "
                f"(+rate: {team.summary()['test_positive_rate']:.2f})"
            )

        return self.teams

    def _compute_keyword_scores(
        self, samples: list[CodeReviewSample]
    ) -> dict[str, dict[int, float]]:
        """Score each sample against each team's keywords."""
        scores: dict[str, dict[int, float]] = defaultdict(dict)
        for i, sample in enumerate(samples):
            text = sample.comment.lower()
            for team_name, patterns in self._patterns.items():
                match_count = sum(1 for p in patterns if p.search(text))
                scores[team_name][i] = match_count / max(len(patterns), 1)
        return scores

    def _compute_length_scores(
        self, samples: list[CodeReviewSample]
    ) -> dict[str, dict[int, float]]:
        """Score based on comment length (pragmatic=short, thorough=long)."""
        scores: dict[str, dict[int, float]] = defaultdict(dict)
        lengths = [len(s.comment) for s in samples]
        if not lengths:
            return scores
        median_len = np.median(lengths)
        p25 = np.percentile(lengths, 25)
        p75 = np.percentile(lengths, 75)

        for i, sample in enumerate(samples):
            clen = len(sample.comment)
            for team_name in self.teams:
                if team_name == "pragmatic":
                    scores[team_name][i] = max(0, 1.0 - clen / max(p25 * 2, 1))
                elif team_name == "thorough":
                    scores[team_name][i] = max(0, min(1.0, (clen - p75) / max(p75, 1)))
                else:
                    scores[team_name][i] = 0.0
        return scores

    def _rebalance_if_needed(
        self,
        assignments: dict[str, list[CodeReviewSample]],
        min_test: int,
    ):
        """Redistribute samples from over-represented teams to under-represented ones."""
        while True:
            sizes = {k: len(v) for k, v in assignments.items()}
            min_team = min(sizes, key=sizes.get)
            max_team = max(sizes, key=sizes.get)
            if sizes[min_team] >= min_test + 20:
                break
            if sizes[max_team] <= min_test + 50:
                break
            n_move = min(50, (sizes[max_team] - sizes[min_team]) // 2)
            if n_move <= 0:
                break
            moved = assignments[max_team][-n_move:]
            assignments[max_team] = assignments[max_team][:-n_move]
            assignments[min_team].extend(moved)

    def _create_splits(
        self,
        assignments: dict[str, list[CodeReviewSample]],
        train_range: tuple[int, int],
        min_test: int,
    ):
        """Split each team's samples into train and test sets."""
        for team_name, team_samples in assignments.items():
            self.rng.shuffle(team_samples)
            n_train = self.rng.randint(train_range[0], train_range[1])
            n_train = min(n_train, len(team_samples) - min_test)
            n_train = max(n_train, min(20, len(team_samples) // 2))

            self.teams[team_name].train_samples = team_samples[:n_train]
            self.teams[team_name].test_samples = team_samples[n_train:]

    def _build_vote_histories(self):
        """Simulate upvote/downvote history from training labels.

        This provides the embedding baseline with realistic historical feedback.
        """
        for team in self.teams.values():
            team.vote_history = []
            for sample in team.train_samples:
                team.vote_history.append({
                    "comment": sample.comment,
                    "diff": sample.diff,
                    "vote": "upvote" if sample.label == 1 else "downvote",
                    "score": 1.0 if sample.label == 1 else -1.0,
                })

    def save(self, path: str | Path):
        """Persist team assignments to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, team in self.teams.items():
            team_dir = path / name
            team_dir.mkdir(exist_ok=True)

            with open(team_dir / "train.jsonl", "w") as f:
                for s in team.train_samples:
                    f.write(json.dumps(s.to_dict()) + "\n")
            with open(team_dir / "test.jsonl", "w") as f:
                for s in team.test_samples:
                    f.write(json.dumps(s.to_dict()) + "\n")
            with open(team_dir / "votes.json", "w") as f:
                json.dump(team.vote_history, f, indent=2)
            with open(team_dir / "meta.json", "w") as f:
                json.dump(team.summary(), f, indent=2)

        logger.success(f"Teams saved to {path}")

    @classmethod
    def load(cls, path: str | Path, team_configs: list[dict]) -> "TeamSimulator":
        """Load team assignments from disk."""
        path = Path(path)
        sim = cls(team_configs)
        for name in sim.teams:
            team_dir = path / name
            if not team_dir.exists():
                logger.warning(f"Team dir not found: {team_dir}")
                continue

            train_file = team_dir / "train.jsonl"
            test_file = team_dir / "test.jsonl"
            votes_file = team_dir / "votes.json"

            if train_file.exists():
                with open(train_file) as f:
                    sim.teams[name].train_samples = [
                        CodeReviewSample(**json.loads(line)) for line in f if line.strip()
                    ]
            if test_file.exists():
                with open(test_file) as f:
                    sim.teams[name].test_samples = [
                        CodeReviewSample(**json.loads(line)) for line in f if line.strip()
                    ]
            if votes_file.exists():
                with open(votes_file) as f:
                    sim.teams[name].vote_history = json.load(f)

        return sim
