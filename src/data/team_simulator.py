"""Simulate 5 development teams using real comment_type labels.

Teams map directly to the comment_type field from the dataset:
  - security: security-related comments (bug + security types)
  - style: style/formatting comments
  - performance: performance optimization comments
  - pragmatic: nitpicks, quick fixes, short comments
  - thorough: suggestions, refactors, questions, detailed reviews

This uses REAL comment categories from human reviewers, not synthetic
keyword matching.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.data.parser import CodeReviewSample


COMMENT_TYPE_TO_TEAM = {
    "security": "security",
    "bug": "security",
    "style": "style",
    "performance": "performance",
    "nitpick": "pragmatic",
    "none": "pragmatic",
    "suggestion": "thorough",
    "refactor": "thorough",
    "question": "thorough",
}


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
    """Assigns code review samples to teams using real comment_type labels."""

    def __init__(self, team_configs: list[dict], seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.teams: dict[str, Team] = {}
        for tc in team_configs:
            self.teams[tc["name"]] = Team(
                name=tc["name"],
                description=tc["description"],
                keywords=[kw.lower() for kw in tc.get("keywords", [])],
            )

    def assign_samples(
        self,
        samples: list[CodeReviewSample],
        train_range: tuple[int, int] = (20, 50),
        min_test: int = 200,
        fallback_random: bool = True,
    ) -> dict[str, Team]:
        """Assign samples to teams based on comment_type metadata.

        Uses COMMENT_TYPE_TO_TEAM mapping for direct assignment.
        Falls back to keyword matching for samples without comment_type,
        and random assignment for the remainder.
        """
        logger.info(f"Assigning {len(samples)} samples to {len(self.teams)} teams")

        team_assignments: dict[str, list[CodeReviewSample]] = defaultdict(list)
        unassigned = []

        for sample in samples:
            comment_type = getattr(sample, "comment_type", None) or ""
            team_name = COMMENT_TYPE_TO_TEAM.get(comment_type.lower().strip())

            if team_name and team_name in self.teams:
                team_assignments[team_name].append(sample)
            else:
                team_name = self._keyword_fallback(sample)
                if team_name:
                    team_assignments[team_name].append(sample)
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

    def _keyword_fallback(self, sample: CodeReviewSample) -> str | None:
        """Fall back to keyword matching if comment_type is unavailable."""
        text = sample.comment.lower()
        best_team = None
        best_score = 0.0
        for name, team in self.teams.items():
            if not team.keywords:
                continue
            hits = sum(1 for kw in team.keywords if kw in text)
            score = hits / len(team.keywords)
            if score > best_score and score > 0.05:
                best_score = score
                best_team = name
        return best_team

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
        """Build vote history from training labels.

        For real data, label=1 means the reviewer's comment led to a code change
        (addressed = upvote), label=0 means the code was clean / no change needed
        (downvote for surfacing unnecessary comments).
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
