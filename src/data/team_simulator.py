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
        target_positive_rate: float = 0.6,
    ) -> dict[str, Team]:
        """Assign samples to teams with balanced label distribution.

        1. Route by comment_type (positive samples get meaningful types).
        2. Collect ALL label=0 samples into a shared negative pool.
        3. Distribute negatives across teams to hit target_positive_rate.

        Without step 2-3, most teams would be ~100% positive (comment_type
        only exists on label=1 samples), making the filtering task trivial.
        """
        logger.info(f"Assigning {len(samples)} samples to {len(self.teams)} teams")

        team_positives: dict[str, list[CodeReviewSample]] = defaultdict(list)
        negative_pool: list[CodeReviewSample] = []
        unassigned_pos: list[CodeReviewSample] = []

        for sample in samples:
            if sample.label == 0:
                negative_pool.append(sample)
                continue

            comment_type = getattr(sample, "comment_type", None) or ""
            team_name = COMMENT_TYPE_TO_TEAM.get(comment_type.lower().strip())

            if team_name and team_name in self.teams:
                team_positives[team_name].append(sample)
            else:
                team_name = self._keyword_fallback(sample)
                if team_name:
                    team_positives[team_name].append(sample)
                else:
                    unassigned_pos.append(sample)

        if unassigned_pos:
            self.rng.shuffle(unassigned_pos)
            team_names = list(self.teams.keys())
            for i, sample in enumerate(unassigned_pos):
                team_positives[team_names[i % len(team_names)]].append(sample)

        logger.info(
            f"  Positive pool: {sum(len(v) for v in team_positives.values())} "
            f"across {len(team_positives)} teams"
        )
        logger.info(f"  Negative pool: {len(negative_pool)} samples to distribute")

        team_assignments = self._balance_labels(
            team_positives, negative_pool, target_positive_rate
        )

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

    def _balance_labels(
        self,
        team_positives: dict[str, list[CodeReviewSample]],
        negative_pool: list[CodeReviewSample],
        target_positive_rate: float,
    ) -> dict[str, list[CodeReviewSample]]:
        """Distribute negatives across teams to achieve target label balance.

        For each team with N positives, adds N * (1 - rate) / rate negatives
        so the final ratio is approximately target_positive_rate.
        """
        self.rng.shuffle(negative_pool)
        neg_idx = 0
        assignments: dict[str, list[CodeReviewSample]] = {}

        for team_name in self.teams:
            positives = team_positives.get(team_name, [])
            n_pos = len(positives)
            if n_pos == 0:
                assignments[team_name] = []
                continue

            n_neg_needed = int(n_pos * (1 - target_positive_rate) / target_positive_rate)
            n_neg_available = len(negative_pool) - neg_idx
            n_neg = min(n_neg_needed, n_neg_available)

            negatives = negative_pool[neg_idx:neg_idx + n_neg]
            neg_idx += n_neg

            combined = positives + negatives
            self.rng.shuffle(combined)
            assignments[team_name] = combined

            actual_rate = n_pos / len(combined) if combined else 0
            logger.info(
                f"  {team_name}: {n_pos} pos + {n_neg} neg "
                f"= {len(combined)} total (positive_rate={actual_rate:.2f})"
            )

        remaining = len(negative_pool) - neg_idx
        if remaining > 0:
            logger.info(f"  {remaining} negatives unused (pool exhausted target)")

        return assignments

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
        """Split each team's samples into train and test, preserving label ratio.

        Stratified split: positives and negatives are split independently
        so both train and test maintain the same positive/negative ratio.
        """
        for team_name, team_samples in assignments.items():
            positives = [s for s in team_samples if s.label == 1]
            negatives = [s for s in team_samples if s.label == 0]
            self.rng.shuffle(positives)
            self.rng.shuffle(negatives)

            n_total = len(team_samples)
            n_train = self.rng.randint(train_range[0], train_range[1])
            n_train = min(n_train, n_total - min_test)
            n_train = max(n_train, min(20, n_total // 2))

            pos_rate = len(positives) / n_total if n_total > 0 else 0.5
            n_train_pos = max(1, int(n_train * pos_rate))
            n_train_neg = n_train - n_train_pos

            n_train_pos = min(n_train_pos, len(positives))
            n_train_neg = min(n_train_neg, len(negatives))

            train = positives[:n_train_pos] + negatives[:n_train_neg]
            test = positives[n_train_pos:] + negatives[n_train_neg:]

            self.rng.shuffle(train)
            self.rng.shuffle(test)

            self.teams[team_name].train_samples = train
            self.teams[team_name].test_samples = test

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
