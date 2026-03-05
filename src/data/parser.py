"""Parse raw dataset into cleaned (diff, comment, label) triplets.

Handles tokenization, truncation, and quality filtering.
Now carries comment_type metadata for real team assignment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterator

from datasets import Dataset
from loguru import logger


@dataclass
class CodeReviewSample:
    """A single code review sample ready for model consumption."""

    diff: str
    comment: str
    label: int  # 1 = addressed, 0 = ignored/clean
    diff_tokens: int = 0
    comment_tokens: int = 0
    comment_type: str = ""
    quality_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def parse_to_triplets(
    dataset: Dataset,
    max_tokens: int = 512,
    min_diff_length: int = 10,
    min_comment_length: int = 5,
    tokenizer_name: str | None = None,
) -> list[CodeReviewSample]:
    """Parse a HuggingFace Dataset into cleaned CodeReviewSample objects.

    Steps:
      1. Filter by minimum length requirements
      2. Clean diff formatting
      3. Clean comment text
      4. Tokenize and truncate to max_tokens
      5. Validate label values
      6. Carry comment_type and quality_score metadata
    """
    tokenizer = _get_tokenizer(tokenizer_name)
    samples = []
    n_filtered = {"short_diff": 0, "short_comment": 0, "bad_label": 0, "encoding": 0}

    for row in dataset:
        diff = _clean_diff(row["diff"])
        comment = _clean_comment(row["comment"])
        label = row["label"]

        if len(diff) < min_diff_length:
            n_filtered["short_diff"] += 1
            continue
        if len(comment) < min_comment_length:
            n_filtered["short_comment"] += 1
            continue
        if label not in (0, 1):
            n_filtered["bad_label"] += 1
            continue

        try:
            diff, diff_tok_count = _truncate(diff, max_tokens, tokenizer)
            comment, comment_tok_count = _truncate(comment, max_tokens // 4, tokenizer)
        except Exception:
            n_filtered["encoding"] += 1
            continue

        comment_type = row.get("comment_type", "") or ""
        quality_score = row.get("quality_score", 0.0) or 0.0

        samples.append(CodeReviewSample(
            diff=diff,
            comment=comment,
            label=label,
            diff_tokens=diff_tok_count,
            comment_tokens=comment_tok_count,
            comment_type=str(comment_type),
            quality_score=float(quality_score),
        ))

    total = len(dataset)
    kept = len(samples)
    logger.info(
        f"Parsed {kept}/{total} samples "
        f"(filtered: {n_filtered})"
    )
    return samples


def _get_tokenizer(name: str | None):
    """Load a tokenizer for truncation. Falls back to a simple whitespace splitter."""
    if name:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Could not load tokenizer {name}: {e}")
    return None


def _truncate(text: str, max_tokens: int, tokenizer) -> tuple[str, int]:
    """Truncate text to max_tokens. Returns (truncated_text, token_count)."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
        return tokenizer.decode(ids, skip_special_tokens=True), len(ids)

    words = text.split()
    if len(words) > max_tokens:
        words = words[:max_tokens]
    return " ".join(words), len(words)


def _clean_diff(diff: str) -> str:
    """Clean a code diff for model consumption."""
    diff = diff.strip()
    diff = re.sub(r"diff --git .+?\n", "", diff)
    diff = re.sub(r"index [0-9a-f]+\.\.[0-9a-f]+ \d+\n", "", diff)
    diff = re.sub(r"\n{3,}", "\n\n", diff)
    lines = diff.split("\n")
    cleaned = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _clean_comment(comment: str) -> str:
    """Clean a review comment."""
    comment = comment.strip()
    comment = re.sub(r"http\S+", "[URL]", comment)
    comment = re.sub(r"@\w+", "@user", comment)
    comment = re.sub(r"\s+", " ", comment)
    return comment


def samples_to_dataset(samples: list[CodeReviewSample]) -> Dataset:
    """Convert a list of CodeReviewSample back into a HuggingFace Dataset."""
    records = [s.to_dict() for s in samples]
    return Dataset.from_list(records)


def stream_triplets(
    dataset: Dataset,
    max_tokens: int = 512,
    batch_size: int = 1000,
) -> Iterator[list[CodeReviewSample]]:
    """Stream triplets in batches for memory-efficient processing."""
    batch = []
    for row in dataset:
        diff = _clean_diff(row["diff"])
        comment = _clean_comment(row["comment"])
        label = row["label"]
        if len(diff) >= 10 and len(comment) >= 5 and label in (0, 1):
            diff, dtok = _truncate(diff, max_tokens, None)
            comment, ctok = _truncate(comment, max_tokens // 4, None)
            batch.append(CodeReviewSample(
                diff=diff, comment=comment, label=label,
                diff_tokens=dtok, comment_tokens=ctok,
                comment_type=str(row.get("comment_type", "")),
                quality_score=float(row.get("quality_score", 0.0)),
            ))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
