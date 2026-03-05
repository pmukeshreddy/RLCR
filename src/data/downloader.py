"""Download the real GitHub Code Review dataset.

Uses ronantakizawa/github-codereview from HuggingFace:
  - 167K+ positive examples (real review comments that led to code changes)
  - 51K+ negative examples (code that passed review without comments)
  - Quality scores, comment types, 37 languages, 725 repos
  - Pre-split train/val/test by repository

NO synthetic fallback. Real data only.
"""

from __future__ import annotations

from pathlib import Path

from datasets import DatasetDict, load_dataset
from loguru import logger


_HF_DATASET_ID = "ronantakizawa/github-codereview"


def download_code_reviewer(
    cache_dir: str = "./data/cache",
    raw_dir: str = "./data/raw",
    force: bool = False,
) -> DatasetDict:
    """Download the real code review dataset from HuggingFace.

    Returns a DatasetDict with 'train', 'validation', 'test' splits.
    Each sample has: diff, comment, label, quality_score, comment_type,
    language, repo_name.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    processed_cache = cache_path / "code_reviewer_processed"

    if processed_cache.exists() and not force:
        logger.info(f"Loading cached dataset from {processed_cache}")
        return DatasetDict.load_from_disk(str(processed_cache))

    logger.info(f"Downloading real dataset: {_HF_DATASET_ID}")
    raw = load_dataset(_HF_DATASET_ID, cache_dir=cache_dir)
    ds = _normalize(raw)

    ds.save_to_disk(str(processed_cache))
    logger.success(f"Dataset saved to {processed_cache}")
    return ds


def _normalize(raw: DatasetDict) -> DatasetDict:
    """Normalize the github-codereview schema into our standard format.

    Mapping:
      diff_context  → diff  (the PR diff hunk)
      reviewer_comment → comment
      is_negative=False → label=1 (comment was useful, code changed)
      is_negative=True  → label=0 (no issues, clean code)
      quality_score, comment_type, language, repo_name → metadata
    """
    from datasets import Dataset

    normalized = {}
    for split_name, split_data in raw.items():
        records = []
        n_skipped = 0
        for row in split_data:
            diff = row.get("diff_context", "") or ""
            if not diff or len(diff.strip()) < 10:
                diff = row.get("before_code", "") or ""
            comment = row.get("reviewer_comment", "") or ""
            is_negative = row.get("is_negative", False)

            if len(diff.strip()) < 10:
                n_skipped += 1
                continue
            if len(comment.strip()) < 3:
                n_skipped += 1
                continue

            label = 0 if is_negative else 1
            quality_score = row.get("quality_score", 0.0) or 0.0
            comment_type = row.get("comment_type", "none") or "none"
            language = row.get("language", "unknown") or "unknown"
            repo_name = row.get("repo_name", "") or ""

            records.append({
                "diff": diff.strip(),
                "comment": comment.strip(),
                "label": label,
                "quality_score": float(quality_score),
                "comment_type": str(comment_type),
                "language": str(language),
                "repo_name": str(repo_name),
            })

        if records:
            normalized[split_name] = Dataset.from_list(records)
            n_pos = sum(1 for r in records if r["label"] == 1)
            n_neg = len(records) - n_pos
            logger.info(
                f"  {split_name}: {len(records)} samples "
                f"(pos={n_pos}, neg={n_neg}, skipped={n_skipped})"
            )

    if not normalized:
        raise RuntimeError(
            f"Failed to load any data from {_HF_DATASET_ID}. "
            "Check your internet connection and HuggingFace access."
        )

    logger.success(
        f"Loaded real dataset: "
        f"{sum(len(v) for v in normalized.values())} total samples "
        f"from {_HF_DATASET_ID}"
    )
    return DatasetDict(normalized)
