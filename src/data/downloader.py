"""Download and cache the Microsoft CodeReviewer dataset.

The dataset contains ~116K code review samples from open-source GitHub projects.
We use two approaches:
  1. HuggingFace `datasets` library (preferred)
  2. Direct download from the CodeReviewer GitHub release (fallback)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset
from loguru import logger


_HF_DATASET_ID = "microsoft/CodeReviewer"
_GH_RAW_BASE = (
    "https://raw.githubusercontent.com/microsoft/CodeBERT/master/CodeReviewer/data"
)


def download_code_reviewer(
    cache_dir: str = "./data/cache",
    raw_dir: str = "./data/raw",
    force: bool = False,
) -> DatasetDict:
    """Download the CodeReviewer dataset, trying HuggingFace first then GitHub.

    Returns a DatasetDict with 'train', 'validation', 'test' splits.
    Each sample has keys: 'diff', 'comment', 'label'.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    processed_cache = cache_path / "code_reviewer_processed"

    if processed_cache.exists() and not force:
        logger.info(f"Loading cached dataset from {processed_cache}")
        return DatasetDict.load_from_disk(str(processed_cache))

    ds = _try_huggingface(cache_dir)
    if ds is None:
        ds = _try_github_raw(raw_dir, cache_dir)
    if ds is None:
        logger.warning("Both download methods failed — generating synthetic fallback")
        ds = _generate_synthetic_fallback()

    ds.save_to_disk(str(processed_cache))
    logger.success(f"Dataset saved to {processed_cache}")
    return ds


def _try_huggingface(cache_dir: str) -> DatasetDict | None:
    """Attempt download via HuggingFace datasets."""
    logger.info(f"Attempting HuggingFace download: {_HF_DATASET_ID}")
    try:
        raw = load_dataset(_HF_DATASET_ID, cache_dir=cache_dir, trust_remote_code=True)
        return _normalize_hf_dataset(raw)
    except Exception as e:
        logger.warning(f"HuggingFace download failed: {e}")
        # Try specific subsets
        for subset in ["cls", "msg", "ref"]:
            try:
                raw = load_dataset(
                    _HF_DATASET_ID, subset, cache_dir=cache_dir, trust_remote_code=True
                )
                return _normalize_hf_dataset(raw)
            except Exception:
                continue
    return None


def _normalize_hf_dataset(raw: DatasetDict) -> DatasetDict:
    """Normalize whatever schema HuggingFace gives us into our standard format."""
    from datasets import Dataset, DatasetDict

    def _normalize_sample(sample: dict) -> dict:
        diff = (
            sample.get("diff", "")
            or sample.get("patch", "")
            or sample.get("input", "")
            or sample.get("old_code", "")
        )
        comment = (
            sample.get("comment", "")
            or sample.get("msg", "")
            or sample.get("review_comment", "")
            or sample.get("output", "")
        )
        label_raw = sample.get("label", sample.get("target", sample.get("accept", -1)))
        if isinstance(label_raw, str):
            label = 1 if label_raw.strip().lower() in ("1", "true", "yes", "positive") else 0
        elif isinstance(label_raw, (int, float)):
            label = int(label_raw)
        else:
            label = 0
        return {"diff": str(diff), "comment": str(comment), "label": label}

    normalized = {}
    for split_name, split_data in raw.items():
        records = [_normalize_sample(s) for s in split_data]
        records = [r for r in records if len(r["diff"]) > 5 and len(r["comment"]) > 3]
        if records:
            normalized[split_name] = Dataset.from_list(records)

    if not normalized:
        raise ValueError("No valid samples found after normalization")

    if "train" not in normalized:
        only_key = list(normalized.keys())[0]
        full = normalized[only_key]
        splits = full.train_test_split(test_size=0.2, seed=42)
        val_test = splits["test"].train_test_split(test_size=0.5, seed=42)
        normalized = {
            "train": splits["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }

    logger.success(
        f"Normalized dataset: {', '.join(f'{k}={len(v)}' for k, v in normalized.items())}"
    )
    return DatasetDict(normalized)


def _try_github_raw(raw_dir: str, cache_dir: str) -> DatasetDict | None:
    """Fallback: download raw files from CodeReviewer GitHub repo."""
    import urllib.request

    logger.info("Attempting direct GitHub download")
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    splits = {"train": "train", "validation": "valid", "test": "test"}
    all_records: dict[str, list] = {}

    for our_split, gh_split in splits.items():
        for task in ["cls"]:
            url = f"{_GH_RAW_BASE}/{task}/{gh_split}.jsonl"
            local = raw_path / f"{task}_{gh_split}.jsonl"
            try:
                if not local.exists():
                    logger.info(f"Downloading {url}")
                    urllib.request.urlretrieve(url, str(local))
                records = []
                with open(local) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        diff = obj.get("input", obj.get("diff", ""))
                        comment = obj.get("output", obj.get("comment", ""))
                        label = int(obj.get("label", obj.get("target", 0)))
                        if len(diff) > 5 and len(comment) > 3:
                            records.append({"diff": diff, "comment": comment, "label": label})
                if records:
                    all_records[our_split] = records
                    logger.info(f"  {our_split}: {len(records)} samples")
            except Exception as e:
                logger.warning(f"  Failed for {url}: {e}")

    if not all_records:
        return None

    from datasets import Dataset, DatasetDict

    return DatasetDict({k: Dataset.from_list(v) for k, v in all_records.items()})


def _generate_synthetic_fallback() -> DatasetDict:
    """Generate synthetic code review data for development/testing.

    Uses realistic patterns so the pipeline can be validated even without
    the real dataset.
    """
    import random
    from datasets import Dataset, DatasetDict

    random.seed(42)

    DIFF_TEMPLATES = [
        "- old_value = get_input()\n+ new_value = sanitize(get_input())",
        "- for item in items:\n-     process(item)\n+ results = [process(i) for i in items]",
        "- password = request.form['pwd']\n+ password = hash_password(request.form['pwd'])",
        "- x = data\n+ x: List[int] = data",
        "- def calc(a,b):\n+ def calculate_total(amount: float, tax_rate: float) -> float:",
        "- conn = db.connect(host)\n+ conn = db.connect(host, ssl=True, timeout=30)",
        "- print(result)\n+ logger.info(f'Result: {{result}}')",
        "- import *\n+ from module import specific_function",
        "- cache = {}\n+ cache = LRUCache(maxsize=1024)",
        "- time.sleep(5)\n+ await asyncio.sleep(5)",
    ]

    COMMENT_TEMPLATES = {
        "security": [
            "This input should be sanitized to prevent injection attacks.",
            "Credentials should not be stored in plaintext.",
            "Missing authentication check before accessing user data.",
            "Use parameterized queries to prevent SQL injection.",
            "Consider adding rate limiting to this endpoint.",
        ],
        "style": [
            "Variable name 'x' is not descriptive enough.",
            "Please follow camelCase naming convention.",
            "Missing blank line between function definitions.",
            "This import should be at the top of the file.",
            "Inconsistent indentation — use 4 spaces.",
        ],
        "performance": [
            "This loop has O(n²) complexity, consider using a set.",
            "Cache this database query result to avoid repeated calls.",
            "Use bulk insert instead of inserting one by one.",
            "This could benefit from lazy loading.",
            "Consider using async I/O for this network call.",
        ],
        "pragmatic": [
            "nit: typo in variable name",
            "TODO: add error handling here",
            "This unused import can be removed.",
            "Minor: consider using f-string instead.",
            "Just add a null check here.",
        ],
        "thorough": [
            "Consider applying the Strategy pattern here to make this more extensible. For example, you could define an interface and swap implementations at runtime.",
            "This function violates the Single Responsibility Principle. I'd suggest splitting it into two: one for validation and one for processing. This improves testability.",
            "The error handling here could be more robust. Consider defining custom exception types and handling each case explicitly. Here's an example: ...",
            "This data structure choice will cause issues at scale. A B-tree or skip list would maintain O(log n) lookups while supporting range queries.",
            "I'd recommend adding comprehensive docstrings following NumPy style. Document parameters, return values, and include usage examples.",
        ],
    }

    records = []
    for _ in range(10000):
        team = random.choice(list(COMMENT_TEMPLATES.keys()))
        comment = random.choice(COMMENT_TEMPLATES[team])
        diff = random.choice(DIFF_TEMPLATES)
        addressed = random.random() < (0.7 if team in ("security", "thorough") else 0.4)
        records.append({
            "diff": diff,
            "comment": comment,
            "label": int(addressed),
        })

    random.shuffle(records)
    n = len(records)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    return DatasetDict({
        "train": Dataset.from_list(records[:train_end]),
        "validation": Dataset.from_list(records[train_end:val_end]),
        "test": Dataset.from_list(records[val_end:]),
    })
