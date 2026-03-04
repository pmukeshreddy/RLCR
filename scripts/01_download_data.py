#!/usr/bin/env python3
"""Step 1: Download Microsoft CodeReviewer dataset and build reward labels.

Downloads ~116K code review samples from:
  - HuggingFace (primary)
  - GitHub CodeReviewer repo (fallback)
  - Synthetic generation (dev fallback)

Parses into (diff, comment, was_addressed) triplets, cleans, and tokenizes
with truncation to 512 tokens.

Usage:
    python scripts/01_download_data.py [--force] [--config configs/default.yaml]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import load_config, set_seed
from src.data.downloader import download_code_reviewer
from src.data.parser import parse_to_triplets, samples_to_dataset

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Download and process CodeReviewer dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config)
    set_seed(cfg)

    console.rule("[bold blue]Step 1: Data Pipeline[/bold blue]")
    console.print("Downloading Microsoft CodeReviewer dataset (~116K samples)...")

    ds = download_code_reviewer(
        cache_dir=cfg.data.cache_dir,
        raw_dir=cfg.data.raw_dir,
        force=args.force,
    )

    console.rule("[bold]Parsing into triplets[/bold]")
    all_samples = {}
    for split_name in ds:
        samples = parse_to_triplets(
            ds[split_name],
            max_tokens=cfg.data.max_tokens,
            min_diff_length=cfg.data.min_diff_length,
            min_comment_length=cfg.data.min_comment_length,
        )
        all_samples[split_name] = samples

    table = Table(title="Dataset Summary")
    table.add_column("Split", style="cyan")
    table.add_column("Samples", style="green")
    table.add_column("Addressed (label=1)", style="yellow")
    table.add_column("Ignored (label=0)", style="red")
    table.add_column("Address Rate", style="magenta")

    for split_name, samples in all_samples.items():
        n = len(samples)
        n_addressed = sum(1 for s in samples if s.label == 1)
        n_ignored = n - n_addressed
        rate = n_addressed / max(n, 1)
        table.add_row(split_name, str(n), str(n_addressed), str(n_ignored), f"{rate:.3f}")

    console.print(table)

    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in all_samples.items():
        dataset = samples_to_dataset(samples)
        dataset.save_to_disk(str(processed_dir / split_name))
        logger.info(f"Saved {split_name}: {len(samples)} samples → {processed_dir / split_name}")

    console.print("\n[bold green]✓ Step 1 complete![/bold green]")
    console.print(f"  Processed data saved to: {processed_dir}")


if __name__ == "__main__":
    main()
