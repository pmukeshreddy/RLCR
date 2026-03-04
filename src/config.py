"""Configuration management for RLCR."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(overrides: dict[str, Any] | None = None, config_path: str | None = None) -> DictConfig:
    """Load configuration from YAML with optional overrides."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def get_device(cfg: DictConfig) -> str:
    import torch

    device = cfg.project.get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(cfg: DictConfig) -> None:
    import random

    import numpy as np
    import torch

    seed = cfg.project.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
