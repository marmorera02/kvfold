"""Determinism helpers (spec 0.2).

Every experiment pins seed=0, temperature=0.0, top_k=1 and a fixed thread
count. This module centralizes those pins plus the cross-run identity check
used by E0 and reused wherever bit-identical reproduction must be verified.
"""
import os
import random

import numpy as np

SEED = 0
TEMPERATURE = 0.0
TOP_K = 1


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def assert_identical(a, b, label: str) -> None:
    if a != b:
        raise AssertionError(f"{label}: outputs differ between runs")
