"""Shared llama.cpp model loader with deterministic pins (spec 0.1, 0.2).

Single CPU instance serves both scoring and generation. n_threads must be
pinned explicitly (physical cores, not SMT threads) -- see env.json for the
recorded value on this host.
"""
import hashlib
import os
from pathlib import Path

from llama_cpp import Llama

MODELS_DIR = Path(os.environ.get("PYRAMID_MODELS_DIR", "/workspace/models"))
MODEL_FILENAME = os.environ.get("PYRAMID_MODEL", "Qwen2.5-3B-Instruct-Q5_K_M.gguf")

SEED = 0
TEMPERATURE = 0.0
TOP_K = 1
N_BATCH = 512
# Ryzen 7 PRO 8840U: 8 physical cores / 16 SMT threads. Pin to physical cores.
N_THREADS = int(os.environ.get("PYRAMID_N_THREADS", "8"))
N_CTX = int(os.environ.get("PYRAMID_N_CTX", "8192"))


def model_path() -> Path:
    p = MODELS_DIR / MODEL_FILENAME
    if not p.exists():
        raise FileNotFoundError(
            f"Pinned model not found: {p}. Download it into {MODELS_DIR} first."
        )
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(n_ctx: int = N_CTX, logits_all: bool = True) -> Llama:
    return Llama(
        model_path=str(model_path()),
        n_ctx=n_ctx,
        n_batch=N_BATCH,
        n_threads=N_THREADS,
        n_threads_batch=N_THREADS,
        seed=SEED,
        logits_all=logits_all,
        verbose=False,
    )
