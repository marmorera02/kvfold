"""Environment fingerprinting and result I/O (spec 0.2, 0.5).

Every result JSON embeds the env fingerprint; results with differing
fingerprints must never be compared.
"""
import json
import os
import platform
from pathlib import Path

import llama_cpp

from . import model as model_mod

RESULTS_DIR = Path(os.environ.get("PYRAMID_RESULTS_DIR", "/workspace/results"))


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or "unknown"


def build_env_fingerprint() -> dict:
    mp = model_mod.model_path()
    return {
        "model_filename": mp.name,
        "model_sha256": model_mod.sha256_file(mp),
        "llama_cpp_python_version": getattr(llama_cpp, "__version__", "unknown"),
        "n_ctx": model_mod.N_CTX,
        "n_batch": model_mod.N_BATCH,
        "n_threads": model_mod.N_THREADS,
        "seed": model_mod.SEED,
        "temperature": model_mod.TEMPERATURE,
        "top_k": model_mod.TOP_K,
        "cpu_model": cpu_model(),
        "container_image_digest": os.environ.get("PYRAMID_IMAGE_DIGEST"),
    }


def write_env_json() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = build_env_fingerprint()
    with open(RESULTS_DIR / "env.json", "w") as f:
        json.dump(env, f, indent=2)
    return env


def save_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"env": build_env_fingerprint(), **payload}
    path = RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)
