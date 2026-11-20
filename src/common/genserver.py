"""Generation client for the host GPU llama-server (E2 compute-migration
directive, G1). Hard-pins the same determinism settings as CPU generation.
Never silently falls back to CPU on connection failure -- that would produce
an undocumented mixed-backend artifact; callers must handle the raised
error explicitly (e.g. by failing the run).
"""
import os
import time

import requests

GENSERVER_URL = os.environ.get("PYRAMID_GENSERVER_URL", "http://host.docker.internal:8080")
MAX_RETRIES = 3
RETRY_BACKOFF_S = 2.0


def generate(prompt: str, max_tokens: int, seed: int = 0, temperature: float = 0.0, top_k: int = 1) -> str:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(f"{GENSERVER_URL}/v1/completions", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["text"]
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    raise RuntimeError(
        f"GPU generation server at {GENSERVER_URL} unreachable after {MAX_RETRIES} attempts. "
        f"Refusing to fall back to CPU generation silently -- a mixed-backend level set "
        f"would be an undocumented artifact. Fix the server or switch generation_backend "
        f"to cpu_local explicitly."
    ) from last_exc


def health_check() -> bool:
    try:
        resp = requests.get(f"{GENSERVER_URL}/v1/models", timeout=10)
        return resp.status_code == 200
    except requests.RequestException:
        return False
