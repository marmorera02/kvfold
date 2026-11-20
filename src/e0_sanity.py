"""E0 -- Infrastructure sanity (spec section E0).

Proves: (a) per-token logprobs/ranks are obtainable, (b) they and greedy
generations are bit-identical across two fresh processes.

Usage:
    python -m src.e0_sanity                       # orchestrator: spawns fresh
                                                    # worker processes, compares,
                                                    # writes results/e0.json
    python -m src.e0_sanity --worker score --out FILE
    python -m src.e0_sanity --worker generate --out FILE
"""
import argparse
import json
import sys
import time
from pathlib import Path

from src.common import io as io_mod

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIXED_PARAGRAPH_PATH = DATA_DIR / "e0_fixed_paragraph.txt"
FIXED_PROMPT = (
    "Continue the following story in the same voice, for a few sentences.\n\n"
    "The lighthouse keeper opened his second notebook and wrote:"
)
SCORE_TOKEN_COUNT = 500
GENERATE_TOKEN_COUNT = 100


def _first_n_tokens_text(llm, n: int) -> str:
    raw = FIXED_PARAGRAPH_PATH.read_text(encoding="utf-8")
    tokens = llm.tokenize(raw.encode("utf-8"), add_bos=False)
    tokens = tokens[:n]
    return llm.detokenize(tokens).decode("utf-8", errors="ignore")


def _run_score(out_path: Path) -> None:
    from src.common import model as model_mod
    from src.common.scoring import score_tokens

    llm = model_mod.load_model(logits_all=True)
    text = _first_n_tokens_text(llm, SCORE_TOKEN_COUNT)

    t0 = time.perf_counter()
    scores = score_tokens(llm, text=text, context="")
    elapsed = time.perf_counter() - t0

    payload = {
        "n_tokens_scored": len(scores),
        "elapsed_s": elapsed,
        "tokens_per_s_prefill": len(scores) / elapsed if elapsed > 0 else None,
        "scores": [
            {
                "token_id": s.token_id,
                "logprob_true": s.logprob_true,
                "rank_true": s.rank_true,
                "entropy": s.entropy,
            }
            for s in scores
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))


def _run_generate(out_path: Path) -> None:
    from src.common import model as model_mod

    llm = model_mod.load_model(logits_all=False, n_ctx=2048)

    t0 = time.perf_counter()
    result = llm(
        FIXED_PROMPT,
        max_tokens=GENERATE_TOKEN_COUNT,
        temperature=model_mod.TEMPERATURE,
        top_k=model_mod.TOP_K,
        seed=model_mod.SEED,
        echo=False,
    )
    elapsed = time.perf_counter() - t0

    text_out = result["choices"][0]["text"]
    n_out_tokens = len(llm.tokenize(text_out.encode("utf-8"), add_bos=False))

    payload = {
        "text": text_out,
        "elapsed_s": elapsed,
        "tokens_per_s_decode": n_out_tokens / elapsed if elapsed > 0 else None,
        "n_tokens_generated": n_out_tokens,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))


def _spawn_worker(worker: str, out_path: Path) -> None:
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "src.e0_sanity", "--worker", worker, "--out", str(out_path)],
        check=True,
    )


def _orchestrate() -> None:
    results_dir = io_mod.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    score_run1 = results_dir / "e0_score_run1.json"
    score_run2 = results_dir / "e0_score_run2.json"
    gen_run1 = results_dir / "e0_generate_run1.json"
    gen_run2 = results_dir / "e0_generate_run2.json"

    print("E0: scoring run 1 (fresh process)...")
    _spawn_worker("score", score_run1)
    print("E0: scoring run 2 (fresh process)...")
    _spawn_worker("score", score_run2)
    print("E0: generating run 1 (fresh process)...")
    _spawn_worker("generate", gen_run1)
    print("E0: generating run 2 (fresh process)...")
    _spawn_worker("generate", gen_run2)

    s1 = json.loads(score_run1.read_text())
    s2 = json.loads(score_run2.read_text())
    g1 = json.loads(gen_run1.read_text())
    g2 = json.loads(gen_run2.read_text())

    rank_stream_1 = [t["rank_true"] for t in s1["scores"]]
    rank_stream_2 = [t["rank_true"] for t in s2["scores"]]
    scores_identical = rank_stream_1 == rank_stream_2
    logprobs_identical = all(
        a["logprob_true"] == b["logprob_true"] for a, b in zip(s1["scores"], s2["scores"])
    )
    generation_identical = g1["text"] == g2["text"]

    pass_ = scores_identical and logprobs_identical and generation_identical

    env = io_mod.build_env_fingerprint()
    summary = {
        "env": env,
        "pass": pass_,
        "rank_streams_identical": scores_identical,
        "logprobs_bit_identical": logprobs_identical,
        "generation_identical": generation_identical,
        "prefill_tokens_per_s": s1["tokens_per_s_prefill"],
        "decode_tokens_per_s": g1["tokens_per_s_decode"],
        "score_wall_clock_s": s1["elapsed_s"],
        "n_tokens_scored": s1["n_tokens_scored"],
        "n_tokens_generated": g1["n_tokens_generated"],
        "generated_text_sample": g1["text"],
    }

    out_path = results_dir / "e0.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"rank streams identical:    {scores_identical}")
    print(f"logprobs bit-identical:    {logprobs_identical}")
    print(f"generation identical:      {generation_identical}")
    print(f"prefill tokens/s:          {s1['tokens_per_s_prefill']:.1f}")
    print(f"decode tokens/s:           {g1['tokens_per_s_decode']:.1f}")
    print(f"scoring wall-clock (500 tok): {s1['elapsed_s']:.2f}s")
    print(f"\nE0 {'PASS' if pass_ else 'FAIL'} -- see {out_path}")

    if not pass_:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["score", "generate"], default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker == "score":
        _run_score(args.out)
    elif args.worker == "generate":
        _run_generate(args.out)
    else:
        _orchestrate()


if __name__ == "__main__":
    main()
