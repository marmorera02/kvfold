"""Cycle 3, Iteration 1 -- E6.b: reload-vs-reprefill speed table (P-N).

For each chunk size, compares (a) prefilling that many tokens from scratch
against (b) restoring a pre-saved KV state of the same size, both in fresh
processes, 3 repetitions, medians. CPU backend only (mandatory per
COMPUTE_MIGRATION_DIRECTIVE -- GPU generation failed its G2 determinism
gate, and GPU scoring/state-restore was never attempted).
"""
import argparse
import json
import pickle
import time
from pathlib import Path

from src.common import io as io_mod
from src.common import model as model_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CHUNK_SIZES = [1000, 2000, 4000, 8000, 16000]
N_REPS = 3


def _fixed_tokens(llm, n: int) -> list[int]:
    corpus_path = DATA_DIR / "corpus_seed0.jsonl"
    clauses = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    full_text = "\n\n".join(f"Clause {c['id']}. {c['text']}" for c in clauses)
    # Corpus alone (~18k tokens) may not cover 16k after tokenization overhead;
    # tile it if needed to guarantee enough tokens for the largest chunk size.
    tokens = llm.tokenize(full_text.encode("utf-8"), add_bos=True)
    while len(tokens) < n:
        tokens = tokens + llm.tokenize(full_text.encode("utf-8"), add_bos=False)
    return tokens[:n]


def worker_make_state(chunk_size: int, state_path: Path) -> None:
    llm = model_mod.load_model(n_ctx=chunk_size + 512, logits_all=False)
    tokens = _fixed_tokens(llm, chunk_size)
    llm.reset()
    llm.eval(tokens)
    state = llm.save_state()
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def worker_time_reprefill(chunk_size: int, out_path: Path) -> None:
    llm = model_mod.load_model(n_ctx=chunk_size + 512, logits_all=False)
    tokens = _fixed_tokens(llm, chunk_size)
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(tokens)
    elapsed = time.perf_counter() - t0
    out_path.write_text(json.dumps({"elapsed_s": elapsed}))


def worker_time_reload(chunk_size: int, state_path: Path, out_path: Path) -> None:
    llm = model_mod.load_model(n_ctx=chunk_size + 512, logits_all=False)
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    t0 = time.perf_counter()
    llm.load_state(state)
    elapsed = time.perf_counter() - t0
    out_path.write_text(json.dumps({"elapsed_s": elapsed}))


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _orchestrate() -> None:
    import subprocess
    import sys

    results_dir = io_mod.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    table = {}

    for chunk in CHUNK_SIZES:
        print(f"=== chunk {chunk} ===")
        state_path = results_dir / f"e6b_state_{chunk}.pkl"
        print("  building state file...")
        subprocess.run(
            [sys.executable, "-m", "src.e6_kvcache.speed_bench",
             "--worker", "make_state", "--chunk-size", str(chunk),
             "--state-path", str(state_path)],
            check=True,
        )

        reprefill_times = []
        for rep in range(N_REPS):
            out_path = results_dir / f"e6b_reprefill_{chunk}_{rep}.json"
            subprocess.run(
                [sys.executable, "-m", "src.e6_kvcache.speed_bench",
                 "--worker", "time_reprefill", "--chunk-size", str(chunk),
                 "--out-path", str(out_path)],
                check=True,
            )
            reprefill_times.append(json.loads(out_path.read_text())["elapsed_s"])
            print(f"  reprefill rep {rep}: {reprefill_times[-1]:.3f}s")

        reload_times = []
        for rep in range(N_REPS):
            out_path = results_dir / f"e6b_reload_{chunk}_{rep}.json"
            subprocess.run(
                [sys.executable, "-m", "src.e6_kvcache.speed_bench",
                 "--worker", "time_reload", "--chunk-size", str(chunk),
                 "--state-path", str(state_path), "--out-path", str(out_path)],
                check=True,
            )
            reload_times.append(json.loads(out_path.read_text())["elapsed_s"])
            print(f"  reload rep {rep}: {reload_times[-1]:.3f}s")

        reprefill_median = _median(reprefill_times)
        reload_median = _median(reload_times)
        table[chunk] = {
            "reprefill_times_s": reprefill_times,
            "reprefill_median_s": reprefill_median,
            "reload_times_s": reload_times,
            "reload_median_s": reload_median,
            "speedup_x": reprefill_median / reload_median if reload_median > 0 else None,
        }
        print(f"  reprefill median: {reprefill_median:.3f}s, reload median: {reload_median:.3f}s, "
              f"speedup: {table[chunk]['speedup_x']:.1f}x")

    p_n_holds = table[4000]["speedup_x"] >= 20
    speedups = [table[c]["speedup_x"] for c in CHUNK_SIZES]
    growing = all(speedups[i] <= speedups[i + 1] for i in range(len(speedups) - 1))

    summary = {
        "env": io_mod.build_env_fingerprint(),
        "chunk_sizes": CHUNK_SIZES,
        "n_reps": N_REPS,
        "table": table,
        "P_N_holds_at_4k": p_n_holds,
        "speedup_monotonically_growing": growing,
    }
    out_path = io_mod.RESULTS_DIR / "e6b_speed_bench.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nP_N holds at 4k (>=20x): {p_n_holds}")
    print(f"speedup growing with size: {growing}")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["make_state", "time_reprefill", "time_reload"], default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()

    if args.worker == "make_state":
        worker_make_state(args.chunk_size, args.state_path)
    elif args.worker == "time_reprefill":
        worker_time_reprefill(args.chunk_size, args.out_path)
    elif args.worker == "time_reload":
        worker_time_reload(args.chunk_size, args.state_path, args.out_path)
    else:
        _orchestrate()


if __name__ == "__main__":
    main()
