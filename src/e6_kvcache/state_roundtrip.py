"""Cycle 3, Iteration 1 -- E6.a: KV state save/restore round-trip (P-M).

Process A prefills a fixed context, saves state (pickled LlamaState) to
disk, then greedily continues generation in that same unbroken session
("unbroken continuation"). Process B (fresh process) loads the saved
state and greedily continues from the restored point ("restored
continuation"). Both continuations use the identical computation shape
(sequential single-token eval), per the E1 lesson -- this is a valid
apples-to-apples comparison, unlike comparing a batched pass to a
sequential one.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CONTEXT_TOKENS = 2000
GENERATE_TOKENS = 100


def _fixed_context_tokens(llm, n: int) -> list[int]:
    corpus_path = DATA_DIR / "corpus_seed0.jsonl"
    clauses = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    full_text = "\n\n".join(f"Clause {c['id']}. {c['text']}" for c in clauses)
    tokens = llm.tokenize(full_text.encode("utf-8"), add_bos=True)
    return tokens[:n]


def _greedy_continue(llm, total_so_far: int, n_new: int) -> list[int]:
    tokens = []
    total = total_so_far
    for _ in range(n_new):
        logits = np.asarray(llm.scores[total - 1])
        next_id = int(np.argmax(logits))
        tokens.append(next_id)
        llm.eval([next_id])
        total += 1
    return tokens


def worker_prefill_and_save(state_path: Path, continuation_path: Path) -> None:
    llm = model_mod.load_model(n_ctx=4096, logits_all=True)
    tokens = _fixed_context_tokens(llm, CONTEXT_TOKENS)
    llm.reset()
    llm.eval(tokens)

    state = llm.save_state()
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    continuation = _greedy_continue(llm, len(tokens), GENERATE_TOKENS)
    with open(continuation_path, "w") as f:
        json.dump({"n_context_tokens": len(tokens), "continuation_tokens": continuation}, f)


def worker_load_and_continue(state_path: Path, continuation_path: Path) -> None:
    llm = model_mod.load_model(n_ctx=4096, logits_all=True)
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    llm.load_state(state)

    continuation = _greedy_continue(llm, state.n_tokens, GENERATE_TOKENS)
    with open(continuation_path, "w") as f:
        json.dump({"n_context_tokens": state.n_tokens, "continuation_tokens": continuation}, f)


def _orchestrate() -> None:
    import subprocess
    import sys

    results_dir = io_mod.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    state_path = results_dir / "e6a_state.pkl"
    unbroken_path = results_dir / "e6a_unbroken.json"
    restored_path = results_dir / "e6a_restored.json"

    print("E6.a: process A (prefill + save + unbroken continuation)...")
    subprocess.run(
        [sys.executable, "-m", "src.e6_kvcache.state_roundtrip",
         "--worker", "prefill_and_save", "--state-path", str(state_path),
         "--continuation-path", str(unbroken_path)],
        check=True,
    )
    print("E6.a: process B (fresh process, load state + restored continuation)...")
    subprocess.run(
        [sys.executable, "-m", "src.e6_kvcache.state_roundtrip",
         "--worker", "load_and_continue", "--state-path", str(state_path),
         "--continuation-path", str(restored_path)],
        check=True,
    )

    unbroken = json.loads(unbroken_path.read_text())
    restored = json.loads(restored_path.read_text())

    n_ctx_match = unbroken["n_context_tokens"] == restored["n_context_tokens"]
    tokens_match = unbroken["continuation_tokens"] == restored["continuation_tokens"]
    p_m_holds = n_ctx_match and tokens_match

    first_mismatch = None
    if not tokens_match:
        for i, (a, b) in enumerate(zip(unbroken["continuation_tokens"], restored["continuation_tokens"])):
            if a != b:
                first_mismatch = i
                break

    summary = {
        "env": io_mod.build_env_fingerprint(),
        "context_tokens": CONTEXT_TOKENS,
        "generate_tokens": GENERATE_TOKENS,
        "n_context_tokens_match": n_ctx_match,
        "continuation_tokens_match": tokens_match,
        "P_M_holds": p_m_holds,
        "first_mismatch_index": first_mismatch,
        "unbroken_continuation_tokens": unbroken["continuation_tokens"],
        "restored_continuation_tokens": restored["continuation_tokens"],
    }
    out_path = io_mod.RESULTS_DIR / "e6a_state_roundtrip.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"P_M_holds: {p_m_holds}")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["prefill_and_save", "load_and_continue"], default=None)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--continuation-path", type=Path, default=None)
    args = parser.parse_args()

    if args.worker == "prefill_and_save":
        worker_prefill_and_save(args.state_path, args.continuation_path)
    elif args.worker == "load_and_continue":
        worker_load_and_continue(args.state_path, args.continuation_path)
    else:
        _orchestrate()


if __name__ == "__main__":
    main()
