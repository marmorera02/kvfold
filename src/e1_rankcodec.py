"""E1 -- Rank-stream lossless codec (spec section E1).

Claim tested: storage cost per token under LM-coding equals its surprisal;
the bits/token curve spikes exactly at planted surprising content;
reconstruction is bit-exact. Also documents the expected, motivating failure:
NO spike at the fluent-but-contradictory needle (token-level surprisal is
blind to long-range contradiction -- this is what E2's cross-scale profile
is designed to catch).
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import zstandard as zstd

from src.common import io as io_mod
from src.common import model as model_mod
from src.common.scoring import score_tokens

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DECODE_WINDOW = 512
FIRST_N_CHECK = 2048
ROLLING_WINDOW = 25
STABILITY_CHECK_N = 200


def _varint_encode(values: list[int]) -> bytes:
    out = bytearray()
    for v in values:
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def _load_corpus_text(seed: int) -> tuple[str, list[dict]]:
    path = DATA_DIR / f"corpus_seed{seed}.jsonl"
    clauses = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            clauses.append(json.loads(line))
    full_text = "\n\n".join(f"Clause {c['id']}. {c['text']}" for c in clauses)
    return full_text, clauses


def _clause_token_starts(llm, full_text: str, clauses: list[dict]) -> dict[int, int]:
    """Approximate token offset (into text_tokens, 0-indexed) of each clause's start,
    via cumulative tokenization of text prefixes ending at each clause boundary."""
    parts = []
    starts = {}
    cursor = 0
    for c in clauses:
        header = f"Clause {c['id']}. "
        piece = header + c["text"]
        parts.append(piece)
        prefix_text = "\n\n".join(parts[:-1] + [""]) if len(parts) > 1 else ""
        prefix_tokens = llm.tokenize(prefix_text.encode("utf-8"), add_bos=False) if prefix_text else []
        starts[c["id"]] = len(prefix_tokens)
        cursor += len(piece) + 2
    return starts


def _reconstruct_window(llm, full_tokens: list[int], ranks: list[int], text_start_idx: int,
                         window_start: int, window_len: int, left_ctx: int = 2048):
    abs_start = text_start_idx + window_start
    left = max(0, abs_start - left_ctx)
    seed_ctx = full_tokens[left:abs_start]
    llm.reset()
    if seed_ctx:
        llm.eval(seed_ctx)
    total = len(seed_ctx)
    reconstructed = []
    for j in range(window_len):
        logits_row = np.asarray(llm.scores[total - 1])
        order = np.argsort(-logits_row)
        r = ranks[window_start + j]
        next_tok = int(order[r])
        reconstructed.append(next_tok)
        llm.eval([next_tok])
        total += 1
    true_tokens = full_tokens[abs_start:abs_start + window_len]
    return reconstructed, true_tokens


def _logit_stability_check(llm, full_tokens: list[int], n: int) -> dict:
    """Isolate whether batched (prefill) and sequential (single-token decode) eval give
    the same logits for the same true-token context -- independent of any rank-lookup
    logic. Feeds the TRUE tokens back at each step (not rank-reconstructed ones), so any
    divergence found here is purely a property of the CPU/quantized inference kernels,
    not of the codec's rank encoding."""
    llm.reset()
    llm.eval(full_tokens[:n + 1])
    batched = np.asarray(llm.scores[:n + 1]).copy()

    llm.reset()
    llm.eval([full_tokens[0]])
    seq_rows = [np.asarray(llm.scores[0]).copy()]
    for i in range(1, n):
        llm.eval([full_tokens[i]])
        seq_rows.append(np.asarray(llm.scores[i]).copy())
    sequential = np.array(seq_rows)

    mismatches = 0
    for i in range(n):
        true_next = full_tokens[i + 1]
        b_rank = int(np.sum(batched[i] > batched[i][true_next]))
        s_rank = int(np.sum(sequential[i] > sequential[i][true_next]))
        if b_rank != s_rank:
            mismatches += 1

    abs_diff = np.abs(batched[:n] - sequential)
    return {
        "n_positions_checked": n,
        "rank_mismatch_count": mismatches,
        "rank_mismatch_fraction": mismatches / n,
        "mean_abs_logit_diff": float(np.mean(abs_diff)),
        "max_abs_logit_diff": float(np.max(abs_diff)),
    }


def run(seed: int) -> dict:
    full_text, clauses = _load_corpus_text(seed)
    ground_truth = json.loads((DATA_DIR / f"ground_truth_seed{seed}.json").read_text())

    llm = model_mod.load_model(logits_all=True)

    scores = score_tokens(llm, text=full_text, context="")
    ranks = [s.rank_true for s in scores]
    surprisal_bits = [-s.logprob_true for s in scores]
    n_tokens = len(scores)

    clause_starts = _clause_token_starts(llm, full_text, clauses)

    ctx_tokens = llm.tokenize(b"", add_bos=True)
    text_tokens = llm.tokenize(full_text.encode("utf-8"), add_bos=False)
    full_tokens = ctx_tokens + text_tokens
    text_start_idx = len(ctx_tokens)

    logit_stability = _logit_stability_check(llm, full_tokens, STABILITY_CHECK_N)

    rank_bytes = _varint_encode(ranks)
    lm_compressed = zstd.ZstdCompressor(level=19).compress(rank_bytes)
    lm_bits_per_token = len(lm_compressed) * 8 / n_tokens
    lm_bits_per_char = len(lm_compressed) * 8 / len(full_text)

    raw_bytes = full_text.encode("utf-8")
    zstd_compressed = zstd.ZstdCompressor(level=19).compress(raw_bytes)
    gzip_compressed = gzip.compress(raw_bytes, compresslevel=9)
    zstd_bits_per_token = len(zstd_compressed) * 8 / n_tokens
    zstd_bits_per_char = len(zstd_compressed) * 8 / len(raw_bytes)
    gzip_bits_per_token = len(gzip_compressed) * 8 / n_tokens
    gzip_bits_per_char = len(gzip_compressed) * 8 / len(raw_bytes)

    frac_rank0 = sum(1 for r in ranks if r == 0) / n_tokens

    reconstruction_ok = True
    mismatches = []

    recon, true = _reconstruct_window(llm, full_tokens, ranks, text_start_idx, 0,
                                       min(FIRST_N_CHECK, n_tokens))
    if recon != true:
        reconstruction_ok = False
        mismatches.append({"window": "first_2048", "first_bad_index":
                            next(i for i, (a, b) in enumerate(zip(recon, true)) if a != b)})

    needle_windows = {}
    for needle in ground_truth["needles"]:
        clause_id = needle["id"]
        ntype = needle["type"]
        tok_start = clause_starts[clause_id]
        clause = next(c for c in clauses if c["id"] == clause_id)
        clause_len_toks = len(llm.tokenize(clause["text"].encode("utf-8"), add_bos=False))
        w_start = max(0, tok_start - DECODE_WINDOW // 4)
        w_len = min(DECODE_WINDOW, n_tokens - w_start)
        recon, true = _reconstruct_window(llm, full_tokens, ranks, text_start_idx, w_start, w_len)
        ok = recon == true
        if not ok:
            reconstruction_ok = False
            mismatches.append({"window": f"needle_{ntype}", "clause_id": clause_id,
                                "first_bad_index": next(i for i, (a, b) in enumerate(zip(recon, true)) if a != b)})

        local_slice = surprisal_bits[tok_start:tok_start + clause_len_toks]
        needle_windows[ntype] = {
            "clause_id": clause_id,
            "token_start": tok_start,
            "n_tokens": clause_len_toks,
            "local_mean_bits_per_token": float(np.mean(local_slice)) if local_slice else None,
            "local_max_bits_per_token": float(np.max(local_slice)) if local_slice else None,
        }

    doc_median_bits = float(np.median(surprisal_bits))
    for ntype, w in needle_windows.items():
        w["document_median_bits_per_token"] = doc_median_bits
        w["ratio_to_median"] = (w["local_mean_bits_per_token"] / doc_median_bits
                                 if w["local_mean_bits_per_token"] is not None else None)

    rolling = np.convolve(surprisal_bits, np.ones(ROLLING_WINDOW) / ROLLING_WINDOW, mode="valid").tolist()

    payload = {
        "seed": seed,
        "n_tokens": n_tokens,
        "n_chars": len(full_text),
        "bits_per_token": {
            "lm_codec": lm_bits_per_token,
            "zstd19": zstd_bits_per_token,
            "gzip9": gzip_bits_per_token,
        },
        "bits_per_char": {
            "lm_codec": lm_bits_per_char,
            "zstd19": zstd_bits_per_char,
            "gzip9": gzip_bits_per_char,
        },
        "fraction_rank0": frac_rank0,
        "reconstruction_exact": reconstruction_ok,
        "mismatches": mismatches,
        "logit_stability_batched_vs_sequential": logit_stability,
        "document_median_bits_per_token": doc_median_bits,
        "needle_profiles": needle_windows,
        "rolling_bits_per_token": rolling,
        "rolling_window": ROLLING_WINDOW,
        "needle_token_positions": {n["type"]: clause_starts[n["id"]] for n in ground_truth["needles"]},
    }
    return payload


def plot(payload: dict, seed: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    rolling = payload["rolling_bits_per_token"]
    ax.plot(rolling, linewidth=0.8, color="#3b6ea5")
    ax.set_xlabel(f"token position (rolling mean, window={payload['rolling_window']})")
    ax.set_ylabel("bits/token (surprisal)")
    ax.set_title(f"E1 seed {seed}: token-level surprisal across document")

    colors = {"n1_weird_benign": "#d62728", "n2_contradiction": "#9467bd",
              "n3_novel": "#2ca02c", "n4_control": "#7f7f7f"}
    offset = ROLLING_WINDOW // 2
    for ntype, pos in payload["needle_token_positions"].items():
        x = max(0, pos - offset)
        ax.axvline(x, color=colors.get(ntype, "black"), linestyle="--", alpha=0.8, label=ntype)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = run(args.seed)
    out_path = io_mod.save_result(f"e1_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")

    plot_path = io_mod.RESULTS_DIR / f"e1_seed{args.seed}_bits_per_token.png"
    plot(payload, args.seed, plot_path)
    print(f"wrote {plot_path}")

    print(f"\nbits/token -- LM codec: {payload['bits_per_token']['lm_codec']:.3f}  "
          f"zstd19: {payload['bits_per_token']['zstd19']:.3f}  "
          f"gzip9: {payload['bits_per_token']['gzip9']:.3f}")
    print(f"fraction rank-0 tokens: {payload['fraction_rank0']:.3f}")
    print(f"reconstruction exact: {payload['reconstruction_exact']}")
    ls = payload["logit_stability_batched_vs_sequential"]
    print(f"batched-vs-sequential rank mismatch: {ls['rank_mismatch_count']}/{ls['n_positions_checked']} "
          f"positions (mean|diff|={ls['mean_abs_logit_diff']:.4f}, max|diff|={ls['max_abs_logit_diff']:.4f})")
    print(f"document median bits/token: {payload['document_median_bits_per_token']:.3f}")
    for ntype, w in payload["needle_profiles"].items():
        print(f"  {ntype}: local mean bits/token = {w['local_mean_bits_per_token']:.3f} "
              f"(ratio to median: {w['ratio_to_median']:.2f})")


if __name__ == "__main__":
    main()
