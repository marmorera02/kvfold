"""E3 -- verify.py: the propose-then-verify folding transaction (spec E3,
step 5), plus reconstruction checking.

Reconstruction uses the SAME computation shape (single batched eval) for
both the original encode pass and the later verification replay, per the
E1 lesson that batched and sequential eval give measurably different
logits on this CPU/quantized stack. This validates the storage format
(varint round-trip, rank-to-token lookup under determinism) -- it is not
a test of independent sequential-generation-based unfolding, which is a
different question (Iteration 3).
"""
import math

import numpy as np

from src.common.scoring import score_tokens

BUDGET_BITS_PER_TOKEN = 1.0
RANK_MODEL_CAP = 256  # ranks >= this collapse into one overflow bucket


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


def _varint_decode(data: bytes) -> list[int]:
    values = []
    v = 0
    shift = 0
    for byte in data:
        v |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            values.append(v)
            v = 0
            shift = 0
    return values


def build_rank_model(all_ranks: list[int], vocab_size: int) -> np.ndarray:
    """Laplace-smoothed empirical distribution over ranks 0..RANK_MODEL_CAP-1,
    plus one overflow bucket for rank >= RANK_MODEL_CAP, fit from the WHOLE
    document's ranks. This is the fix for two failed approaches: zstd's fixed
    per-frame overhead (~9-13 bytes) swamps any small proposal (a 5-token
    proposal has theta=5 bits, less than one frame header, guaranteeing
    rejection regardless of redundancy -- confirmed empirically, boilerplate
    folded 0%); raw varint is byte-aligned, so even the cheapest possible
    rank (0) costs a full 8 bits, still 8x over a 1 bit/token budget,
    independent of actual predictability. An entropy estimate against a
    shared, whole-document rank-frequency model has neither problem: no
    per-call container overhead, no byte-alignment floor, and small spans
    fairly inherit the corpus's real rank distribution (dominated by rank 0,
    per E1's 67% rank-0 finding) rather than being penalized for their own
    short length."""
    counts = np.ones(RANK_MODEL_CAP + 1)  # add-1 smoothing
    for r in all_ranks:
        bucket = r if r < RANK_MODEL_CAP else RANK_MODEL_CAP
        counts[bucket] += 1
    probs = counts / counts.sum()
    return probs


def _rank_bits(rank: int, rank_probs: np.ndarray, vocab_size: int) -> float:
    bucket = rank if rank < RANK_MODEL_CAP else RANK_MODEL_CAP
    bits = -math.log2(rank_probs[bucket])
    if bucket == RANK_MODEL_CAP:
        bits += math.log2(max(vocab_size - RANK_MODEL_CAP, 1))
    return bits


def run_transaction(llm, atoms: list[dict], proposals: list[dict],
                     rank_probs: np.ndarray, vocab_size: int):
    """Processes proposals in cost-ascending order. Returns (committed, promoted, folded).
    The fold/reject decision uses the entropy estimate against `rank_probs`
    (see build_rank_model); the exact ranks are still stored (varint-encoded)
    for lossless reconstruction, decoupling "should we fold" from "what do
    we replay later"."""
    committed = []
    promoted = []
    folded = [False] * len(atoms)

    for prop in proposals:
        a_start, a_end = prop["atom_start"], prop["atom_end"]
        context_text = "".join(atoms[i]["text"] for i in range(a_start) if not folded[i])
        span_text = "".join(atoms[i]["text"] for i in range(a_start, a_end))

        if not span_text.strip():
            committed.append({
                "atom_start": a_start, "atom_end": a_end, "span_text": span_text,
                "residual_bytes": b"", "n_tokens": 0, "context_text": context_text,
                "residual_bits": 0, "stored_bits": 0, "theta": 0.0,
            })
            for i in range(a_start, a_end):
                folded[i] = True
            continue

        scores = score_tokens(llm, text=span_text, context=context_text)
        ranks = [s.rank_true for s in scores]
        residual_bytes = _varint_encode(ranks)
        stored_bits = len(residual_bytes) * 8
        entropy_bits = sum(_rank_bits(r, rank_probs, vocab_size) for r in ranks)
        theta = BUDGET_BITS_PER_TOKEN * len(ranks)

        if entropy_bits <= theta:
            committed.append({
                "atom_start": a_start, "atom_end": a_end, "span_text": span_text,
                "residual_bytes": residual_bytes, "n_tokens": len(ranks),
                "context_text": context_text, "residual_bits": entropy_bits,
                "stored_bits": stored_bits, "theta": theta,
            })
            for i in range(a_start, a_end):
                folded[i] = True
        else:
            promoted.append({
                "atom_start": a_start, "atom_end": a_end,
                "residual_bits": entropy_bits, "theta": theta,
            })

    return committed, promoted, folded


def reconstruct_and_check(llm, commit: dict) -> tuple[list[int], list[int], bool]:
    """Decode the stored residual (raw varint), replay it against a
    freshly-scored (same shape: one batched eval) distribution of the
    identical context+span, and recover token ids. Returns
    (reconstructed, true, match)."""
    if commit["n_tokens"] == 0:
        return [], [], True  # trivial whitespace-only span, nothing to reconstruct
    ranks = _varint_decode(commit["residual_bytes"])

    ctx_tokens = llm.tokenize(commit["context_text"].encode("utf-8"), add_bos=True)
    text_tokens = llm.tokenize(commit["span_text"].encode("utf-8"), add_bos=False)
    full = ctx_tokens + text_tokens
    score_from = len(ctx_tokens)

    llm.reset()
    llm.eval(full)
    all_scores = np.asarray(llm.scores)[:len(full)]

    reconstructed = []
    for j, r in enumerate(ranks):
        pos = score_from + j
        logits_row = all_scores[pos - 1]
        order = np.argsort(-logits_row)
        reconstructed.append(int(order[r]))

    return reconstructed, text_tokens, reconstructed == text_tokens
