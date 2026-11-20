"""E3 Iteration 2b-fast -- linear single-pass fold-detection using
incremental KV-cache extension, replacing verify_2b.run_transaction_2b's
per-candidate full context re-prefill.

Root cause fixed (found while diagnosing why fold-detection took upwards
of an hour on a ~1500-token document): run_transaction_2b processes
proposals in watershed.py's COST-ASCENDING order (not document order),
and for each one rebuilds `context_text` as a fresh string and calls
score_tokens(), which does llm.reset() + a full prefill of that whole
context from scratch -- every single time, for ~250 candidates on this
document. Since candidates are visited in jumbled (non-document) order,
this is effectively O(candidates x document_length), not O(document_length).
Nothing about surprise-gated folding *requires* this; it's an engineering
inefficiency in how context got rebuilt, not a property of the idea.

This module makes ONE linear, left-to-right pass over the document,
extending a single persistent KV cache incrementally instead of
re-prefilling, with one save_state()/load_state() checkpoint per valley
run (not per atom) for rollback when part of a run gets folded.

TWO bugs found and fixed while getting this correct:

1. Reconstruction always tokenizes an accumulated context as ONE WHOLE
   STRING via a single llm.tokenize() call. A first attempt built the
   cache by tokenizing each new segment separately and appending --
   reconstruction then disagreed with what was actually cached, because
   GPT-style BPE pretokenizes on "leading-space + word" chunks: a chunk's
   tokenization is only stable regardless of preceding context if the
   split boundary falls exactly at such a chunk edge. Ending a piece with
   TRAILING whitespace and starting the next piece fresh (this project's
   atoms each carry their own trailing separator) is NOT such a boundary
   -- confirmed empirically: tokenizing "Clause 1. " alone ends in a bare
   space token, but tokenizing "Clause 1. All ..." as one string merges
   that space into the following word instead, so the two tokenizations
   diverge at the very last token of every single piece. A second attempt
   tried to detect this via a whole-string re-tokenize-and-compare check
   with a full reset+re-eval fallback -- but the divergence above means
   that fallback fires on nearly every step (confirmed: 200/226 in a
   400-atom diagnostic), silently reintroducing the exact O(candidates x
   position) blowup this rewrite exists to eliminate.

2. The actual fix: never let the "committed" (tokenized+eval'd) text end
   in whitespace. Any trailing whitespace is held back as `pending_ws` and
   prepended to whatever text comes next before that next piece is
   tokenized -- so every tokenize() call's boundary falls exactly at a
   "leading-space + word" chunk edge, which IS stable regardless of
   preceding context (verified directly: this makes the incrementally
   built token sequence match a hypothetical whole-string tokenization
   exactly, with no reset ever required for this reason).

DISCLOSED BEHAVIOR CHANGE: this processes valley runs in DOCUMENT order,
not run_transaction_2b's cost-ascending order. Hard correctness guarantees
(exact reconstruction, zero gated-atom leakage) are enforced identically
and independently per commit regardless of order. What CAN differ is the
FOLD FRACTION / which specific atoms get folded, because a run's live
context now always reflects strictly-earlier document positions'
decisions rather than whatever happened to be decided earlier in cost
order (which could be from anywhere in the document). This is measured
and reported honestly (compare_2b_fast.py), not assumed identical.
"""
import numpy as np

from src.common.scoring import _logprobs_and_entropy_bits
from src.e3_folding.verify import _rank_bits, _varint_encode


def _score_new_positions(llm, cache_len_before: int, new_tokens: list[int]) -> list[dict]:
    """After llm.eval(new_tokens) has extended the cache from
    cache_len_before, returns per-position rank_true for each of
    new_tokens (logits at absolute position i-1 predict token i, matching
    scoring.py's TokenScore convention)."""
    scores = np.asarray(llm.scores)
    out = []
    for local_i, true_id in enumerate(new_tokens):
        abs_pos = cache_len_before + local_i
        logits_row = scores[abs_pos - 1]
        log_probs_bits, _entropy_bits = _logprobs_and_entropy_bits(logits_row)
        rank_true = int(np.sum(log_probs_bits > log_probs_bits[true_id]))
        out.append({"rank_true": rank_true})
    return out


class _IncrementalCache:
    """Maintains a persistent KV cache that always corresponds exactly to
    tokenize(self.committed_text, add_bos=True) -- see module docstring
    for why trailing whitespace is held back in self.pending_ws rather
    than tokenized immediately."""

    def __init__(self, llm):
        self.llm = llm
        self.tokens: list[int] = []
        self.committed_text = ""
        self.pending_ws = ""

    def add(self, new_text: str) -> None:
        combined = self.pending_ws + new_text
        stripped_len = len(combined.rstrip())
        self.pending_ws = combined[stripped_len:]
        to_commit = combined[:stripped_len]
        if to_commit:
            tokens = self.llm.tokenize(to_commit.encode("utf-8"), add_bos=(len(self.tokens) == 0))
            self.llm.eval(tokens)
            self.tokens = self.tokens + tokens
            self.committed_text += to_commit

    def snapshot(self):
        return (self.llm.save_state(), list(self.tokens), self.committed_text, self.pending_ws)

    def restore(self, snapshot) -> None:
        state, tokens, committed_text, pending_ws = snapshot
        self.llm.load_state(state)
        self.tokens = list(tokens)
        self.committed_text = committed_text
        self.pending_ws = pending_ws


def run_transaction_2b_fast(llm, atoms: list[dict], is_anchor: list[bool],
                             rank_probs: np.ndarray, vocab_size: int,
                             theta_fold: float, theta_flag: float):
    """Single linear document-order pass. Returns (committed, folded),
    same shape/fields as verify_2b.run_transaction_2b's output (so
    reconstruct_and_check_2b works unchanged on either)."""
    committed = []
    folded = [False] * len(atoms)
    n = len(atoms)

    llm.reset()
    cache = _IncrementalCache(llm)

    i = 0
    while i < n:
        if is_anchor[i]:
            cache.add(atoms[i]["text"])
            i += 1
            continue

        start = i
        while i < n and not is_anchor[i]:
            i += 1
        end = i  # exclusive

        atom_indices = list(range(start, end))
        atom_texts = [atoms[k]["text"] for k in atom_indices]
        run_text = "".join(atom_texts)
        run_context_text = cache.committed_text  # exactly what's cached right now

        if not run_text.strip():
            committed.append({
                "atom_indices": atom_indices, "span_text": run_text,
                "residual_bytes": b"", "n_tokens": 0,
                "run_context_text": run_context_text, "run_span_text": run_text,
                "local_token_start": 0, "local_token_end": 0,
                "bin": "silent", "residual_bits": 0,
                "per_atom_bin": ["silent"] * len(atom_indices),
                "run_atom_start": start,
            })
            for k in atom_indices:
                folded[k] = True
            continue

        checkpoint = cache.snapshot()

        # Score the run: fold in any pending whitespace so this tokenize()
        # call's boundary is a stable "leading-space + word" chunk edge.
        run_span_text = checkpoint[3] + run_text  # pending_ws (at checkpoint time) + run_text
        cache_len_before_run = len(cache.tokens)
        run_tokens = llm.tokenize(run_span_text.encode("utf-8"), add_bos=(cache_len_before_run == 0))
        llm.eval(run_tokens)
        token_scores = _score_new_positions(llm, cache_len_before_run, run_tokens)

        # Per-atom token boundaries within run_span_text (cumulative-prefix
        # tokenization, same method as unit_token_spans). The leading
        # pending_ws is NOT given its own separately-computed token count
        # (tokenizing it in isolation and using that as an offset was the
        # bug: a leading space commonly merges into a single token with
        # the word that follows it, so "tokens for pending_ws alone" is
        # not the same count as "tokens consumed by pending_ws once
        # followed by real text" -- confirmed as the cause of an
        # off-by-one reconstruction failure at every group that starts a
        # run). Instead, the cumulative prefix simply STARTS at
        # pending_ws with prev=0, so any merged boundary token is
        # attributed to the first atom's own span, exactly matching how
        # run_tokens was actually tokenized.
        local_spans = []
        prefix = checkpoint[3]  # pending_ws, folded into the first atom's span
        prev = 0
        for t in atom_texts:
            prefix += t
            cnt = len(llm.tokenize(prefix.encode("utf-8"), add_bos=False))
            local_spans.append((prev, cnt))
            prev = cnt

        per_atom_cost = []
        for lo, hi in local_spans:
            local_ranks = [token_scores[j]["rank_true"] for j in range(lo, hi)]
            c = float(np.mean([_rank_bits(r, rank_probs, vocab_size) for r in local_ranks])) if local_ranks else 0.0
            per_atom_cost.append(c)

        per_atom_bin = []
        for local_i, c in enumerate(per_atom_cost):
            atom_idx = atom_indices[local_i]
            if atoms[atom_idx]["gated"]:
                per_atom_bin.append("live")
            elif c <= theta_fold:
                per_atom_bin.append("silent")
            elif c <= theta_flag:
                per_atom_bin.append("flagged")
            else:
                per_atom_bin.append("live")

        # Restore to before this run, then commit exactly the live
        # remainder (cache.add handles the pending-whitespace bookkeeping
        # correctly regardless of outcome).
        live_local = [j for j, b in enumerate(per_atom_bin) if b == "live"]
        cache.restore(checkpoint)
        if live_local:
            live_run_text = "".join(atom_texts[j] for j in live_local)
            cache.add(live_run_text)

        j = 0
        m = len(atom_indices)
        while j < m:
            if per_atom_bin[j] == "live":
                j += 1
                continue
            k = j
            while k < m and per_atom_bin[k] != "live":
                k += 1
            lo, hi = local_spans[j][0], local_spans[k - 1][1]
            group_ranks = [token_scores[p]["rank_true"] for p in range(lo, hi)]
            committed.append({
                "atom_indices": atom_indices[j:k],
                "span_text": "".join(atom_texts[j:k]),
                "residual_bytes": _varint_encode(group_ranks),
                "n_tokens": len(group_ranks),
                "run_context_text": run_context_text,
                "run_span_text": run_span_text,
                "local_token_start": lo,
                "local_token_end": hi,
                "bin": "flagged" if "flagged" in per_atom_bin[j:k] else "silent",
                "per_atom_bin": per_atom_bin[j:k],
                "residual_bits": sum(_rank_bits(r, rank_probs, vocab_size) for r in group_ranks),
                "run_atom_start": start,
            })
            for atom_idx in atom_indices[j:k]:
                folded[atom_idx] = True
            j = k

    return committed, folded


def reconstruct_and_check_2b_fast(llm, atoms: list[dict], is_anchor: list[bool],
                                   rank_probs: np.ndarray, vocab_size: int,
                                   theta_fold: float, theta_flag: float,
                                   target_commit: dict) -> tuple[list[int], list[int], bool]:
    """Verifies one commit produced by run_transaction_2b_fast, replaying
    the SAME incremental construction from the start up through the
    target commit's run, rather than re-evaluating context_text+span_text
    in isolation (verify_2b.reconstruct_and_check_2b's approach).

    This distinction matters and is not cosmetic: run_transaction_2b_fast
    scores each run's tokens on top of a long-running, incrementally
    extended KV cache, while an isolated fresh eval() of just that run's
    own context+text is a DIFFERENT computation shape. Per the E1 lesson
    (established at the very start of this program: batched vs sequential
    eval give measurably different logits on this stack), two
    computations of the same nominal thing are only comparable if they
    share the same shape -- confirmed here empirically: verifying fast-
    variant commits via the isolated-eval method produced occasional
    off-by-one-token mismatches at exactly the position immediately after
    a context/text boundary, which stopped occurring once verification
    was changed to replay the actual incremental history instead.
    """
    if target_commit["n_tokens"] == 0:
        return [], [], True

    from src.e3_folding.verify import _varint_decode
    target_ranks = _varint_decode(target_commit["residual_bytes"])
    target_run_start = target_commit["run_atom_start"]

    llm.reset()
    cache = _IncrementalCache(llm)
    n = len(atoms)
    i = 0
    while i < n:
        if is_anchor[i]:
            cache.add(atoms[i]["text"])
            i += 1
            continue

        start = i
        while i < n and not is_anchor[i]:
            i += 1
        end = i

        atom_indices = list(range(start, end))
        atom_texts = [atoms[k]["text"] for k in atom_indices]
        run_text = "".join(atom_texts)

        if not run_text.strip():
            if start == target_run_start:
                return [], [], target_commit["n_tokens"] == 0
            continue

        checkpoint = cache.snapshot()
        pending_ws_at_checkpoint = checkpoint[3]
        run_span_text = pending_ws_at_checkpoint + run_text
        cache_len_before_run = len(cache.tokens)
        run_tokens = llm.tokenize(run_span_text.encode("utf-8"), add_bos=(cache_len_before_run == 0))
        llm.eval(run_tokens)
        token_scores = _score_new_positions(llm, cache_len_before_run, run_tokens)

        if start == target_run_start:
            lo, hi = target_commit["local_token_start"], target_commit["local_token_end"]
            all_scores = np.asarray(llm.scores)
            reconstructed = []
            for j, r in enumerate(target_ranks):
                pos = cache_len_before_run + lo + j
                logits_row = all_scores[pos - 1]
                order = np.argsort(-logits_row)
                reconstructed.append(int(order[r]))
            true_tokens = run_tokens[lo:hi]
            return reconstructed, true_tokens, reconstructed == true_tokens

        local_spans = []
        prefix = pending_ws_at_checkpoint
        prev = 0
        for t in atom_texts:
            prefix += t
            cnt = len(llm.tokenize(prefix.encode("utf-8"), add_bos=False))
            local_spans.append((prev, cnt))
            prev = cnt

        per_atom_cost = []
        for lo, hi in local_spans:
            local_ranks = [token_scores[j]["rank_true"] for j in range(lo, hi)]
            c = float(np.mean([_rank_bits(r, rank_probs, vocab_size) for r in local_ranks])) if local_ranks else 0.0
            per_atom_cost.append(c)

        per_atom_bin = []
        for local_i, c in enumerate(per_atom_cost):
            atom_idx = atom_indices[local_i]
            if atoms[atom_idx]["gated"]:
                per_atom_bin.append("live")
            elif c <= theta_fold:
                per_atom_bin.append("silent")
            elif c <= theta_flag:
                per_atom_bin.append("flagged")
            else:
                per_atom_bin.append("live")

        live_local = [j for j, b in enumerate(per_atom_bin) if b == "live"]
        cache.restore(checkpoint)
        if live_local:
            live_run_text = "".join(atom_texts[j] for j in live_local)
            cache.add(live_run_text)

    raise AssertionError(f"target run at atom_start={target_run_start} was never reached during replay")
