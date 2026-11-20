"""E3 Iteration 2b -- recursive bisection + 3-bin salience policy
(fold-silent / fold-and-flag / stay-live), replacing Iteration 2's
all-or-nothing per-span budget check. The verified-transaction mechanism
itself is unchanged (P-T already validated it): every committed unit is
still scored via score_tokens against its actual live/run context, and
reconstruction still replays that same context to recover tokens exactly.

Design note: the whole valley run is scored in ONE batched pass (context =
live text at the run's start), giving per-token ranks for every atom in the
run at once -- this is what makes per-atom bisection nearly free (no extra
model calls versus Iteration 2's per-run scoring). Atoms landing in "live"
break an otherwise-committable run into pieces; each committed piece's
reconstruction context is the run-start context PLUS the full original
text of everything earlier in the SAME run (live or folded), matching
exactly what the one-pass encoding conditioned on.
"""
import numpy as np

from src.common.scoring import score_tokens, unit_token_spans
from src.e3_folding.verify import _rank_bits, _varint_encode


def process_valley_run(llm, atoms: list[dict], atom_indices: list[int], context_text: str,
                        rank_probs: np.ndarray, vocab_size: int,
                        theta_fold: float, theta_flag: float):
    """Returns (committed_units, live_atom_indices) for one maximal valley run."""
    atom_texts = [atoms[i]["text"] for i in atom_indices]
    span_text = "".join(atom_texts)

    if not span_text.strip():
        return [{
            "atom_indices": atom_indices, "span_text": span_text,
            "residual_bytes": b"", "n_tokens": 0, "context_text": context_text,
            "bin": "silent", "residual_bits": 0, "per_atom_bin": ["silent"] * len(atom_indices),
        }], [], [0.0] * len(atom_indices)

    scores = score_tokens(llm, text=span_text, context=context_text)
    ranks = [s.rank_true for s in scores]
    _, local_spans = unit_token_spans(llm, atom_texts, joiner="")

    per_atom_cost = []
    for start, end in local_spans:
        local_ranks = ranks[start:end]
        c = float(np.mean([_rank_bits(r, rank_probs, vocab_size) for r in local_ranks])) if local_ranks else 0.0
        per_atom_cost.append(c)

    per_atom_bin = []
    for i, c in enumerate(per_atom_cost):
        if atoms[atom_indices[i]]["gated"]:
            per_atom_bin.append("live")
        elif c <= theta_fold:
            per_atom_bin.append("silent")
        elif c <= theta_flag:
            per_atom_bin.append("flagged")
        else:
            per_atom_bin.append("live")

    committed_units = []
    live_atom_indices = []
    i = 0
    n = len(atom_indices)
    while i < n:
        if per_atom_bin[i] == "live":
            live_atom_indices.append(atom_indices[i])
            i += 1
            continue
        j = i
        while j < n and per_atom_bin[j] != "live":
            j += 1

        group_local_start = local_spans[i][0]
        group_local_end = local_spans[j - 1][1]
        group_ranks = ranks[group_local_start:group_local_end]
        group_span_text = "".join(atom_texts[i:j])

        committed_units.append({
            "atom_indices": atom_indices[i:j],
            "span_text": group_span_text,
            "residual_bytes": _varint_encode(group_ranks),
            "n_tokens": len(group_ranks),
            # Reconstruction must replay the EXACT same context/text split as
            # this encoding pass (run_context_text, run_span_text = the WHOLE
            # run, unchanged) -- re-splitting at the group boundary instead
            # (e.g. run_context_text + atom_texts[:i] as a new "context")
            # tokenizes at a different boundary than the original pass used,
            # and BPE merges are not guaranteed consistent across a different
            # split point even with identical concatenated string content.
            # Confirmed as the root cause of a 0/10 reconstruction failure.
            "run_context_text": context_text,
            "run_span_text": span_text,
            "local_token_start": group_local_start,
            "local_token_end": group_local_end,
            "bin": "flagged" if "flagged" in per_atom_bin[i:j] else "silent",
            "per_atom_bin": per_atom_bin[i:j],
            "residual_bits": sum(_rank_bits(r, rank_probs, vocab_size) for r in group_ranks),
        })
        i = j

    return committed_units, live_atom_indices, per_atom_cost


def reconstruct_and_check_2b(llm, commit: dict) -> tuple[list[int], list[int], bool]:
    """2b variant of reconstruct_and_check: replays the ranks against the
    WHOLE run's context+text (matching the original encoding pass exactly),
    then extracts just this group's token slice."""
    if commit["n_tokens"] == 0:
        return [], [], True

    from src.e3_folding.verify import _varint_decode
    ranks = _varint_decode(commit["residual_bytes"])

    ctx_tokens = llm.tokenize(commit["run_context_text"].encode("utf-8"), add_bos=True)
    text_tokens = llm.tokenize(commit["run_span_text"].encode("utf-8"), add_bos=False)
    full = ctx_tokens + text_tokens
    score_from = len(ctx_tokens)

    llm.reset()
    llm.eval(full)
    all_scores = np.asarray(llm.scores)[:len(full)]

    local_start = commit["local_token_start"]
    true_tokens = text_tokens[local_start:commit["local_token_end"]]

    reconstructed = []
    for j, r in enumerate(ranks):
        pos = score_from + local_start + j
        logits_row = all_scores[pos - 1]
        order = np.argsort(-logits_row)
        reconstructed.append(int(order[r]))

    return reconstructed, true_tokens, reconstructed == true_tokens


def run_transaction_2b(llm, atoms: list[dict], proposals: list[dict],
                        rank_probs: np.ndarray, vocab_size: int,
                        theta_fold: float, theta_flag: float):
    """Processes maximal valley runs (from watershed, unchanged) in
    cost-ascending order. Returns (committed, folded) where folded[i] is
    True iff atom i ended up in a committed (silent or flagged) group."""
    committed = []
    folded = [False] * len(atoms)

    for prop in proposals:
        atom_indices = list(range(prop["atom_start"], prop["atom_end"]))
        context_text = "".join(atoms[i]["text"] for i in range(prop["atom_start"]) if not folded[i])

        units, live_indices, _per_atom_cost = process_valley_run(
            llm, atoms, atom_indices, context_text, rank_probs, vocab_size, theta_fold, theta_flag,
        )
        for u in units:
            committed.append(u)
            for i in u["atom_indices"]:
                folded[i] = True
        # live_indices atoms are simply never marked folded (implicit)

    return committed, folded


def run_transaction_2b_instrumented(llm, atoms: list[dict], proposals: list[dict],
                                     rank_probs: np.ndarray, vocab_size: int,
                                     theta_fold: float, theta_flag: float,
                                     atom_full_context_costs: list[float]):
    """Rider I2c: same as run_transaction_2b, but also logs, for every atom
    actually verified, its full-context cost (from the initial whole-document
    pass) alongside its live-context cost (from this real transaction) and
    the fraction of the document already folded at the moment it was
    verified -- instrumentation for quantifying the 'price of context
    destruction' (P-V's diagnosed ~26-point gap)."""
    committed = []
    folded = [False] * len(atoms)
    total_tokens = sum(a["token_end"] - a["token_start"] for a in atoms)
    diagnostics = []

    for prop in proposals:
        atom_indices = list(range(prop["atom_start"], prop["atom_end"]))
        context_text = "".join(atoms[i]["text"] for i in range(prop["atom_start"]) if not folded[i])

        n_folded_tokens_so_far = sum(
            atoms[i]["token_end"] - atoms[i]["token_start"] for i in range(len(atoms)) if folded[i]
        )
        fraction_folded_so_far = n_folded_tokens_so_far / total_tokens if total_tokens else 0.0

        units, live_indices, per_atom_cost = process_valley_run(
            llm, atoms, atom_indices, context_text, rank_probs, vocab_size, theta_fold, theta_flag,
        )
        for local_i, atom_idx in enumerate(atom_indices):
            diagnostics.append({
                "atom_idx": atom_idx,
                "full_context_cost": atom_full_context_costs[atom_idx],
                "live_context_cost": per_atom_cost[local_i],
                "fraction_folded_at_verification_time": fraction_folded_so_far,
            })
        for u in units:
            committed.append(u)
            for i in u["atom_indices"]:
                folded[i] = True

    return committed, folded, diagnostics
