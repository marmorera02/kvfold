"""E3 -- run.py: orchestrator for the trimmed folding experiment
(Cycle 3 Iteration 2). Two text types (corpus boilerplate, prose), one
fold level, per `cycle3_iteration2_predictions.md`.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod
from src.common.scoring import score_tokens, unit_token_spans
from src.e3_folding.atomize import atomize
from src.e3_folding.verify import build_rank_model, reconstruct_and_check, run_transaction
from src.e3_folding.watershed import build_proposals

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
WORD_BUDGET = 1200  # ~1500 tokens


def _load_boilerplate_text() -> str:
    corpus_path = DATA_DIR / "corpus_seed0.jsonl"
    clauses = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    parts = []
    word_count = 0
    for c in clauses:
        parts.append(f"Clause {c['id']}. {c['text']}")
        word_count += len(c["text"].split())
        if word_count >= WORD_BUDGET:
            break
    return "\n\n".join(parts)


def _load_prose_text() -> str:
    return (DATA_DIR / "e0_fixed_paragraph.txt").read_text(encoding="utf-8")


def run_for_text(llm, label: str, text: str, seed_for_sampling: int = 0) -> dict:
    doc, atoms = atomize(text)
    atom_texts = [a["text"] for a in atoms]
    reconstructed_text, atom_token_spans = unit_token_spans(llm, atom_texts, joiner="")
    if reconstructed_text != text:
        raise AssertionError(f"[{label}] atom partition does not reconstruct the original text exactly")

    scores = score_tokens(llm, text=reconstructed_text, context="")
    surprisal = [-s.logprob_true for s in scores]
    atom_surprisals = []
    for start, end in atom_token_spans:
        local = surprisal[start:end]
        atom_surprisals.append(float(np.mean(local)) if local else 0.0)

    whole_doc_ranks = [s.rank_true for s in scores]
    rank_probs = build_rank_model(whole_doc_ranks, llm.n_vocab())

    proposals, is_anchor, z = build_proposals(atoms, atom_surprisals)
    committed, promoted, folded = run_transaction(llm, atoms, proposals, rank_probs, llm.n_vocab())

    gated_violations = []
    for c in committed:
        for i in range(c["atom_start"], c["atom_end"]):
            if atoms[i]["gated"]:
                gated_violations.append(i)
    if gated_violations:
        raise AssertionError(f"[{label}] gated atom(s) inside committed fold: {gated_violations}")

    n_folded_tokens = sum(c["n_tokens"] for c in committed)
    total_tokens = len(surprisal)
    fold_fraction = n_folded_tokens / total_tokens if total_tokens else 0.0

    rng = random.Random(seed_for_sampling)
    n_sample = min(10, len(committed))
    sample = rng.sample(committed, n_sample) if committed else []
    recon_results = []
    for c in sample:
        _, _, match = reconstruct_and_check(llm, c)
        recon_results.append(match)
    n_recon_ok = sum(recon_results)

    residual_bits_total = sum(c["residual_bits"] for c in committed)
    n_decisions = len(committed) + len(promoted)
    rejection_rate = len(promoted) / n_decisions if n_decisions else 0.0

    return {
        "label": label,
        "total_atoms": len(atoms),
        "total_tokens": total_tokens,
        "n_committed": len(committed),
        "n_promoted": len(promoted),
        "n_folded_tokens": n_folded_tokens,
        "fold_fraction": fold_fraction,
        "rejection_rate": rejection_rate,
        "residual_bits_total": residual_bits_total,
        "residual_bits_per_folded_token": (residual_bits_total / n_folded_tokens) if n_folded_tokens else None,
        "gated_atoms_total": sum(1 for a in atoms if a["gated"]),
        "gated_violations": gated_violations,
        "n_reconstruction_sampled": n_sample,
        "n_reconstruction_ok": int(n_recon_ok),
        "reconstruction_all_exact": (n_recon_ok == n_sample) if n_sample else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    llm = model_mod.load_model(logits_all=True)

    boilerplate_text = _load_boilerplate_text()
    prose_text = _load_prose_text()

    print("scoring/folding boilerplate...")
    boilerplate_result = run_for_text(llm, "boilerplate", boilerplate_text)
    print(f"  fold_fraction={boilerplate_result['fold_fraction']:.3f} "
          f"recon={boilerplate_result['n_reconstruction_ok']}/{boilerplate_result['n_reconstruction_sampled']}")

    print("scoring/folding prose...")
    prose_result = run_for_text(llm, "prose", prose_text)
    print(f"  fold_fraction={prose_result['fold_fraction']:.3f} "
          f"recon={prose_result['n_reconstruction_ok']}/{prose_result['n_reconstruction_sampled']}")

    e1_rank0_fraction = 0.674  # from results/e1_seed0.json
    p_s_boilerplate = abs(boilerplate_result["fold_fraction"] - e1_rank0_fraction) <= 0.15
    p_s_ordering = boilerplate_result["fold_fraction"] > prose_result["fold_fraction"]
    p_s_holds = p_s_boilerplate and p_s_ordering

    total_sampled = boilerplate_result["n_reconstruction_sampled"] + prose_result["n_reconstruction_sampled"]
    total_ok = boilerplate_result["n_reconstruction_ok"] + prose_result["n_reconstruction_ok"]
    p_t_reconstruction = (total_ok == total_sampled) if total_sampled else None
    p_t_no_gate_violations = (len(boilerplate_result["gated_violations"]) == 0
                               and len(prose_result["gated_violations"]) == 0)
    p_t_holds = bool(p_t_reconstruction) and p_t_no_gate_violations

    summary = {
        "boilerplate": boilerplate_result,
        "prose": prose_result,
        "P_S_boilerplate_near_e1_rank0": p_s_boilerplate,
        "P_S_ordering_holds": p_s_ordering,
        "P_S_holds": p_s_holds,
        "P_T_reconstruction_10_of_10": p_t_reconstruction,
        "P_T_no_gate_violations": p_t_no_gate_violations,
        "P_T_holds": p_t_holds,
    }
    out_path = io_mod.save_result("cycle3_iteration2_run.json", summary)
    print(f"\nP_S_holds: {p_s_holds}")
    print(f"P_T_holds: {p_t_holds}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
