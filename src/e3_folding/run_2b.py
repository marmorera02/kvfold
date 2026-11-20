"""E3 Iteration 2b: fold-policy recalibration (granularity + budget).
Keeps the verified-transaction mechanism from Iteration 2 (P-T validated
it) but replaces the all-or-nothing per-span budget with recursive
bisection + a 3-bin salience policy (silent/flagged/live), with thresholds
derived from the corpus's own measured cost distribution rather than
invented.
"""
import argparse
import random

import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod
from src.common.scoring import score_tokens, unit_token_spans
from src.e3_folding.atomize import atomize
from src.e3_folding.run import _load_boilerplate_text, _load_prose_text
from src.e3_folding.verify import _rank_bits, build_rank_model
from src.e3_folding.verify_2b import reconstruct_and_check_2b, run_transaction_2b
from src.e3_folding.watershed import build_proposals


def _document_data(llm, text: str):
    doc, atoms = atomize(text)
    atom_texts = [a["text"] for a in atoms]
    reconstructed_text, atom_token_spans = unit_token_spans(llm, atom_texts, joiner="")
    if reconstructed_text != text:
        raise AssertionError("atom partition does not reconstruct the original text exactly")

    scores = score_tokens(llm, text=reconstructed_text, context="")
    surprisal = [-s.logprob_true for s in scores]
    whole_doc_ranks = [s.rank_true for s in scores]
    rank_probs = build_rank_model(whole_doc_ranks, llm.n_vocab())
    per_token_costs = [_rank_bits(r, rank_probs, llm.n_vocab()) for r in whole_doc_ranks]

    atom_surprisals, atom_mean_costs = [], []
    for start, end in atom_token_spans:
        local_s = surprisal[start:end]
        local_c = per_token_costs[start:end]
        atom_surprisals.append(float(np.mean(local_s)) if local_s else 0.0)
        atom_mean_costs.append(float(np.mean(local_c)) if local_c else 0.0)

    return {
        "atoms": atoms, "total_tokens": len(surprisal),
        "rank_probs": rank_probs, "per_token_costs": per_token_costs,
        "atom_surprisals": atom_surprisals, "atom_mean_costs": atom_mean_costs,
    }


def _derive_thresholds(per_token_costs: list[float]) -> tuple[float, float]:
    arr = np.array(per_token_costs)
    theta_fold = float(np.median(arr))
    theta_flag = float(arr.mean() + arr.std())
    return theta_fold, theta_flag


def _zscore(values: list[float]) -> list[float]:
    arr = np.array(values)
    std = arr.std()
    if std == 0:
        return [0.0] * len(values)
    return ((arr - arr.mean()) / std).tolist()


def _analytic_prediction(atoms, atom_surprisals, atom_mean_costs, theta_fold, theta_flag) -> float:
    """P-V (Change 3): predicted fold fraction using ONLY already-computed
    per-atom data from the single whole-document scoring pass -- no new
    model calls. Simulates the same anchor test (gated OR z>=0.5 surprisal,
    unchanged from Iteration 2's watershed) plus the new 3-bin classification."""
    z = _zscore(atom_surprisals)
    n_fold_tokens = 0
    total_tokens = 0
    for i, atom in enumerate(atoms):
        n_tok = atom["token_end"] - atom["token_start"]
        total_tokens += n_tok
        is_anchor = atom["gated"] or z[i] >= 0.5
        if is_anchor:
            continue
        if atom_mean_costs[i] <= theta_flag:  # silent or flagged both count as folded
            n_fold_tokens += n_tok
    return n_fold_tokens / total_tokens if total_tokens else 0.0


def run_for_text(llm, label: str, text: str, seed_for_sampling: int = 0) -> dict:
    data = _document_data(llm, text)
    atoms = data["atoms"]
    theta_fold, theta_flag = _derive_thresholds(data["per_token_costs"])
    print(f"  [{label}] derived thresholds: theta_fold={theta_fold:.3f} bits/token, "
          f"theta_flag={theta_flag:.3f} bits/token")

    p_v_prediction = _analytic_prediction(
        atoms, data["atom_surprisals"], data["atom_mean_costs"], theta_fold, theta_flag,
    )
    print(f"  [{label}] P-V analytic prediction: {p_v_prediction:.4f}")

    proposals, is_anchor, z = build_proposals(atoms, data["atom_surprisals"])
    committed, folded = run_transaction_2b(
        llm, atoms, proposals, data["rank_probs"], llm.n_vocab(), theta_fold, theta_flag,
    )

    gated_violations = []
    for c in committed:
        for i in c["atom_indices"]:
            if atoms[i]["gated"]:
                gated_violations.append(i)
    if gated_violations:
        raise AssertionError(f"[{label}] gated atom(s) inside committed fold: {gated_violations}")

    n_folded_tokens = sum(c["n_tokens"] for c in committed)
    total_tokens = data["total_tokens"]
    fold_fraction = n_folded_tokens / total_tokens if total_tokens else 0.0
    p_v_holds = abs(fold_fraction - p_v_prediction) <= 0.10

    n_silent_tokens = sum(c["n_tokens"] for c in committed if c["bin"] == "silent")
    n_flagged_tokens = sum(c["n_tokens"] for c in committed if c["bin"] == "flagged")
    flag_density = n_flagged_tokens / n_folded_tokens if n_folded_tokens else 0.0

    silent_units = [c for c in committed if c["bin"] == "silent" and c["n_tokens"] > 0]
    flagged_units = [c for c in committed if c["bin"] == "flagged" and c["n_tokens"] > 0]
    rng = random.Random(seed_for_sampling)
    n_flag_sample = min(3, len(flagged_units))
    n_silent_sample = min(10 - n_flag_sample, len(silent_units))
    sample = (rng.sample(flagged_units, n_flag_sample) if flagged_units else []) + \
             (rng.sample(silent_units, n_silent_sample) if silent_units else [])

    recon_results = []
    for c in sample:
        _, _, match = reconstruct_and_check_2b(llm, c)
        recon_results.append(match)
    n_recon_ok = sum(recon_results)

    residual_bits_silent = sum(c["residual_bits"] for c in committed if c["bin"] == "silent")
    residual_bits_flagged = sum(c["residual_bits"] for c in committed if c["bin"] == "flagged")

    return {
        "label": label,
        "theta_fold": theta_fold,
        "theta_flag": theta_flag,
        "P_V_prediction": p_v_prediction,
        "fold_fraction": fold_fraction,
        "P_V_holds": p_v_holds,
        "total_atoms": len(atoms),
        "total_tokens": total_tokens,
        "n_committed_units": len(committed),
        "n_folded_tokens": n_folded_tokens,
        "n_silent_tokens": n_silent_tokens,
        "n_flagged_tokens": n_flagged_tokens,
        "flag_density": flag_density,
        "residual_bits_per_silent_token": (residual_bits_silent / n_silent_tokens) if n_silent_tokens else None,
        "residual_bits_per_flagged_token": (residual_bits_flagged / n_flagged_tokens) if n_flagged_tokens else None,
        "gated_atoms_total": sum(1 for a in atoms if a["gated"]),
        "gated_violations": gated_violations,
        "n_reconstruction_sampled": len(sample),
        "n_reconstruction_flagged_sampled": n_flag_sample,
        "n_reconstruction_ok": int(n_recon_ok),
        "reconstruction_all_exact": (n_recon_ok == len(sample)) if sample else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    llm = model_mod.load_model(logits_all=True)

    print("Iteration 2b: boilerplate...")
    boilerplate_result = run_for_text(llm, "boilerplate", _load_boilerplate_text())
    print("Iteration 2b: prose...")
    prose_result = run_for_text(llm, "prose", _load_prose_text())

    p_v_overall_holds = boilerplate_result["P_V_holds"] and prose_result["P_V_holds"]
    total_sampled = boilerplate_result["n_reconstruction_sampled"] + prose_result["n_reconstruction_sampled"]
    total_ok = boilerplate_result["n_reconstruction_ok"] + prose_result["n_reconstruction_ok"]
    reconstruction_holds = (total_ok == total_sampled) if total_sampled else None
    no_gate_violations = (len(boilerplate_result["gated_violations"]) == 0
                           and len(prose_result["gated_violations"]) == 0)

    summary = {
        "boilerplate": boilerplate_result,
        "prose": prose_result,
        "P_V_holds": p_v_overall_holds,
        "reconstruction_holds": reconstruction_holds,
        "no_gate_violations": no_gate_violations,
        "gate_holds": bool(p_v_overall_holds and reconstruction_holds and no_gate_violations),
    }
    out_path = io_mod.save_result("cycle3_iteration2b_run.json", summary)
    print(f"\nboilerplate: fold_fraction={boilerplate_result['fold_fraction']:.4f} "
          f"(predicted {boilerplate_result['P_V_prediction']:.4f}), "
          f"flag_density={boilerplate_result['flag_density']:.3f}")
    print(f"prose: fold_fraction={prose_result['fold_fraction']:.4f} "
          f"(predicted {prose_result['P_V_prediction']:.4f}), "
          f"flag_density={prose_result['flag_density']:.3f}")
    print(f"P_V_holds: {p_v_overall_holds}")
    print(f"reconstruction_holds: {reconstruction_holds}")
    print(f"gate_holds: {summary['gate_holds']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
