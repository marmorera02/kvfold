"""Validates and times verify_2b_fast.run_transaction_2b_fast against the
already-known Iteration 2b baseline (results/cycle3_iteration2b_run.json),
on the same boilerplate document. Checks the hard correctness guarantees
(gated-atom containment, exact reconstruction) and reports the fold
fraction alongside the original's, honestly, since processing order
differs (see verify_2b_fast.py's docstring).
"""
import json
import random
import time

from src.common import io as io_mod
from src.common import model as model_mod
from src.e3_folding.run import _load_boilerplate_text
from src.e3_folding.run_2b import _derive_thresholds, _document_data
from src.e3_folding.verify_2b_fast import reconstruct_and_check_2b_fast, run_transaction_2b_fast
from src.e3_folding.watershed import build_proposals


def main() -> None:
    llm = model_mod.load_model(logits_all=True)
    text = _load_boilerplate_text()

    t0 = time.perf_counter()
    data = _document_data(llm, text)
    t_document_data = time.perf_counter() - t0
    atoms = data["atoms"]
    theta_fold, theta_flag = _derive_thresholds(data["per_token_costs"])
    print(f"document_data (atomize+whole-doc score): {t_document_data:.1f}s")
    print(f"theta_fold={theta_fold:.3f} theta_flag={theta_flag:.3f}")

    proposals, is_anchor, z = build_proposals(atoms, data["atom_surprisals"])
    print(f"{len(proposals)} proposals, {sum(is_anchor)} anchor atoms of {len(atoms)}")

    t0 = time.perf_counter()
    committed, folded = run_transaction_2b_fast(
        llm, atoms, is_anchor, data["rank_probs"], llm.n_vocab(), theta_fold, theta_flag,
    )
    t_transaction = time.perf_counter() - t0
    print(f"run_transaction_2b_fast: {t_transaction:.1f}s")

    gated_violations = [i for c in committed for i in c["atom_indices"] if atoms[i]["gated"]]

    n_folded_tokens = sum(c["n_tokens"] for c in committed)
    total_tokens = data["total_tokens"]
    fold_fraction = n_folded_tokens / total_tokens if total_tokens else 0.0

    silent_units = [c for c in committed if c["bin"] == "silent" and c["n_tokens"] > 0]
    flagged_units = [c for c in committed if c["bin"] == "flagged" and c["n_tokens"] > 0]
    rng = random.Random(0)
    n_flag_sample = min(3, len(flagged_units))
    n_silent_sample = min(10 - n_flag_sample, len(silent_units))
    sample = (rng.sample(flagged_units, n_flag_sample) if flagged_units else []) + \
             (rng.sample(silent_units, n_silent_sample) if silent_units else [])

    recon_results = []
    for c in sample:
        _, _, match = reconstruct_and_check_2b_fast(
            llm, atoms, is_anchor, data["rank_probs"], llm.n_vocab(), theta_fold, theta_flag, c,
        )
        recon_results.append(match)
    n_recon_ok = sum(recon_results)

    baseline = json.loads((io_mod.RESULTS_DIR / "cycle3_iteration2b_run.json").read_text())
    baseline_fold_fraction = baseline["boilerplate"]["fold_fraction"]

    summary = {
        "t_document_data_s": t_document_data,
        "t_transaction_fast_s": t_transaction,
        "theta_fold": theta_fold, "theta_flag": theta_flag,
        "n_proposals": len(proposals),
        "n_atoms": len(atoms),
        "n_anchor_atoms": int(sum(is_anchor)),
        "fold_fraction_fast": fold_fraction,
        "fold_fraction_original_cost_order": baseline_fold_fraction,
        "fold_fraction_delta": fold_fraction - baseline_fold_fraction,
        "gated_atoms_total": sum(1 for a in atoms if a["gated"]),
        "gated_violations": gated_violations,
        "n_reconstruction_sampled": len(sample),
        "n_reconstruction_ok": int(n_recon_ok),
        "reconstruction_all_exact": (n_recon_ok == len(sample)) if sample else None,
    }
    out_path = io_mod.save_result("verify_2b_fast_comparison.json", summary)

    print(f"\nfold_fraction (fast, document-order): {fold_fraction:.4f}")
    print(f"fold_fraction (original, cost-order):  {baseline_fold_fraction:.4f}")
    print(f"delta: {fold_fraction - baseline_fold_fraction:+.4f}")
    print(f"gated_violations: {len(gated_violations)}")
    print(f"reconstruction: {n_recon_ok}/{len(sample)} exact")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
