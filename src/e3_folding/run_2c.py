"""Rider I2c (owner decision, post-2b): instrument the boilerplate fold pass
to quantify the "price of context destruction" -- how much a token's
verified cost inflates when scored against live (post-fold) context versus
the initial full-document context, as a function of how much of the
document is already folded. Produces a corrected analytic predictor
(full-context simulation + measured inflation) and reports its error
against the actual 38.3% fold fraction. No pass bar -- this is calibration,
not a gate, per the owner's directive.
"""
import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod
from src.e3_folding.run import _load_boilerplate_text
from src.e3_folding.run_2b import _analytic_prediction, _derive_thresholds, _document_data
from src.e3_folding.verify_2b import run_transaction_2b_instrumented
from src.e3_folding.watershed import build_proposals


def main() -> None:
    llm = model_mod.load_model(logits_all=True)

    text = _load_boilerplate_text()
    data = _document_data(llm, text)
    atoms = data["atoms"]
    theta_fold, theta_flag = _derive_thresholds(data["per_token_costs"])

    proposals, is_anchor, z = build_proposals(atoms, data["atom_surprisals"])
    committed, folded, diagnostics = run_transaction_2b_instrumented(
        llm, atoms, proposals, data["rank_probs"], llm.n_vocab(),
        theta_fold, theta_flag, data["atom_mean_costs"],
    )

    n_folded_tokens = sum(c["n_tokens"] for c in committed)
    actual_fold_fraction = n_folded_tokens / data["total_tokens"] if data["total_tokens"] else 0.0

    # (a) cost inflation vs. fraction already folded
    inflations = []
    for d in diagnostics:
        inflation = d["live_context_cost"] - d["full_context_cost"]
        inflations.append({
            "atom_idx": d["atom_idx"],
            "full_context_cost": d["full_context_cost"],
            "live_context_cost": d["live_context_cost"],
            "inflation": inflation,
            "fraction_folded_at_verification_time": d["fraction_folded_at_verification_time"],
        })

    # bucket by fraction_folded decile for a readable curve
    deciles = {}
    for rec in inflations:
        bucket = min(int(rec["fraction_folded_at_verification_time"] * 10), 9)
        deciles.setdefault(bucket, []).append(rec["inflation"])
    curve = {
        f"{b*10}-{b*10+10}%": {
            "n": len(vals), "mean_inflation": float(np.mean(vals)), "median_inflation": float(np.median(vals)),
        }
        for b, vals in sorted(deciles.items())
    }

    overall_mean_inflation = float(np.mean([r["inflation"] for r in inflations])) if inflations else 0.0

    # (b) corrected analytic predictor: full-context simulation + measured mean inflation applied uniformly
    corrected_prediction = _analytic_prediction(
        atoms, data["atom_surprisals"],
        [c + overall_mean_inflation for c in data["atom_mean_costs"]],
        theta_fold, theta_flag,
    )
    original_prediction = _analytic_prediction(
        atoms, data["atom_surprisals"], data["atom_mean_costs"], theta_fold, theta_flag,
    )

    summary = {
        "label": "boilerplate",
        "theta_fold": theta_fold,
        "theta_flag": theta_flag,
        "actual_fold_fraction": actual_fold_fraction,
        "original_P_V_prediction": original_prediction,
        "overall_mean_cost_inflation_bits": overall_mean_inflation,
        "cost_inflation_by_decile_folded": curve,
        "corrected_prediction": corrected_prediction,
        "corrected_prediction_error": abs(corrected_prediction - actual_fold_fraction),
        "original_prediction_error": abs(original_prediction - actual_fold_fraction),
        "n_atoms_verified": len(diagnostics),
    }
    out_path = io_mod.save_result("cycle3_iteration2c_run.json", summary)

    print(f"actual fold fraction: {actual_fold_fraction:.4f}")
    print(f"original P-V prediction: {original_prediction:.4f} (error {summary['original_prediction_error']:.4f})")
    print(f"overall mean cost inflation: {overall_mean_inflation:.4f} bits/token")
    print(f"corrected prediction: {corrected_prediction:.4f} (error {summary['corrected_prediction_error']:.4f})")
    print("\ncost inflation by fraction-already-folded decile:")
    for bucket, row in curve.items():
        print(f"  {bucket}: n={row['n']} mean_inflation={row['mean_inflation']:.3f} "
              f"median_inflation={row['median_inflation']:.3f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
