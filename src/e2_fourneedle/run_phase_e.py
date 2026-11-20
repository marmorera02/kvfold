"""Phase E orchestrator: builds levels, scores, deltas, assumptions, and
NLI for wikiA/wikiB, then evaluates P-J (zero-contradiction precision) and
P-K (recall) per results/phaseE_predictions.md -- the same delta-routed-NLI
methodology as Task 2 (bottom-15th-percentile L2 section by z_delta,
assumption-set NLI >= 0.9 within the routed section counts as a
contradiction verdict), applied to real Wikipedia-register prose instead
of the templated legal corpus.
"""
import json

import numpy as np

from src.common import io as io_mod
from src.e2_fourneedle import assumptions_wiki, build_levels_wiki, delta_score, nli_v2_wiki, score_levels

DATA_DIR = build_levels_wiki.DATA_DIR
SECTION_SIZE = build_levels_wiki.SECTION_SIZE
ROUTING_PERCENTILE = 15
NLI_THRESHOLD = 0.9

VARIANTS = ["wikiA", "wikiB"]


def _build_pipeline(variant: str) -> None:
    print(f"=== Phase E pipeline: {variant} ===")
    levels = build_levels_wiki.build_levels(variant)
    (DATA_DIR / f"levels_seed{variant}.json").write_text(json.dumps(levels, indent=2))

    scores = score_levels.run(variant)
    io_mod.save_result(f"e2_scores_seed{variant}.json", scores)

    deltas = delta_score.run(variant)
    io_mod.save_result(f"e2_delta_seed{variant}.json", deltas)

    assumptions = assumptions_wiki.run(variant)
    io_mod.save_result(f"e2_assumptions_seed{variant}.json", assumptions)

    nli = nli_v2_wiki.run(variant)
    io_mod.save_result(f"e2_nli_v2_seed{variant}.json", nli)


def _routed_units(variant: str) -> tuple[set[int], dict]:
    """Returns (set of L0 clause ids belonging to a bottom-15th-percentile
    L2 section, diagnostic info)."""
    deltas = json.loads((io_mod.RESULTS_DIR / f"e2_delta_seed{variant}.json").read_text())
    l2_deltas = deltas["levels"]["l2"]
    z_values = [s["z_delta"] for s in l2_deltas]
    cutoff = float(np.percentile(z_values, ROUTING_PERCENTILE))
    candidate_section_ids = {s["unit_id"] for s in l2_deltas if s["z_delta"] <= cutoff}

    levels = json.loads((DATA_DIR / f"levels_seed{variant}.json").read_text())
    l2_units = levels["l2"]
    routed_clause_ids = set()
    for sec in l2_units:
        if sec["id"] in candidate_section_ids:
            routed_clause_ids.update(sec["source_l0_ids"])

    return routed_clause_ids, {
        "z_delta_L2_by_section": {s["unit_id"]: s["z_delta"] for s in l2_deltas},
        "cutoff": cutoff,
        "candidate_section_ids": sorted(candidate_section_ids),
        "routed_clause_ids": sorted(routed_clause_ids),
    }


def _evaluate(variant: str) -> dict:
    routed_ids, routing_info = _routed_units(variant)
    nli = json.loads((io_mod.RESULTS_DIR / f"e2_nli_v2_seed{variant}.json").read_text())
    ground_truth = json.loads((DATA_DIR / f"ground_truth_seed{variant}.json").read_text())

    nli_by_id = {r["unit_id"]: r["max_contradiction"] for r in nli["nli"]}
    contradiction_ids = {n["id"] for n in ground_truth["needles"] if "contradiction" in n["type"]}

    verdicts = []
    for uid in sorted(nli_by_id):
        routed = uid in routed_ids
        nli_score = nli_by_id[uid]
        flagged = routed and nli_score >= NLI_THRESHOLD
        verdicts.append({"clause_id": uid, "routed": routed, "nli_score": nli_score, "flagged": flagged})

    flagged_ids = {v["clause_id"] for v in verdicts if v["flagged"]}
    false_positives = sorted(flagged_ids - contradiction_ids)
    true_positives = sorted(flagged_ids & contradiction_ids)
    missed = sorted(contradiction_ids - flagged_ids)

    return {
        "variant": variant,
        "routing_info": routing_info,
        "verdicts": verdicts,
        "contradiction_ids_ground_truth": sorted(contradiction_ids),
        "flagged_ids": sorted(flagged_ids),
        "false_positives": false_positives,
        "true_positives": true_positives,
        "missed_contradictions": missed,
    }


def main() -> None:
    for variant in VARIANTS:
        _build_pipeline(variant)

    eval_wikiA = _evaluate("wikiA")
    eval_wikiB = _evaluate("wikiB")

    p_j_holds = len(eval_wikiB["false_positives"]) == 0
    p_k_holds = (85 in eval_wikiA["true_positives"])

    if p_j_holds and p_k_holds:
        gate_verdict = "P-J and P-K both HOLD: locator/router framing transfers to real prose."
    elif p_k_holds and not p_j_holds:
        gate_verdict = "P-K holds, P-J FAILS: recall transfers, precision does not."
    else:
        gate_verdict = "P-K FAILS: the routing/extraction mechanism itself does not transfer to non-templated prose."

    summary = {
        "P_J_holds": p_j_holds,
        "P_K_holds": p_k_holds,
        "gate_verdict": gate_verdict,
        "wikiA": eval_wikiA,
        "wikiB": eval_wikiB,
    }
    out_path = io_mod.save_result("phaseE_run.json", summary)
    print(f"\nwikiA false positives: {eval_wikiA['false_positives']}")
    print(f"wikiA true positives: {eval_wikiA['true_positives']}  missed: {eval_wikiA['missed_contradictions']}")
    print(f"wikiB false positives: {eval_wikiB['false_positives']}")
    print(f"P_J_holds: {p_j_holds}  P_K_holds: {p_k_holds}")
    print(gate_verdict)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
