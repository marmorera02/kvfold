"""E7 orchestrator: extract ledgers for seeds 3/4/5, evaluate P-Q (recall)
and P-R (deterministic conflict detection), per results/e7_predictions.md.
"""
import json

from src.common import io as io_mod
from src.e7_ledger import extract
from src.e7_ledger.conflicts import find_conflicts

DATA_DIR = extract.DATA_DIR
SEEDS = [3, 4, 5]


def _should_extract_ids(ground_truth: dict) -> list[int]:
    ids = list(ground_truth["liability_frame_clause_ids"]) + list(ground_truth["extra_frame_clause_ids"])
    ids += [n["id"] for n in ground_truth["needles"] if "contradiction" in n["type"]]
    return sorted(set(ids))


def main() -> None:
    per_seed = {}
    all_should_extract = []
    all_extracted_hits = []

    for seed in SEEDS:
        print(f"=== E7 extraction: seed {seed} ===")
        ledger_payload = extract.run(seed)
        io_mod.save_result(f"e7_ledger_seed{seed}.json", ledger_payload)

        ground_truth = json.loads((DATA_DIR / f"ground_truth_seed{seed}.json").read_text())
        should_extract = _should_extract_ids(ground_truth)
        extracted_ids = {r["clause_id"] for r in ledger_payload["records"] if r["ledger_entry"] is not None}
        hits = [cid for cid in should_extract if cid in extracted_ids]
        misses = [cid for cid in should_extract if cid not in extracted_ids]

        entries = [
            {"clause_id": r["clause_id"], **r["ledger_entry"]}
            for r in ledger_payload["records"] if r["ledger_entry"] is not None
        ]
        conflicts = find_conflicts(entries)

        per_seed[seed] = {
            "should_extract_ids": should_extract,
            "extracted_ids": sorted(extracted_ids),
            "hits": hits,
            "misses": misses,
            "recall": len(hits) / len(should_extract) if should_extract else None,
            "n_ledger_entries": len(entries),
            "conflicts": conflicts,
            "n_conflicts": len(conflicts),
        }
        all_should_extract.extend((seed, cid) for cid in should_extract)
        all_extracted_hits.extend((seed, cid) for cid in hits)
        print(f"  seed {seed}: recall {len(hits)}/{len(should_extract)}, "
              f"misses={misses}, conflicts found={len(conflicts)}")

    overall_recall = len(all_extracted_hits) / len(all_should_extract) if all_should_extract else None
    clause_12_extracted = 12 in per_seed[5]["extracted_ids"]

    p_q_holds = bool(overall_recall is not None and overall_recall >= 0.9 and clause_12_extracted)

    # P-R checks
    def _has_conflict_pair(seed: int, id_a: int, id_b_set: set[int]) -> bool:
        for c in per_seed[seed]["conflicts"]:
            pair = {c["clause_id_a"], c["clause_id_b"]}
            if id_a in pair and (pair & id_b_set):
                return True
        return False

    seed3_zero_fp = per_seed[3]["n_conflicts"] == 0
    seed4_zero_fp = per_seed[4]["n_conflicts"] == 0
    seed5_catches_n2a = _has_conflict_pair(5, 87, {8, 25, 50})
    seed5_catches_n2b = _has_conflict_pair(5, 78, {12})

    p_r_holds = bool(seed3_zero_fp and seed4_zero_fp and seed5_catches_n2a and seed5_catches_n2b)

    if not p_q_holds:
        gate_verdict = "P-Q FAILS: extraction track closes at this model scale (P-R evaluated below for completeness, but cannot rescue this verdict)."
    elif p_q_holds and p_r_holds:
        gate_verdict = "P-Q and P-R both HOLD: structured extraction viable at this model scale; detection-v3 (full pyramid integration) noted as out-of-scope follow-on."
    else:
        gate_verdict = "P-Q holds, P-R FAILS: extraction generalizes but the deterministic conflict rule does not; see per-seed conflict detail."

    summary = {
        "seeds": SEEDS,
        "overall_recall": overall_recall,
        "clause_12_extracted": clause_12_extracted,
        "P_Q_holds": p_q_holds,
        "seed3_zero_false_positives": seed3_zero_fp,
        "seed4_zero_false_positives": seed4_zero_fp,
        "seed5_catches_n2a": seed5_catches_n2a,
        "seed5_catches_n2b": seed5_catches_n2b,
        "P_R_holds": p_r_holds,
        "gate_verdict": gate_verdict,
        "per_seed": per_seed,
    }
    out_path = io_mod.save_result("e7_run.json", summary)
    print(f"\noverall recall: {overall_recall}")
    print(f"clause 12 extracted: {clause_12_extracted}")
    print(f"P_Q_holds: {p_q_holds}  P_R_holds: {p_r_holds}")
    print(gate_verdict)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
