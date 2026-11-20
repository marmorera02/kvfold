"""E2.5 -- Classifier and pass criteria (spec E2.5).

Fixed-threshold rules chosen on seed 0, evaluated frozen on seeds 1 and 2.
Contradiction is checked first since its rule doesn't reference z_L0.
"""
import argparse
import json
from pathlib import Path

from src.common import io as io_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

Z_THRESHOLD = 1.0
NLI_THRESHOLD = 0.5

TYPE_TO_LABEL = {
    "n1_weird_benign": "weird_benign",
    "n2_contradiction": "contradiction",
    "n3_novel": "novel",
    "n4_control": "boilerplate",
}


def classify(z_l0: float, z_l2: float, nli: float) -> str:
    if z_l2 >= Z_THRESHOLD and nli >= NLI_THRESHOLD:
        return "contradiction"
    if z_l0 >= Z_THRESHOLD and z_l2 >= Z_THRESHOLD and nli < NLI_THRESHOLD:
        return "novel"
    if z_l0 >= Z_THRESHOLD and z_l2 < Z_THRESHOLD and nli < NLI_THRESHOLD:
        return "weird_benign"
    if z_l0 < Z_THRESHOLD and z_l2 < Z_THRESHOLD:
        return "boilerplate"
    return "unclassified"


def run(seed: int) -> dict:
    ground_truth = json.loads((DATA_DIR / f"ground_truth_seed{seed}.json").read_text())
    levels_raw = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())
    scores = json.loads((io_mod.RESULTS_DIR / f"e2_scores_seed{seed}.json").read_text())
    nli = json.loads((io_mod.RESULTS_DIR / f"e2_nli_seed{seed}.json").read_text())

    l0_by_id = {u["unit_id"]: u for u in scores["levels"]["l0"]}
    l1_by_id = {u["unit_id"]: u for u in scores["levels"]["l1"]}
    l2_scores_by_id = {u["unit_id"]: u for u in scores["levels"]["l2"]}
    nli_by_l1id = {u["unit_id"]: u for u in nli["nli"]}

    l2_id_by_l0id = {}
    for l2_unit in levels_raw["l2"]:
        for l0id in l2_unit["source_l0_ids"]:
            l2_id_by_l0id[l0id] = l2_unit["id"]

    profiles = {}
    for needle in ground_truth["needles"]:
        cid = needle["id"]
        ntype = needle["type"]
        z_l0 = l0_by_id[cid]["z_mean"]
        z_l1 = l1_by_id[cid]["z_mean"]
        z_l2 = l2_scores_by_id[l2_id_by_l0id[cid]]["z_mean"]
        nli_score = nli_by_l1id[cid]["max_contradiction"]

        predicted = classify(z_l0, z_l2, nli_score)
        profiles[ntype] = {
            "clause_id": cid,
            "z_l0": z_l0, "z_l1": z_l1, "z_l2": z_l2,
            "nli_max_contradiction": nli_score,
            "predicted": predicted,
            "expected": TYPE_TO_LABEL[ntype],
        }

    correct = {ntype: (p["predicted"] == p["expected"]) for ntype, p in profiles.items()}
    return {
        "seed": seed,
        "profiles": profiles,
        "correct": correct,
        "n_correct": sum(correct.values()),
        "n_total": len(correct),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run(args.seed)
    out_path = io_mod.save_result(f"e2_classify_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")
    for ntype, p in payload["profiles"].items():
        status = "OK" if payload["correct"][ntype] else "WRONG"
        print(f"  {ntype}: predicted={p['predicted']} expected={p['expected']} [{status}] "
              f"z_l0={p['z_l0']:.2f} z_l1={p['z_l1']:.2f} z_l2={p['z_l2']:.2f} "
              f"nli={p['nli_max_contradiction']:.2f}")
    print(f"{payload['n_correct']}/{payload['n_total']} correct")


if __name__ == "__main__":
    main()
