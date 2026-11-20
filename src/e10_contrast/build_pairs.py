"""E10 -- assumption-contrast pair extraction (E10 DIRECTIVE).

Builds the (unit u, assumption A) pair set from EXISTING artifacts only --
no corpus generation, no re-scoring of old results. A pair is every
guard-passed (hypothesis unit, premise assumption) combination recorded in
e2_nli_v2_seed{seed}.json, joined against e2_assumptions_seed{seed}.json
(premise text) and levels_seed{seed}.json (hypothesis L1/L0 text).

Labels (per the directive):
- TRUE_CONTRADICTION: n2/n2a x liability-cap assumptions (seeds 0,1,2,5);
  n2b x exclusivity assumption (seed 5, clause 78 x 12); wikiA clause 85 x
  height assumptions (8, 40, 75).
- KNOWN_FP: the exact Task-2 false-positive pairs (seed 3: 74x66, 83x32;
  seed 4: 85x60), plus every guard-passed pair in wikiB (ground truth has
  zero contradictions there, so every guard-passed pair is a legitimate
  "should not fire" case, per the directive's own framing).
- NEUTRAL: everything else.

Special case, logged explicitly: seed 5's n2b x clause-12 pair does not
exist in e2_nli_v2_seed5.json's guard-passed set, because the LEGACY
assumptions.py classifier never extracted clause 12 at all (confirmed:
its record classifies clause 12 as ROUTINE, assumption=None -- this is
exactly Task 2's diagnosed miss). E7's structured extractor did catch
clause 12, but its ledger fields (SCOPE/ATTRIBUTE/VALUE) are not a natural-
language sentence, so using them directly as "A" would not be comparable
in kind to every other pair's assumption text (all single-sentence
restatements/summaries of a source clause). Instead this pair uses clause
12's own L1 summary text as A -- preserving "a natural-sentence
restatement of the source clause" as the common form across every pair in
this experiment, while still being sourced from the correct clause. This
substitution is recorded in the pair's `assumption_source` field.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

SEEDS = [0, 1, 2, 3, 4, 5, "wikiA", "wikiB"]

TRUE_CONTRADICTION_SPECS = {
    (0, 87): [8, 25, 50],
    (1, 87): [8, 25, 50],
    (2, 87): [8, 25, 50],
    (5, 87): [8, 25, 50],   # n2a
    (5, 78): [12],          # n2b -- special-cased, see module docstring
    ("wikiA", 85): [8, 40, 75],
}
KNOWN_FP_SPECS = {
    (3, 74): [66],
    (3, 83): [32],
    (4, 85): [60],
}


def _load(seed, name_fmt):
    return json.loads(RESULTS_DIR.joinpath(name_fmt.format(seed=seed)).read_text())


def _label(seed, hyp_id, premise_id):
    if premise_id in TRUE_CONTRADICTION_SPECS.get((seed, hyp_id), []):
        return "TRUE_CONTRADICTION"
    if premise_id in KNOWN_FP_SPECS.get((seed, hyp_id), []):
        return "KNOWN_FP"
    if seed == "wikiB":
        return "KNOWN_FP"
    return "NEUTRAL"


def build_pairs_for_seed(seed) -> list[dict]:
    nli = _load(seed, "e2_nli_v2_seed{seed}.json")
    assumptions = _load(seed, "e2_assumptions_seed{seed}.json")
    levels = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())

    assumption_by_id = {r["unit_id"]: r["assumption"] for r in assumptions["records"] if r["assumption"] is not None}
    l1_by_id = {u["id"]: u["text"] for u in levels["l1"]}
    l0_by_id = {c["id"]: c["text"] for c in levels["l0"]}

    pairs = []
    for rec in nli["nli"]:
        hyp_id = rec["unit_id"]
        for premise_id in rec.get("premise_ids_guard_passed", []):
            if premise_id not in assumption_by_id:
                continue  # shouldn't happen (guard is built from running_assumptions), defensive only
            label = _label(seed, hyp_id, premise_id)
            pairs.append({
                "seed": seed,
                "hyp_id": hyp_id,
                "premise_id": premise_id,
                "u_l1_text": l1_by_id[hyp_id],
                "u_l0_text": l0_by_id.get(hyp_id),
                "A_text": assumption_by_id[premise_id],
                "label": label,
                "assumption_source": "legacy_assumptions",
            })

    # special case: seed 5's n2b x clause-12 pair, absent from the legacy
    # guard-passed set because clause 12 was never extracted by
    # assumptions.py (confirmed ROUTINE/None in e2_assumptions_seed5.json).
    if seed == 5:
        pairs.append({
            "seed": 5,
            "hyp_id": 78,
            "premise_id": 12,
            "u_l1_text": l1_by_id[78],
            "u_l0_text": l0_by_id.get(78),
            "A_text": l1_by_id[12],
            "label": "TRUE_CONTRADICTION",
            "assumption_source": "clause_12_own_L1_summary (legacy pipeline never extracted this clause; "
                                  "see module docstring)",
        })

    return pairs


def build_all_pairs() -> list[dict]:
    all_pairs = []
    for seed in SEEDS:
        all_pairs.extend(build_pairs_for_seed(seed))
    return all_pairs


def main() -> None:
    pairs = build_all_pairs()
    by_label = {}
    for p in pairs:
        by_label.setdefault(p["label"], 0)
        by_label[p["label"]] += 1
    print(f"total pairs: {len(pairs)}")
    for label, n in by_label.items():
        print(f"  {label}: {n}")
    out_path = RESULTS_DIR / "e10_pairs.json"
    out_path.write_text(json.dumps(pairs, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
