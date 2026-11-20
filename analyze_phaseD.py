"""Phase D analysis (amended per user directive): P-A' / P-B / P-C / P-D / P-E / P-F
and the 4-config ablation table. Pure post-hoc analysis of existing results/ JSON
-- no new model calls. Run locally (not in Docker); only needs numpy.
"""
import json
import statistics
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"

SEEDS = [0, 1, 2]
NLI_THRESHOLD = 0.9
DELTA_PERCENTILE = 15


def load(seed, name):
    return json.loads((RESULTS / f"{name}_seed{seed}.json").read_text())


def load_data(seed, name):
    return json.loads((DATA / f"{name}_seed{seed}.json").read_text(encoding="utf-8"))


def l2_id_by_l0id(levels_raw):
    out = {}
    for u in levels_raw["l2"]:
        for l0id in u["source_l0_ids"]:
            out[l0id] = u["id"]
    return out


def clean_sections(levels_raw, ground_truth):
    """L2 sections containing no needle and no liability-frame clause."""
    needle_l0 = {n["id"] for n in ground_truth["needles"]}
    frame_l0 = set(ground_truth["liability_frame_clause_ids"])
    excluded = needle_l0 | frame_l0
    clean = []
    for u in levels_raw["l2"]:
        if not (set(u["source_l0_ids"]) & excluded):
            clean.append(u["id"])
    return clean


def auc_rank(pos_scores, neg_scores):
    """AUC via Mann-Whitney U (rank-sum), for small samples."""
    all_scores = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    all_scores.sort(key=lambda x: x[0])
    ranks = {}
    i = 0
    n = len(all_scores)
    while i < n:
        j = i
        while j < n and all_scores[j][0] == all_scores[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[k] for k in range(n) if all_scores[k][1] == 1)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    u_stat = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u_stat / (n_pos * n_neg)


results = {"per_seed": {}, "pooled": {}}

pooled_n3_delta = []
pooled_clean_delta = []

for seed in SEEDS:
    gt = load_data(seed, "ground_truth")
    levels_raw = load_data(seed, "levels")
    scores_v1 = load(seed, "e2_scores")
    delta_qwen = load(seed, "e2_delta")
    delta_llama = json.loads((RESULTS / f"e2_delta_crossmodel_seed{seed}.json").read_text())
    nli = load(seed, "e2_nli_v2")

    l2map = l2_id_by_l0id(levels_raw)
    needle_ids = {n["type"]: n["id"] for n in gt["needles"]}
    n2_l0 = needle_ids["n2_contradiction"]
    n2_l2 = l2map[n2_l0]

    # --- P-A': N2 rank at L2 (both scorers) ---
    def rank_of(delta_payload, target_l2_id):
        l2_units = delta_payload["levels"]["l2"]
        ordered = sorted(l2_units, key=lambda u: u["z_delta"])  # ascending: rank 0 = lowest
        for i, u in enumerate(ordered):
            if u["unit_id"] == target_l2_id:
                return i + 1, len(ordered)  # 1-indexed rank, total count
        return None, len(ordered)

    rank_qwen, n_l2 = rank_of(delta_qwen, n2_l2)
    rank_llama, _ = rank_of(delta_llama, n2_l2)
    p_a_prime = (rank_qwen == 1) and (rank_llama == 1)

    # --- P-B: z_L0(N2) < z_L0(N1) and < z_L0(N3) ---
    z_l0_by_id = {u["unit_id"]: u["z_mean"] for u in scores_v1["levels"]["l0"]}
    z_l1_by_id = {u["unit_id"]: u["z_mean"] for u in scores_v1["levels"]["l1"]}
    z_l0_n1 = z_l0_by_id[needle_ids["n1_weird_benign"]]
    z_l0_n2 = z_l0_by_id[needle_ids["n2_contradiction"]]
    z_l0_n3 = z_l0_by_id[needle_ids["n3_novel"]]
    p_b = (z_l0_n2 < z_l0_n1) and (z_l0_n2 < z_l0_n3)

    # --- P-C: N1 collapses L0->L1 (z_L1<1 while z_L0>2) ---
    z_l1_n1 = z_l1_by_id[needle_ids["n1_weird_benign"]]
    p_c = (z_l0_n1 > 2) and (z_l1_n1 < 1)

    # --- P-D: NLI flags N2>=0.9, corpus frac>=0.5 below 0.15 ---
    nli_scores = [u["max_contradiction"] for u in nli["nli"]]
    n2_nli = next(u["max_contradiction"] for u in nli["nli"] if u["unit_id"] == n2_l0)
    frac_ge_05 = sum(1 for v in nli_scores if v >= 0.5) / len(nli_scores)
    p_d = (n2_nli >= 0.9) and (frac_ge_05 < 0.15)

    # --- P-E: delta-routing (bottom 15th pctile of L2 z_delta -> candidate sections) ---
    l2_deltas_qwen = {u["unit_id"]: u["z_delta"] for u in delta_qwen["levels"]["l2"]}
    threshold_val = np.percentile(list(l2_deltas_qwen.values()), DELTA_PERCENTILE)
    candidate_sections = [sid for sid, v in l2_deltas_qwen.items() if v <= threshold_val]
    n2_in_candidates = n2_l2 in candidate_sections

    candidate_l0_ids = set()
    for u in levels_raw["l2"]:
        if u["id"] in candidate_sections:
            candidate_l0_ids.update(u["source_l0_ids"])
    n_candidates = len(candidate_l0_ids)
    cost_reduction = 1 - (n_candidates / 100)

    candidate_nli_scores = [u["max_contradiction"] for u in nli["nli"] if u["unit_id"] in candidate_l0_ids]
    fp_rate_candidates = (sum(1 for v in candidate_nli_scores if v >= 0.5) / len(candidate_nli_scores)
                           if candidate_nli_scores else None)

    # --- P-F data collection: N3 delta + clean-section deltas (pooled across seeds) ---
    n3_l0 = needle_ids["n3_novel"]
    n3_l2 = l2map[n3_l0]
    n3_delta_l2 = l2_deltas_qwen[n3_l2]
    clean = clean_sections(levels_raw, gt)
    clean_deltas = [l2_deltas_qwen[sid] for sid in clean]
    pooled_n3_delta.append(n3_delta_l2)
    pooled_clean_delta.extend(clean_deltas)

    results["per_seed"][seed] = {
        "n2_rank_at_l2_qwen": [rank_qwen, n_l2],
        "n2_rank_at_l2_llama": [rank_llama, n_l2],
        "P_A_prime_holds": p_a_prime,
        "z_l0": {"n1": z_l0_n1, "n2": z_l0_n2, "n3": z_l0_n3},
        "P_B_holds": p_b,
        "z_l1_n1": z_l1_n1,
        "P_C_holds": p_c,
        "nli_n2": n2_nli,
        "nli_corpus_frac_ge_0.5": frac_ge_05,
        "P_D_holds": p_d,
        "candidate_sections": candidate_sections,
        "n2_in_candidates": n2_in_candidates,
        "n_candidate_l1_units": n_candidates,
        "nli_cost_reduction": cost_reduction,
        "fp_rate_on_candidates_only": fp_rate_candidates,
        "n3_delta_l2": n3_delta_l2,
        "clean_section_deltas": clean_deltas,
        "clean_section_median": statistics.median(clean_deltas),
    }

# --- P-F: pooled AUC ---
auc = auc_rank(pooled_n3_delta, pooled_clean_delta)
results["pooled"] = {
    "n3_deltas_all_seeds": pooled_n3_delta,
    "clean_section_deltas_all_seeds": pooled_clean_delta,
    "clean_median": statistics.median(pooled_clean_delta),
    "P_F_n3_within_0.5_of_0": all(abs(d) < 0.5 for d in pooled_n3_delta),
    "auc_delta_novel_vs_boilerplate": auc,
}

out_path = RESULTS / "e2_phaseD_analysis.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"wrote {out_path}")
print(json.dumps(results, indent=2))
