"""E10 -- assumption-contrast scoring (E10 DIRECTIVE). Computes
Delta_pm(u, A) = S(u|A) - S(u|~A) for the pre-registered pair set
(results/e10_predictions.md), for one scorer per invocation (selected via
the standard PYRAMID_MODEL env var, so this script is scorer-agnostic).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod
from src.common.scoring import score_tokens
from src.e10_contrast.negate import negate

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
NEUTRAL_SAMPLE_SIZE = 300
LENGTH_RATIO_BAND = (0.8, 1.2)

SANITY_PAIRS = [
    # (seed, hyp_id, premise_id) -- 3 liability-cap pairs (seed 0),
    # wikiA's height pair, and the seed-5 n2b/clause-12 special pair.
    (0, 87, 8), (0, 87, 25), (0, 87, 50),
    ("wikiA", 85, 75),
    (5, 78, 12),
]


def _mean_surprisal_bits(llm, text: str, context: str) -> float:
    scores = score_tokens(llm, text=text, context=context)
    return float(np.mean([-s.logprob_true for s in scores])) if scores else 0.0


def _load_pairs() -> list[dict]:
    return json.loads((RESULTS_DIR / "e10_pairs.json").read_text())


def _select_scoring_set(pairs: list[dict]) -> tuple[list[dict], list[int]]:
    true_fp = [p for p in pairs if p["label"] in ("TRUE_CONTRADICTION", "KNOWN_FP")]
    neutral = [p for p in pairs if p["label"] == "NEUTRAL"]
    sample_idx = sorted(random.Random(0).sample(range(len(neutral)), NEUTRAL_SAMPLE_SIZE))
    sampled_neutral = [neutral[i] for i in sample_idx]
    return true_fp + sampled_neutral, sample_idx


def _score_one(llm, u_text: str, A_text: str) -> dict:
    negA_text, method = negate(A_text, llm=llm)
    tok_A = llm.tokenize(A_text.encode("utf-8"), add_bos=False)
    tok_negA = llm.tokenize(negA_text.encode("utf-8"), add_bos=False)
    ratio = (len(tok_negA) / len(tok_A)) if tok_A else None
    length_ok = ratio is not None and (LENGTH_RATIO_BAND[0] <= ratio <= LENGTH_RATIO_BAND[1])

    out = {
        "A_text": A_text, "negA_text": negA_text, "negation_method": method,
        "tok_count_A": len(tok_A), "tok_count_negA": len(tok_negA),
        "length_ratio": ratio, "length_ok": length_ok,
    }
    if not length_ok:
        out["excluded"] = True
        out["exclusion_reason"] = f"length ratio {ratio} outside {LENGTH_RATIO_BAND}"
        return out

    S_A = _mean_surprisal_bits(llm, u_text, A_text + "\n")
    S_negA = _mean_surprisal_bits(llm, u_text, negA_text + "\n")
    out.update({
        "excluded": False,
        "S_A": S_A, "S_negA": S_negA,
        "delta_pm": S_A - S_negA,
    })
    return out


def run_sanity_check(llm, pairs_by_key: dict) -> dict:
    print("=== E10 sanity check (5 hand-picked pairs, sign must flip on swap) ===")
    records = []
    all_ok = True
    for key in SANITY_PAIRS:
        p = pairs_by_key[key]
        u, A = p["u_l1_text"], p["A_text"]
        r1 = _score_one(llm, u, A)
        if r1.get("excluded"):
            print(f"  {key}: EXCLUDED ({r1['exclusion_reason']}) -- cannot sanity-check this pair")
            records.append({"key": key, "skipped": True, **r1})
            continue
        negA = r1["negA_text"]
        # independently recompute with roles swapped (fresh calls, not algebraic reuse)
        S_negA_direct = _mean_surprisal_bits(llm, u, negA + "\n")
        S_A_direct = _mean_surprisal_bits(llm, u, A + "\n")
        delta_normal = S_A_direct - S_negA_direct
        delta_swapped = S_negA_direct - S_A_direct
        signs_opposite = (delta_normal > 0) != (delta_swapped > 0) or delta_normal == 0
        exact_negation = abs(delta_normal + delta_swapped) < 1e-6
        ok = signs_opposite and exact_negation
        all_ok = all_ok and ok
        print(f"  {key}: delta_normal={delta_normal:.4f} delta_swapped={delta_swapped:.4f} "
              f"opposite_signs={signs_opposite} exact_negation={exact_negation} -> {'OK' if ok else 'FAIL'}")
        records.append({
            "key": key, "delta_normal": delta_normal, "delta_swapped": delta_swapped,
            "signs_opposite": signs_opposite, "exact_negation": exact_negation, "ok": ok,
        })
    return {"all_ok": all_ok, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-label", required=True, help="e.g. qwen or llama, for output filenames")
    args = parser.parse_args()

    pairs = _load_pairs()
    pairs_by_key = {(p["seed"], p["hyp_id"], p["premise_id"]): p for p in pairs}

    llm = model_mod.load_model(logits_all=True)

    sanity = run_sanity_check(llm, pairs_by_key)
    if not sanity["all_ok"]:
        out_path = io_mod.save_result(f"e10_sanity_FAILED_{args.scorer_label}.json", sanity)
        print(f"\nSANITY CHECK FAILED -- stopping per directive. Wrote {out_path}")
        raise SystemExit(1)
    print("Sanity check: all 5 pairs OK. Proceeding to full scoring.\n")

    scoring_set, neutral_sample_idx = _select_scoring_set(pairs)
    print(f"scoring {len(scoring_set)} pairs "
          f"({sum(1 for p in scoring_set if p['label']=='TRUE_CONTRADICTION')} TRUE, "
          f"{sum(1 for p in scoring_set if p['label']=='KNOWN_FP')} KNOWN_FP, "
          f"{sum(1 for p in scoring_set if p['label']=='NEUTRAL')} NEUTRAL sampled)")

    records = []
    n_excluded = 0
    for i, p in enumerate(scoring_set):
        r = _score_one(llm, p["u_l1_text"], p["A_text"])
        if r.get("excluded"):
            n_excluded += 1
        records.append({
            "seed": p["seed"], "hyp_id": p["hyp_id"], "premise_id": p["premise_id"],
            "label": p["label"], "assumption_source": p.get("assumption_source"),
            **r,
        })
        if (i + 1) % 50 == 0:
            print(f"  scored {i + 1}/{len(scoring_set)} (excluded so far: {n_excluded})")

    payload = {
        "scorer_label": args.scorer_label,
        "model_filename": model_mod.MODEL_FILENAME,
        "n_pairs_scored": len(records),
        "n_excluded": n_excluded,
        "neutral_sample_idx": neutral_sample_idx,
        "sanity_check": sanity,
        "records": records,
    }
    out_path = io_mod.save_result(f"e10_contrast_{args.scorer_label}.json", payload)
    print(f"\nscored {len(records)} pairs, {n_excluded} excluded (length gate)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
