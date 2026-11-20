"""E2 Phase A -- conditioning-delta scoring (E2-RETEST directive).

delta_score(unit, level) = S(unit | level_context) - S(unit | empty context)

S(unit | ctx) is v1's existing mean_surprisal (reused from e2_scores_seed{K}.json).
S(unit | empty) scores the same unit in isolation. Both are single
batched-prefill passes (score_tokens with context="") -- never a stepped
decode -- so the two numbers are computed via the identical mechanism per
the E1 lesson (batched vs. sequential eval give different logits on this
CPU/quantized stack; comparing across that boundary would be invalid).

Predictions are pre-registered in results/e2_phaseA_predictions.md before
this is run on any seed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod
from src.common.scoring import score_tokens

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

LEVEL_NAMES = ["l0", "l1", "l2"]


def _isolated_mean_surprisal(llm, text: str) -> float:
    scores = score_tokens(llm, text=text, context="")
    surprisal = [-s.logprob_true for s in scores]
    return float(np.mean(surprisal)) if surprisal else 0.0


def _zscore(values: list[float]) -> list[float]:
    arr = np.array(values)
    std = arr.std()
    if std == 0:
        return [0.0] * len(values)
    return ((arr - arr.mean()) / std).tolist()


def run(seed: int) -> dict:
    levels = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())
    existing_scores = json.loads((io_mod.RESULTS_DIR / f"e2_scores_seed{seed}.json").read_text())
    llm = model_mod.load_model(logits_all=True)

    result = {"seed": seed, "levels": {}}
    for level_name in LEVEL_NAMES:
        units = levels[level_name]
        s_ctx_by_id = {u["unit_id"]: u["mean_surprisal"] for u in existing_scores["levels"][level_name]}

        deltas = []
        s_ctx_list = []
        s_null_list = []
        for u in units:
            uid = u["id"]
            s_ctx = s_ctx_by_id[uid]
            s_null = _isolated_mean_surprisal(llm, u["text"])
            deltas.append(s_ctx - s_null)
            s_ctx_list.append(s_ctx)
            s_null_list.append(s_null)

        z_delta = _zscore(deltas)

        level_out = []
        for u, s_ctx, s_null, delta, zd in zip(units, s_ctx_list, s_null_list, deltas, z_delta):
            level_out.append({
                "unit_id": u["id"],
                "S_ctx": s_ctx,
                "S_null": s_null,
                "delta": delta,
                "z_delta": zd,
            })
        result["levels"][level_name] = level_out
        print(f"  {level_name}: {len(level_out)} units delta-scored "
              f"(mean delta={np.mean(deltas):.3f})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run(args.seed)
    out_path = io_mod.save_result(f"e2_delta_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
