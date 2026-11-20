"""E2.3 -- Scale-space scoring (spec E2.3).

For each level k and unit u in document order: surprisal(u | all preceding
units at level k, concatenated), computed as one whole-level score_tokens
pass (respecting the fixed block scheme) sliced per unit. Z-scored within
each level, since levels have different baseline perplexities.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod
from src.common.scoring import score_tokens, unit_token_spans

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

LEVEL_NAMES = ["l0", "l1", "l2", "l3"]


def _score_level(llm, unit_texts: list[str]) -> list[dict]:
    if len(unit_texts) == 1:
        scores = score_tokens(llm, text=unit_texts[0], context="")
        surprisal = [-s.logprob_true for s in scores]
        return [{
            "mean_surprisal": float(np.mean(surprisal)),
            "max_surprisal": float(np.max(surprisal)),
        }]

    full_text, spans = unit_token_spans(llm, unit_texts)
    scores = score_tokens(llm, text=full_text, context="")
    surprisal = [-s.logprob_true for s in scores]

    out = []
    for start, end in spans:
        local = surprisal[start:end]
        out.append({
            "mean_surprisal": float(np.mean(local)) if local else 0.0,
            "max_surprisal": float(np.max(local)) if local else 0.0,
        })
    return out


def _zscore(values: list[float]) -> list[float]:
    arr = np.array(values)
    std = arr.std()
    if std == 0:
        return [0.0] * len(values)
    return ((arr - arr.mean()) / std).tolist()


def run(seed: int) -> dict:
    levels = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())
    llm = model_mod.load_model(logits_all=True)

    result = {"seed": seed, "levels": {}}
    for level_name in LEVEL_NAMES:
        units = levels[level_name]
        texts = [u["text"] for u in units]
        scores = _score_level(llm, texts)
        z_mean = _zscore([s["mean_surprisal"] for s in scores])
        z_max = _zscore([s["max_surprisal"] for s in scores])

        level_out = []
        for u, s, zm, zx in zip(units, scores, z_mean, z_max):
            level_out.append({
                "unit_id": u["id"],
                "mean_surprisal": s["mean_surprisal"],
                "max_surprisal": s["max_surprisal"],
                "z_mean": zm,
                "z_max": zx,
            })
        result["levels"][level_name] = level_out
        print(f"  {level_name}: {len(level_out)} units scored")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run(args.seed)
    out_path = io_mod.save_result(f"e2_scores_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
