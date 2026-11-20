"""E2 Phase C -- cross-model delta-scoring control (E2-RETEST directive).

Reruns Phase A's delta-scoring using a DIFFERENT model to do the scoring,
while keeping the existing Qwen-generated summaries unchanged (not
regenerated). This separates RC1's two candidate mechanisms: if delta
inversion for N2 persists under a different scorer, it's not a
Qwen-self-scoring artifact (topic-priming / register-dominance are still
live candidates); if it disappears under the swap, self-scoring was the
cause.

Unlike delta_score.py, S_ctx is recomputed fresh here rather than reused
from e2_scores_seed{K}.json, since S_ctx and S_null must come from the same
scorer model to be validly compared. Select the scorer via PYRAMID_MODEL env
var (see src/common/model.py); results carry that model's fingerprint and
must not be compared against same-named seeds scored with the primary model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod
from src.common.scoring import score_tokens, unit_token_spans

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LEVEL_NAMES = ["l0", "l1", "l2"]


def _score_level_ctx(llm, unit_texts: list[str]) -> list[float]:
    if len(unit_texts) == 1:
        scores = score_tokens(llm, text=unit_texts[0], context="")
        surprisal = [-s.logprob_true for s in scores]
        return [float(np.mean(surprisal))] if surprisal else [0.0]
    full_text, spans = unit_token_spans(llm, unit_texts)
    scores = score_tokens(llm, text=full_text, context="")
    surprisal = [-s.logprob_true for s in scores]
    return [float(np.mean(surprisal[start:end])) if surprisal[start:end] else 0.0
            for start, end in spans]


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
    llm = model_mod.load_model(logits_all=True)

    result = {"seed": seed, "scorer_model": model_mod.MODEL_FILENAME, "levels": {}}
    for level_name in LEVEL_NAMES:
        units = levels[level_name]
        texts = [u["text"] for u in units]
        s_ctx_list = _score_level_ctx(llm, texts)
        s_null_list = [_isolated_mean_surprisal(llm, t) for t in texts]
        deltas = [c - n for c, n in zip(s_ctx_list, s_null_list)]
        z_delta = _zscore(deltas)

        level_out = []
        for u, s_ctx, s_null, delta, zd in zip(units, s_ctx_list, s_null_list, deltas, z_delta):
            level_out.append({
                "unit_id": u["id"], "S_ctx": s_ctx, "S_null": s_null,
                "delta": delta, "z_delta": zd,
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
    out_path = io_mod.save_result(f"e2_delta_crossmodel_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")
    print(f"scorer model: {payload['scorer_model']} "
          f"(cross-model control -- do not compare fingerprints with e2_scores_seed{args.seed}.json)")


if __name__ == "__main__":
    main()
