"""E2.4 -- NLI verifier (spec E2.4).

For each L1 summary u: embed all L1 summaries; take the top-5 EARLIER
summaries by cosine similarity; run NLI with each as premise, u as
hypothesis; record the max contradiction probability. Implements "high
coarse surprisal proposes, NLI disposes" -- run on ALL units (not just
surprising ones) so precision/recall of the proposal step is measurable.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import io as io_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
TOP_K = 5


def run(seed: int) -> dict:
    levels = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())
    l1_texts = [u["text"] for u in levels["l1"]]
    l1_ids = [u["id"] for u in levels["l1"]]

    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
    embeddings = embedder.encode(l1_texts, normalize_embeddings=True, show_progress_bar=False)

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model.eval()
    label2id = {k.lower(): v for k, v in nli_model.config.label2id.items()}
    contradiction_idx = next(v for k, v in label2id.items() if "contra" in k)

    results = []
    for i, (uid, text) in enumerate(zip(l1_ids, l1_texts)):
        if i == 0:
            results.append({"unit_id": uid, "max_contradiction": 0.0, "top_premise_ids": []})
            continue

        sims = embeddings[:i] @ embeddings[i]
        top_idx = np.argsort(-sims)[:TOP_K]
        premises = [l1_texts[j] for j in top_idx]
        premise_ids = [l1_ids[j] for j in top_idx]

        with torch.no_grad():
            inputs = tokenizer(premises, [text] * len(premises), return_tensors="pt",
                                truncation=True, padding=True)
            logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            contradiction_probs = probs[:, contradiction_idx].tolist()

        results.append({
            "unit_id": uid,
            "max_contradiction": float(max(contradiction_probs)),
            "top_premise_ids": premise_ids,
            "contradiction_probs": contradiction_probs,
        })
        if uid % 20 == 0:
            print(f"  nli: {uid}/{len(l1_ids)}")

    return {"seed": seed, "nli": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run(args.seed)
    out_path = io_mod.save_result(f"e2_nli_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
