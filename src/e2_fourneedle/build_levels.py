"""E2.2 -- Fluent-summary pyramid construction (spec E2.2).

L0 = raw clauses. L1 = one-sentence clause summaries. L2 = one summary per
10-clause section, generated from the ten L1 summaries. L3 = whole-document
summary generated from the ten L2 summaries. All generation is greedy
(seed=0, temperature=0, top_k=1), so the pyramid is deterministic given the
corpus.
"""
import argparse
import json
from pathlib import Path

from src.common import model as model_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

L1_MAX_TOKENS = 80
L2_MAX_TOKENS = 150
L3_MAX_TOKENS = 220
SECTION_SIZE = 10

SYSTEM_PROMPT = (
    "You are a precise legal-contract summarizer. Follow the instruction "
    "exactly and output only the summary, with no preamble."
)


def _generate(llm, user_prompt: str, max_tokens: int) -> str:
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=model_mod.TEMPERATURE,
        top_k=model_mod.TOP_K,
        seed=model_mod.SEED,
    )
    return result["choices"][0]["message"]["content"].strip()


def _load_corpus(seed: int) -> list[dict]:
    path = DATA_DIR / f"corpus_seed{seed}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_levels(seed: int, progress: bool = True) -> dict:
    llm = model_mod.load_model(logits_all=False)
    clauses = _load_corpus(seed)

    l1 = []
    for c in clauses:
        prompt = (
            "Summarize this contract clause in one sentence. Preserve any "
            "negations, exceptions, caps, amounts, and party obligations "
            f"exactly. Clause: {c['text']} Summary:"
        )
        summary = _generate(llm, prompt, L1_MAX_TOKENS)
        l1.append({"id": c["id"], "source_l0_ids": [c["id"]], "text": summary})
        if progress and c["id"] % 20 == 0:
            print(f"  L1: {c['id']}/{len(clauses)}")

    l2 = []
    for section_idx, start in enumerate(range(0, len(l1), SECTION_SIZE), start=1):
        block = l1[start:start + SECTION_SIZE]
        block_text = " ".join(u["text"] for u in block)
        prompt = (
            "Summarize these clause summaries in 2-3 sentences, preserving "
            f"all exceptions, caps, and contradictions: {block_text}"
        )
        summary = _generate(llm, prompt, L2_MAX_TOKENS)
        l2.append({
            "id": section_idx,
            "source_l1_ids": [u["id"] for u in block],
            "source_l0_ids": [cid for u in block for cid in u["source_l0_ids"]],
            "text": summary,
        })
        if progress:
            print(f"  L2: {section_idx}/10")

    l2_block_text = " ".join(u["text"] for u in l2)
    l3_prompt = (
        "Summarize these section summaries in 2-3 sentences, preserving all "
        f"exceptions, caps, and contradictions: {l2_block_text}"
    )
    l3_text = _generate(llm, l3_prompt, L3_MAX_TOKENS)
    l3 = [{
        "id": 1,
        "source_l2_ids": [u["id"] for u in l2],
        "source_l0_ids": [cid for u in l2 for cid in u["source_l0_ids"]],
        "text": l3_text,
    }]
    if progress:
        print("  L3: done")

    return {"seed": seed, "l0": clauses, "l1": l1, "l2": l2, "l3": l3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    levels = build_levels(args.seed)
    out_path = DATA_DIR / f"levels_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(levels, f, indent=2)
    print(f"wrote {out_path}")
    print(f"L1: {len(levels['l1'])} summaries, L2: {len(levels['l2'])}, L3: {len(levels['l3'])}")


if __name__ == "__main__":
    main()
