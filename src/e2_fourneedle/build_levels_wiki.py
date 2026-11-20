"""Phase E -- fluent-summary pyramid construction, generalized from
build_levels.py for non-legal, human-written prose. Mechanism identical
(same L0-L3 structure, same greedy/deterministic generation, same
SECTION_SIZE); only the prompt wording changes, since the original wording
("contract clause", "party obligations") is domain-specific and would bias
a register-transfer test if left in place.
"""
import json
from pathlib import Path

from src.common import model as model_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

L1_MAX_TOKENS = 80
L2_MAX_TOKENS = 150
L3_MAX_TOKENS = 220
SECTION_SIZE = 10

SYSTEM_PROMPT = (
    "You are a precise encyclopedia-style summarizer. Follow the "
    "instruction exactly and output only the summary, with no preamble."
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


def _load_corpus(variant: str) -> list[dict]:
    path = DATA_DIR / f"corpus_seed{variant}.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_levels(variant: str, progress: bool = True) -> dict:
    llm = model_mod.load_model(logits_all=False)
    clauses = _load_corpus(variant)

    l1 = []
    for c in clauses:
        prompt = (
            "Summarize this passage in one sentence. Preserve any "
            "negations, exceptions, numbers, dates, and factual claims "
            f"exactly. Passage: {c['text']} Summary:"
        )
        summary = _generate(llm, prompt, L1_MAX_TOKENS)
        l1.append({"id": c["id"], "source_l0_ids": [c["id"]], "text": summary})
        if progress and c["id"] % 20 == 0:
            print(f"  L1: {c['id']}/{len(clauses)}")

    l2 = []
    n_sections = len(l1) // SECTION_SIZE
    for s in range(n_sections):
        section_units = l1[s * SECTION_SIZE:(s + 1) * SECTION_SIZE]
        joined = " ".join(u["text"] for u in section_units)
        prompt = (
            "Summarize these ten passage summaries as one paragraph, "
            "preserving any negations, exceptions, numbers, dates, and "
            f"factual claims exactly. Summaries: {joined} Summary:"
        )
        summary = _generate(llm, prompt, L2_MAX_TOKENS)
        l2.append({
            "id": s + 1,
            "source_l0_ids": [u["id"] for u in section_units],
            "text": summary,
        })
        if progress:
            print(f"  L2: section {s + 1}/{n_sections}")

    joined_l2 = " ".join(u["text"] for u in l2)
    prompt = (
        "Summarize this document-level text as one paragraph, preserving "
        "any negations, exceptions, numbers, dates, and factual claims "
        f"exactly. Text: {joined_l2} Summary:"
    )
    l3_text = _generate(llm, prompt, L3_MAX_TOKENS)
    l3 = [{"id": 1, "source_l0_ids": [c["id"] for c in clauses], "text": l3_text}]
    if progress:
        print("  L3: done")

    return {"variant": variant, "l0": clauses, "l1": l1, "l2": l2, "l3": l3}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="wikiA")
    args = parser.parse_args()
    levels = build_levels(args.variant)
    out_path = DATA_DIR / f"levels_seed{args.variant}.json"
    out_path.write_text(json.dumps(levels, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
