"""Phase E -- running assumption-set extraction, generalized from
assumptions.py for non-legal prose. Same two-step classify-then-restate
mechanism (single-shot "restate or say NONE" already failed catastrophically
once, on the legal corpus, for the same structural reason prose summaries
in general tend to sound claim-shaped); only the prompt wording and
few-shot examples change, to fit "specific, checkable factual claim" rather
than "contract cap/exclusion."

Few-shot examples below are synthetic (invented for this prompt, not drawn
from the corpus this module scores) to keep the recall test honest, same
practice as e7_ledger/extract.py.
"""
import json
from pathlib import Path

from src.common import model as model_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CLASSIFY_MAX_TOKENS = 6
RESTATE_MAX_TOKENS = 80
SYSTEM_PROMPT = (
    "You are a precise fact-checking analyst. Follow the instruction "
    "exactly and output only what is asked, with no preamble."
)

CLASSIFY_PROMPT_TEMPLATE = (
    "Does this passage summary state a SPECIFIC, CHECKABLE FACTUAL CLAIM "
    "(a measurement, date, quantity, or named property that a later "
    "passage could reaffirm or contradict) -- as opposed to ORDINARY "
    "descriptive or narrative content with no specific checkable claim?\n\n"
    "Examples:\n"
    "\"The bridge's main span measures 2,737 metres between its two "
    "towers.\" -> FACTUAL_CLAIM\n"
    "\"Construction crews often worked through heavy fog rolling in from "
    "the bay.\" -> ORDINARY\n"
    "\"The ship was registered at a gross tonnage of 46,328 tons upon "
    "completion.\" -> FACTUAL_CLAIM\n"
    "\"Passengers spent their evenings socializing in the first-class "
    "lounge.\" -> ORDINARY\n\n"
    "Now classify this summary. Answer with exactly one word: "
    "FACTUAL_CLAIM or ORDINARY.\n\nSummary: {text}"
)
RESTATE_PROMPT_TEMPLATE = (
    "Restate the specific factual claim in this summary as one concise "
    "declarative sentence, preserving its exact numbers and scope. "
    "Summary: {text}"
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


def run(variant: str) -> dict:
    levels = json.loads((DATA_DIR / f"levels_seed{variant}.json").read_text())
    l1_units = levels["l1"]
    llm = model_mod.load_model(logits_all=False)

    records = []
    for u in l1_units:
        classify_prompt = CLASSIFY_PROMPT_TEMPLATE.format(text=u["text"])
        classification = _generate(llm, classify_prompt, CLASSIFY_MAX_TOKENS)
        is_claim = "FACTUAL_CLAIM" in classification.upper() and "ORDINARY" not in classification.upper()

        assumption = None
        if is_claim:
            restate_prompt = RESTATE_PROMPT_TEMPLATE.format(text=u["text"])
            assumption = _generate(llm, restate_prompt, RESTATE_MAX_TOKENS)

        records.append({
            "unit_id": u["id"],
            "classification": classification,
            "assumption": assumption,
        })
        if u["id"] % 20 == 0:
            print(f"  assumptions_wiki: {u['id']}/{len(l1_units)}")

    n_assumptions = sum(1 for r in records if r["assumption"] is not None)
    print(f"  extracted {n_assumptions} factual claims from {len(records)} units")
    return {"variant": variant, "records": records, "n_assumptions": n_assumptions}


def main() -> None:
    import argparse
    from src.common import io as io_mod
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="wikiA")
    args = parser.parse_args()
    payload = run(args.variant)
    out_path = io_mod.save_result(f"e2_assumptions_seed{args.variant}.json", payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
