"""E2 Phase B (part 1) -- running assumption-set extraction (E2-RETEST directive).

Walks L1 summaries in document order. For each, asks (greedy, fixed prompt)
whether it states a general rule/cap/exclusion/obligation that applies
beyond a single transaction; if so, that restatement becomes a standing
"assumption" attributed to its source clause id. Routine transactional
clauses produce NONE and contribute nothing.

This replaces top-5-cosine neighbor retrieval as the NLI premise source:
instead of comparing each clause to whatever clauses happen to sound
similar, we compare it only to the small set of rules the document has
actually established so far -- which is what a contradiction is relative to.
"""
import argparse
import json
from pathlib import Path

from src.common import io as io_mod
from src.common import model as model_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CLASSIFY_MAX_TOKENS = 6
RESTATE_MAX_TOKENS = 80
SYSTEM_PROMPT = (
    "You are a precise contract analyst. Follow the instruction exactly and "
    "output only what is asked, with no preamble."
)

# First pass: a two-class, few-shot-anchored decision. Single-shot
# "restate or say NONE" (v1 of this prompt) failed catastrophically -- the
# model restated all 100/100 units instead of discriminating, because
# contract clauses are grammatically "rule-shaped" even when their content
# is routine. Splitting classification from restatement, and anchoring the
# classification with contrastive examples, fixes this.
CLASSIFY_PROMPT_TEMPLATE = (
    "Does this contract clause summary state a CAP, EXCLUSION, or LIMIT that "
    "constrains obligations across the whole agreement (such as a liability "
    "cap, a damages exclusion, or a scope-limiting rule) -- as opposed to a "
    "routine operational detail (a specific delivery time, payment period, "
    "notice window, or one-off procedural step)?\n\n"
    "Examples:\n"
    "\"Supplier's liability is capped at 100% of the Contract Value; "
    "consequential damages are excluded.\" -> CAP_OR_EXCLUSION\n"
    "\"Buyer shall pay invoices within 60 days of receipt.\" -> ROUTINE\n"
    "\"Deliveries must be accompanied by a packing list and shipped with 14 "
    "days notice.\" -> ROUTINE\n"
    "\"Notwithstanding any other provision, Supplier's liability for "
    "line-down damages shall be unlimited.\" -> CAP_OR_EXCLUSION\n\n"
    "Now classify this summary. Answer with exactly one word: "
    "CAP_OR_EXCLUSION or ROUTINE.\n\nSummary: {text}"
)
RESTATE_PROMPT_TEMPLATE = (
    "Restate the cap, exclusion, or limit in this summary as one concise "
    "declarative sentence, preserving its exact scope and any numbers. "
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


def run(seed: int) -> dict:
    levels = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())
    l1_units = levels["l1"]
    llm = model_mod.load_model(logits_all=False)

    records = []
    for u in l1_units:
        classify_prompt = CLASSIFY_PROMPT_TEMPLATE.format(text=u["text"])
        classification = _generate(llm, classify_prompt, CLASSIFY_MAX_TOKENS)
        is_cap = "CAP_OR_EXCLUSION" in classification.upper()

        assumption = None
        if is_cap:
            restate_prompt = RESTATE_PROMPT_TEMPLATE.format(text=u["text"])
            assumption = _generate(llm, restate_prompt, RESTATE_MAX_TOKENS)

        records.append({
            "unit_id": u["id"],
            "classification": classification,
            "assumption": assumption,
        })
        if u["id"] % 20 == 0:
            print(f"  assumptions: {u['id']}/{len(l1_units)}")

    n_assumptions = sum(1 for r in records if r["assumption"] is not None)
    print(f"  extracted {n_assumptions} assumptions from {len(records)} units")
    return {"seed": seed, "records": records, "n_assumptions": n_assumptions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run(args.seed)
    out_path = io_mod.save_result(f"e2_assumptions_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")
    for r in payload["records"]:
        if r["assumption"] is not None:
            print(f"  [{r['unit_id']}] {r['assumption']}")


if __name__ == "__main__":
    main()
