"""E7 (Cycle 3 parallel track) -- structured ledger extraction.

Alternative to E2 Phase B's "assumption set + NLI" approach. Instead of
restating a clause as free text and later asking a black-box NLI model
whether two restatements contradict, extract each clause's standing rule
(if any) as a structured (SCOPE, ATTRIBUTE, VALUE) triple, so that
conflict detection (conflicts.py) can be a deterministic rule over
structured fields instead of a learned entailment judgment.

Directly targets Task 2's diagnosed failure: clause 12 (seed 5's
exclusivity frame, stated ONCE) was never extracted into the Phase B
assumption set because that classify prompt's few-shot examples were all
liability-flavored ("cap"/"exclusion"/"limit"), so a same-shape-but-
different-topic rule (an exclusivity/sourcing obligation) fell outside
what the model recognized as "rule-shaped." Here the classify step's
few-shot set spans multiple rule TOPICS (liability, exclusivity,
ownership), not just liability, on the theory that topic-general framing
is what was missing, not classification skill in general.

Important methodological note: none of the few-shot examples below reuse
corpus text (verbatim or paraphrased) from any of the clauses this module
will actually be scored against (8, 12, 25, 50, 78, 87). They are
synthetic, invented for this prompt only, so that the recall test (P-Q)
is not measuring memorization of its own prompt.

Prompt iteration log (max 3 allowed by the E7 directive):
  v1 (this file, first run): topic-general classify + scope/attribute/value
     extract, as described above. See e7_recall_results.md for the outcome
     that determined whether v2/v3 were needed.
"""
import argparse
import json
import re
from pathlib import Path

from src.common import io as io_mod
from src.common import model as model_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PROMPT_VERSION = "v1"

CLASSIFY_MAX_TOKENS = 6
EXTRACT_MAX_TOKENS = 60

SYSTEM_PROMPT = (
    "You are a precise contract analyst. Follow the instruction exactly and "
    "output only what is asked, with no preamble."
)

# Deliberately spans three different rule TOPICS (liability, exclusivity,
# ownership) plus one explicit "override" example, so the model isn't only
# anchored on the liability-cap shape the way assumptions.py's classifier
# was. All examples are synthetic -- not drawn from this corpus.
CLASSIFY_PROMPT_TEMPLATE = (
    "Does this contract clause establish a STANDING RULE about a specific "
    "topic -- a cap, limit, exclusion, restriction, permission, or "
    "ownership assignment that a DIFFERENT clause elsewhere in the same "
    "agreement could later confirm, extend, or override -- as opposed to a "
    "ROUTINE operational detail (a one-off delivery time, payment period, "
    "or notice window that no other clause would ever reference or "
    "override)?\n\n"
    "Examples:\n"
    "\"Consultant's total liability under this engagement shall not exceed "
    "$50,000 in aggregate.\" -> RULE\n"
    "\"Consultant shall submit invoices monthly, with payment due within 30 "
    "days.\" -> ROUTINE\n"
    "\"Distributor shall market the Products solely within the Designated "
    "Territory and shall not sell or promote competing products during the "
    "Term.\" -> RULE\n"
    "\"Licensee shall provide a usage report within 10 business days of "
    "each quarter-end.\" -> ROUTINE\n"
    "\"All right, title, and interest in the Deliverables shall vest "
    "exclusively in Client upon final payment.\" -> RULE\n"
    "\"Notwithstanding the exclusivity clause above, Distributor may enter "
    "non-competing distribution arrangements with prior written notice.\" "
    "-> RULE\n\n"
    "Answer with exactly one word: RULE or ROUTINE.\n\nClause: {text}"
)

EXTRACT_PROMPT_TEMPLATE = (
    "Extract the standing rule in this clause as a SCOPE, ATTRIBUTE, and "
    "VALUE.\n"
    "SCOPE: the specific topic the rule governs, 2-4 words (e.g. "
    "'liability', 'sourcing exclusivity', 'IP ownership').\n"
    "ATTRIBUTE: the specific parameter within that scope (e.g. 'cap', "
    "'permitted alternative suppliers', 'ownership party').\n"
    "VALUE: the core permission, restriction, or cap amount that DEFINES "
    "the rule -- not an incidental procedural number (like a number of "
    "days to complete an administrative step) that happens to appear in "
    "the same sentence.\n\n"
    "Examples:\n"
    "Clause: \"Consultant's total liability under this engagement shall "
    "not exceed $50,000 in aggregate, and Consultant shall have no "
    "liability for lost profits or indirect losses.\"\n"
    "SCOPE: liability | ATTRIBUTE: cap | VALUE: $50,000 aggregate; excludes "
    "lost profits and indirect losses\n\n"
    "Clause: \"Distributor shall market the Products solely within the "
    "Designated Territory and shall not sell or promote competing products "
    "during the Term.\"\n"
    "SCOPE: sourcing exclusivity | ATTRIBUTE: permitted alternatives | "
    "VALUE: none permitted, exclusive arrangement\n\n"
    "Clause: \"Notwithstanding the exclusivity clause above, Distributor "
    "may enter into non-competing distribution arrangements with prior "
    "written notice to Client.\"\n"
    "SCOPE: sourcing exclusivity | ATTRIBUTE: permitted alternatives | "
    "VALUE: permitted with prior written notice\n\n"
    "Now extract from this clause. Output exactly one line, in exactly "
    "this format:\n"
    "SCOPE: <label> | ATTRIBUTE: <label> | VALUE: <value>\n\n"
    "Clause: {text}\nOutput:"
)

EXTRACT_LINE_RE = re.compile(
    r"SCOPE:\s*(.+?)\s*\|\s*ATTRIBUTE:\s*(.+?)\s*\|\s*VALUE:\s*(.+)", re.IGNORECASE
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


def _parse_extraction(raw: str) -> dict | None:
    m = EXTRACT_LINE_RE.search(raw.replace("\n", " "))
    if not m:
        return None
    scope, attribute, value = (g.strip().strip(".") for g in m.groups())
    if not scope or not attribute or not value:
        return None
    return {"scope": scope, "attribute": attribute, "value": value}


def run(seed: int) -> dict:
    levels = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())
    l1_units = levels["l1"]
    llm = model_mod.load_model(logits_all=False)

    records = []
    for u in l1_units:
        classify_prompt = CLASSIFY_PROMPT_TEMPLATE.format(text=u["text"])
        classification = _generate(llm, classify_prompt, CLASSIFY_MAX_TOKENS)
        is_rule = "RULE" in classification.upper() and "ROUTINE" not in classification.upper()

        entry = None
        raw_extraction = None
        if is_rule:
            extract_prompt = EXTRACT_PROMPT_TEMPLATE.format(text=u["text"])
            raw_extraction = _generate(llm, extract_prompt, EXTRACT_MAX_TOKENS)
            entry = _parse_extraction(raw_extraction)

        records.append({
            "clause_id": u["id"],
            "l1_text": u["text"],
            "classification": classification,
            "raw_extraction": raw_extraction,
            "ledger_entry": entry,
        })
        if u["id"] % 20 == 0:
            print(f"  e7 extract: {u['id']}/{len(l1_units)}")

    n_extracted = sum(1 for r in records if r["ledger_entry"] is not None)
    n_classified_rule_unparsed = sum(
        1 for r in records if r["raw_extraction"] is not None and r["ledger_entry"] is None
    )
    print(f"  seed {seed}: {n_extracted} ledger entries "
          f"({n_classified_rule_unparsed} RULE-classified but unparseable)")
    return {
        "seed": seed,
        "prompt_version": PROMPT_VERSION,
        "n_units": len(records),
        "n_extracted": n_extracted,
        "n_classified_rule_unparsed": n_classified_rule_unparsed,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run(args.seed)
    out_path = io_mod.save_result(f"e7_ledger_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
