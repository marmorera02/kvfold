"""E2 Phase B (part 2) -- NLI against the running assumption set, gated by a
referent guard (E2-RETEST directive).

Replaces v1's "top-5 by embedding cosine" premise selection, which mostly
retrieved same-template near-duplicates and produced a ~65% false-positive
contradiction rate (see results/E2_FINDINGS.md). Here, candidate premises
are the assumption set built by assumptions.py (restatements of general
rules/caps/obligations, not routine transactional details), restricted to
assumptions sourced from EARLIER clauses only (document-order causality),
and further gated: a premise only counts if it shares a real content
referent with the hypothesis (overlapping non-stopword noun lemma, or a
shared capitalized multiword "defined term"). Contradictions the guard
suppresses are logged separately so the guard's own effect is measurable.
"""
import argparse
import json
import re
from pathlib import Path

import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common import io as io_mod

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFINED_TERM_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
# "Supplier"/"Buyer" appear in nearly every clause (they're the two contract
# parties throughout), so left unexcluded they'd make the guard pass almost
# everything on a trivial shared-entity-name basis. Excluded here along with
# other contract-structure words that carry no discriminating content.
# Product-name head nouns are excluded too: gen_corpus.py slot-fills one of
# six product names into many unrelated clause templates, so two clauses
# sharing a product noun (e.g. both mentioning "housings") is often pure
# coincidence of the random slot-fill, not a genuine shared referent --
# confirmed by inspecting Phase B's residual false positives.
GENERIC_NOUNS = {
    "agreement", "clause", "party", "parties", "supplier", "buyer",
    "schedule", "goods", "product",
    "housing", "module", "assembly", "harness", "enclosure", "sensor",
    "day", "time", "period", "notice", "amount", "provision", "event",
}


def _load_nli():
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    contradiction_idx = next(v for k, v in label2id.items() if "contra" in k)
    return model, tokenizer, contradiction_idx


def _noun_lemmas(nlp, text: str) -> set[str]:
    doc = nlp(text)
    return {
        tok.lemma_.lower() for tok in doc
        if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop
        and tok.lemma_.lower() not in GENERIC_NOUNS
    }


def _defined_terms(text: str) -> set[str]:
    return {m.group(1) for m in DEFINED_TERM_RE.finditer(text)}


def _shares_referent(nlp, premise: str, hypothesis: str) -> bool:
    if _noun_lemmas(nlp, premise) & _noun_lemmas(nlp, hypothesis):
        return True
    if _defined_terms(premise) & _defined_terms(hypothesis):
        return True
    return False


def run(seed: int) -> dict:
    levels = json.loads((DATA_DIR / f"levels_seed{seed}.json").read_text())
    l1_units = levels["l1"]
    assumptions_payload = json.loads((io_mod.RESULTS_DIR / f"e2_assumptions_seed{seed}.json").read_text())
    assumption_by_id = {r["unit_id"]: r["assumption"] for r in assumptions_payload["records"]}

    nli_model, tokenizer, contradiction_idx = _load_nli()
    nlp = spacy.load("en_core_web_sm")

    results = []
    running_assumptions: list[tuple[int, str]] = []
    for u in l1_units:
        uid = u["id"]
        text = u["text"]
        candidates = [(sid, atext) for sid, atext in running_assumptions if sid < uid]
        guarded = [(sid, atext) for sid, atext in candidates if _shares_referent(nlp, atext, text)]

        if not guarded:
            results.append({
                "unit_id": uid, "max_contradiction": 0.0,
                "premise_ids_considered": [sid for sid, _ in candidates],
                "premise_ids_guard_passed": [],
                "note": "no assumption cleared the referent guard",
            })
        else:
            premises = [atext for _, atext in guarded]
            premise_ids = [sid for sid, _ in guarded]
            with torch.no_grad():
                inputs = tokenizer(premises, [text] * len(premises), return_tensors="pt",
                                    truncation=True, max_length=256, padding=True)
                logits = nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                contradiction_probs = probs[:, contradiction_idx].tolist()
            results.append({
                "unit_id": uid,
                "max_contradiction": float(max(contradiction_probs)),
                "premise_ids_considered": [sid for sid, _ in candidates],
                "premise_ids_guard_passed": premise_ids,
                "contradiction_probs": contradiction_probs,
            })

        this_assumption = assumption_by_id.get(uid)
        if this_assumption is not None:
            running_assumptions.append((uid, this_assumption))

        if uid % 20 == 0:
            print(f"  nli_v2: {uid}/{len(l1_units)}")

    return {"seed": seed, "nli": results, "n_assumptions_total": len(running_assumptions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = run(args.seed)
    out_path = io_mod.save_result(f"e2_nli_v2_seed{args.seed}.json", payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
