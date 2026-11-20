"""Phase E -- NLI against the running factual-claim assumption set, gated
by a referent guard, generalized from nli_v2.py for Wikipedia-register
prose. Same mechanism (assumption set restricted to earlier passages,
gated by shared-referent overlap); two domain-specific adaptations, made
proactively rather than discovered empirically, because the underlying
lesson is already known from the legal corpus:

1. GENERIC_NOUNS excludes structural/categorical words ("tower", "bridge",
   "ship", "world", "year"...) that recur constantly regardless of whether
   two passages share a genuine specific claim -- the same role
   "Supplier"/"Buyer" played for the legal corpus. "height", "metres", and
   "tall" are deliberately NOT excluded: they're the genuine, specific
   content link between the reinforced fact and its contradiction, and
   excluding them would break the one recall case this test exists to
   check.
2. nli_v2.py's `_defined_terms` (capitalized multiword phrases) had no
   exclusion list at all, because the legal corpus's defined terms were
   rare enough not to need one. Here, the corpus's own subject names
   ("Eiffel Tower", "Golden Gate Bridge") recur as capitalized phrases in
   nearly every passage sourced from that article, which would trivially
   pass the guard for almost any pair -- so a small denylist is added.
"""
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

GENERIC_NOUNS = {
    "tower", "bridge", "ship", "structure", "building", "world", "year",
    "century", "time", "site", "area", "design", "construction", "use",
    "project", "history", "record",
}
GENERIC_DEFINED_TERMS = {"Eiffel Tower", "Golden Gate Bridge", "Trans-Siberian Railway"}


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
    return {
        m.group(1) for m in DEFINED_TERM_RE.finditer(text)
        if m.group(1) not in GENERIC_DEFINED_TERMS
    }


def _shares_referent(nlp, premise: str, hypothesis: str) -> bool:
    if _noun_lemmas(nlp, premise) & _noun_lemmas(nlp, hypothesis):
        return True
    if _defined_terms(premise) & _defined_terms(hypothesis):
        return True
    return False


def run(variant: str) -> dict:
    levels = json.loads((DATA_DIR / f"levels_seed{variant}.json").read_text())
    l1_units = levels["l1"]
    assumptions_payload = json.loads((io_mod.RESULTS_DIR / f"e2_assumptions_seed{variant}.json").read_text())
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
            print(f"  nli_v2_wiki: {uid}/{len(l1_units)}")

    return {"variant": variant, "nli": results, "n_assumptions_total": len(running_assumptions)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="wikiA")
    args = parser.parse_args()
    payload = run(args.variant)
    out_path = io_mod.save_result(f"e2_nli_v2_seed{args.variant}.json", payload)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
