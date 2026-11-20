"""E10 -- minimal-edit negation of an assumption sentence A -> ~A.

Three tiers, tried in order, first applicable one wins (single swap only,
per the directive):

1. Polarity-swap table: a fixed, ordered list of word/phrase substitutions
   (multi-word negated forms checked before their bare counterparts, so
   e.g. "shall not" is matched before bare "shall" -- otherwise a
   double-negation could result). Only the first occurrence in the text is
   swapped.
2. Dependency-parse negation (spaCy): covers the directive's "numeric
   assumptions" case ("reaches a total height of 330 metres" -> "does not
   reach...") as a special case of a general rule -- find the sentence's
   root verb and negate it: insert "not" after a copula (is/are/was/were)
   or an existing modal auxiliary (shall/may/will/would/should/must/
   can/could); otherwise use do-support ("does not"/"did not" + lemma).
3. LLM fallback (pinned, greedy, fixed prompt), only if neither of the
   above applies (no swap-table match and no usable root verb found).
   Logged as "generated" -- this tier is the directive's explicit fallback,
   not the default path.

Returns (negated_text, method) where method in
{"swap_table", "dependency_parse", "llm_generated"}.
"""
import re

import spacy

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# Ordered: multi-word / negated forms before their bare counterparts.
# Each entry: (compiled regex, replacement). First match anywhere in the
# text wins; only that first occurrence is replaced.
_SWAP_RULES = [
    # Checked first: "Neither party shall be liable" negated via bare
    # "shall" -> "shall not" produces "Neither... shall not be liable", a
    # double negative with the wrong polarity. Swapping the negative-
    # polarity determiner itself ("Neither" -> "Either") is a single,
    # clean edit that correctly flips the sentence's actual polarity.
    (re.compile(r"\bNeither\b"), "Either"),
    (re.compile(r"\bshall not\b", re.IGNORECASE), "shall"),
    (re.compile(r"\bshall\b", re.IGNORECASE), "shall not"),
    (re.compile(r"\bmay not\b", re.IGNORECASE), "may"),
    (re.compile(r"\bmay\b", re.IGNORECASE), "may not"),
    (re.compile(r"\bdoes not own\b", re.IGNORECASE), "owns"),
    (re.compile(r"\bowns\b", re.IGNORECASE), "does not own"),
    (re.compile(r"\bis not\b", re.IGNORECASE), "is"),
    (re.compile(r"\bare not\b", re.IGNORECASE), "are"),
    (re.compile(r"\bis\b", re.IGNORECASE), "is not"),
    (re.compile(r"\bare\b", re.IGNORECASE), "are not"),
    (re.compile(r"\bnon-exclusive(ly)?\b", re.IGNORECASE), r"exclusive\1"),
    (re.compile(r"\bexclusive(ly)?\b", re.IGNORECASE), r"non-exclusive\1"),
    (re.compile(r"\buncapped\b", re.IGNORECASE), "capped"),
    (re.compile(r"\bunlimited\b", re.IGNORECASE), "capped"),
    (re.compile(r"\bcapped\b", re.IGNORECASE), "uncapped"),
    (re.compile(r"\bexcluded\b", re.IGNORECASE), "included"),
    (re.compile(r"\bincluded\b", re.IGNORECASE), "excluded"),
    (re.compile(r"\bclosed\b", re.IGNORECASE), "open"),
    (re.compile(r"\bopen\b", re.IGNORECASE), "closed"),
]

_MODALS = {"shall", "may", "will", "would", "should", "must", "can", "could"}

LLM_FALLBACK_PROMPT = (
    "Rewrite this sentence to assert the opposite, changing as few words "
    "as possible: {text}"
)


def _try_swap_table(text: str) -> str | None:
    for pattern, replacement in _SWAP_RULES:
        if pattern.search(text):
            return pattern.sub(replacement, text, count=1)
    return None


def _try_dependency_negation(text: str) -> str | None:
    doc = _nlp()(text)
    root = None
    for sent in doc.sents:
        root = sent.root
        break
    if root is None or root.pos_ not in ("VERB", "AUX"):
        return None

    # existing modal auxiliary among root's children -> negate right after it
    for child in root.children:
        if child.dep_ == "aux" and child.lower_ in _MODALS:
            insert_at = child.idx + len(child.text)
            return text[:insert_at] + " not" + text[insert_at:]

    if root.lemma_ == "be":
        insert_at = root.idx + len(root.text)
        return text[:insert_at] + " not" + text[insert_at:]

    # do-support: replace the root token with "does not"/"did not" + lemma
    is_past = root.tag_ in ("VBD", "VBN")
    aux = "did not" if is_past else "does not"
    return text[:root.idx] + f"{aux} {root.lemma_}" + text[root.idx + len(root.text):]


def negate(text: str, llm=None) -> tuple[str, str]:
    swapped = _try_swap_table(text)
    if swapped is not None:
        return swapped, "swap_table"

    dep_negated = _try_dependency_negation(text)
    if dep_negated is not None:
        return dep_negated, "dependency_parse"

    if llm is None:
        raise RuntimeError(
            f"no swap-table or dependency-parse rule applied to {text!r}, "
            "and no llm was provided for the fallback tier"
        )
    from src.common import model as model_mod
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Follow the instruction exactly and output only the rewritten sentence."},
            {"role": "user", "content": LLM_FALLBACK_PROMPT.format(text=text)},
        ],
        max_tokens=len(text.split()) + 20,
        temperature=model_mod.TEMPERATURE,
        top_k=model_mod.TOP_K,
        seed=model_mod.SEED,
    )
    return result["choices"][0]["message"]["content"].strip(), "llm_generated"
