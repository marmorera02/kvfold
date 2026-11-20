"""E3 -- atomization (spec E3, step 2). Merges tokens into atoms: named
entities, negation bound to its head, modal+verb pairs. Gated atoms (gate
word list, numerals, named party/entities) can never appear inside a
committed fold -- enforced downstream in watershed.py by forcing them to
anchor status, and re-checked as a hard assertion in verify.py.
"""
import spacy

_NLP = None

GATE_WORDS = {
    "not", "no", "never", "except", "unless", "notwithstanding",
    "shall", "must", "may",
}
MODAL_LEMMAS = {"shall", "must", "may", "can", "will", "would", "could", "should", "might"}
PARTY_ENT_TYPES = {"PERSON", "ORG"}


def _nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _is_gated(doc, lo: int, hi: int) -> bool:
    for i in range(lo, hi + 1):
        tok = doc[i]
        if tok.lower_ in GATE_WORDS:
            return True
        if tok.like_num:
            return True
        if tok.ent_type_ in PARTY_ENT_TYPES:
            return True
    return False


def _resolve_overlaps_and_fill_gaps(n: int, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Guarantees a clean, gap-free, non-overlapping partition of [0, n).
    Only merges spans that genuinely OVERLAP (share an index) -- merely
    adjacent spans must stay separate atoms, or every sentence collapses
    into one giant atom once enough small merges tile it contiguously."""
    spans = sorted(spans)
    merged: list[list[int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    filled = []
    cursor = 0
    for lo, hi in merged:
        while cursor < lo:
            filled.append((cursor, cursor))
            cursor += 1
        filled.append((lo, hi))
        cursor = hi + 1
    while cursor < n:
        filled.append((cursor, cursor))
        cursor += 1
    return filled


def atomize(text: str):
    """Returns (doc, atoms) where atoms is an ordered list of
    {token_start, token_end (exclusive), text, char_start, char_end, gated}."""
    doc = _nlp()(text)
    n = len(doc)
    uf = UnionFind(n)

    for ent in doc.ents:
        for i in range(ent.start, ent.end):
            uf.union(ent.start, i)

    for tok in doc:
        if tok.dep_ == "neg":
            uf.union(tok.i, tok.head.i)

    for tok in doc:
        if tok.lemma_.lower() in MODAL_LEMMAS and tok.pos_ in ("AUX", "VERB"):
            if tok.head.i != tok.i:
                uf.union(tok.i, tok.head.i)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)
    raw_spans = [(min(idxs), max(idxs)) for idxs in groups.values()]
    spans = _resolve_overlaps_and_fill_gaps(n, raw_spans)

    atoms = []
    for lo, hi in spans:
        span = doc[lo:hi + 1]
        atoms.append({
            "token_start": lo,
            "token_end": hi + 1,
            "text": span.text_with_ws if hasattr(span, "text_with_ws") else span.text,
            "char_start": span.start_char,
            "char_end": span.end_char,
            "gated": _is_gated(doc, lo, hi),
        })
    return doc, atoms
