"""Phase E -- Wikipedia-register prose corpus (E2 POST-PHASE-D directive,
Task 3). Tests whether the locator/router framing (delta-based routing +
assumption-set NLI, validated on the synthetic templated-legal corpus in
Phases A-D and replayed open-world in Task 2) transfers to lexically
varied, non-templated, human-written prose -- as opposed to a corpus built
from cycling a small set of fixed templates, which is what produced the
"near-duplicate same-template clause" false-positive mechanism Task 2
diagnosed.

The surrounding text is genuine Wikipedia prose (data/wiki_raw/*.json,
Wikipedia REST API plaintext extracts for Eiffel_Tower, Golden_Gate_Bridge,
Titanic), split into short passages. Needle content (a reinforced fact
restated 3x, one contradiction, one weird-benign aside, one novel/unrelated
insertion) is authored for this module specifically, following the same
experimental-control logic E2's synthetic corpus already used for its own
needles: ground truth requires knowing the exact planted position and
content, which a purely-scraped real corpus can't provide, so the needles
themselves are controlled insertions into a real-prose substrate -- exactly
mirroring the original ask that motivated the E2-RETEST directive ("work
with news articles from a human written source that are slightly altered
to insert a few issues").

Two variants (mirroring Task 2's seeds 0-2 vs. 3/4 exactly):
  "wikiA": reinforced fact x3 + one contradiction (closed-world analogue)
  "wikiB": reinforced fact x3, zero contradictions (open-world control)
"""
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
WIKI_RAW_DIR = DATA_DIR / "wiki_raw"

SOURCE_ARTICLES = ["Eiffel_Tower", "Golden_Gate_Bridge", "Titanic"]

TOTAL_CLAUSES = 100
FRAME_POSITIONS = [8, 40, 75]
N1_POSITION = 20
N2_POSITION = 85
N3_POSITION = 55
N4_POSITION = 30

FRAME_TEXTS = [
    "The Eiffel Tower stands 330 metres tall, a height that made it the "
    "tallest human-made structure in the world when it was completed in "
    "1889.",
    "At its full height of 330 metres, including the antennas at its "
    "summit, the Eiffel Tower remains the tallest structure in Paris to "
    "this day.",
    "Measured to the tip of its uppermost antenna, the Eiffel Tower "
    "reaches a total height of 330 metres above ground level.",
]

N1_TEXT = (
    "Riveting of the tower's wrought-iron lattice was carried out by teams "
    "who, according to a persistent but unverified anecdote repeated among "
    "Parisian tour guides, preferred to work barefoot on the upper "
    "platforms during summer months for better grip on the heated iron, a "
    "claim that appears in no contemporary construction record but is "
    "recited in several modern walking-tour scripts regardless."
)

N2_TEXT = (
    "Following a 2022 antenna-replacement project, engineers confirmed "
    "that the Eiffel Tower's overall height had been revised upward to "
    "345 metres, though the change has not yet been reflected in most "
    "public records or tourist signage."
)

N3_TEXT = (
    "Separately, the site's ongoing structural-monitoring program has, "
    "since 2019, incorporated fibre-optic strain sensors originally "
    "developed for aerospace fuselage inspection -- a technology transfer "
    "unrelated to the tower's original ironwork or its later role in "
    "early radio transmission."
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")
HEADER_RE = re.compile(r"^=+\s*.*?\s*=+$")


def _load_sentences(article: str) -> list[str]:
    path = WIKI_RAW_DIR / f"{article}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    extract = next(iter(data["query"]["pages"].values()))["extract"]

    sentences = []
    for para in extract.split("\n"):
        para = para.strip()
        if not para or HEADER_RE.match(para):
            continue
        for sent in SENTENCE_SPLIT_RE.split(para):
            sent = sent.strip()
            if len(sent) >= 25:  # drop stray fragments/citation markers
                sentences.append(sent)
    return sentences


def _group_into_passages(sentences: list[str], target_chars: int = 320) -> list[str]:
    passages = []
    buf = []
    buf_len = 0
    for sent in sentences:
        buf.append(sent)
        buf_len += len(sent) + 1
        if buf_len >= target_chars:
            passages.append(" ".join(buf))
            buf, buf_len = [], 0
    if buf:
        passages.append(" ".join(buf))
    return passages


def _ordinary_pool() -> list[str]:
    pool = []
    for article in SOURCE_ARTICLES:
        pool.extend(_group_into_passages(_load_sentences(article)))
    return pool


def build_corpus(variant: str) -> tuple[list[dict], dict]:
    assert variant in ("wikiA", "wikiB")
    pool = _ordinary_pool()
    assert len(pool) >= TOTAL_CLAUSES, (
        f"ordinary pool too small ({len(pool)}) for {TOTAL_CLAUSES} clauses"
    )

    special: dict[int, tuple[str, str]] = {
        FRAME_POSITIONS[0]: ("reinforced_fact", FRAME_TEXTS[0]),
        FRAME_POSITIONS[1]: ("reinforced_fact", FRAME_TEXTS[1]),
        FRAME_POSITIONS[2]: ("reinforced_fact", FRAME_TEXTS[2]),
        N1_POSITION: ("n1_weird_benign", N1_TEXT),
        N3_POSITION: ("n3_novel", N3_TEXT),
    }
    needles = [
        {"id": N1_POSITION, "type": "n1_weird_benign"},
        {"id": N3_POSITION, "type": "n3_novel"},
        {"id": N4_POSITION, "type": "n4_control"},
    ]
    if variant == "wikiA":
        special[N2_POSITION] = ("n2_contradiction", N2_TEXT)
        needles.append({"id": N2_POSITION, "type": "n2_contradiction"})

    pool_iter = iter(pool)
    clauses = []
    for clause_id in range(1, TOTAL_CLAUSES + 1):
        if clause_id in special:
            ctype, text = special[clause_id]
            next(pool_iter)  # this ordinary passage is displaced, not consumed twice
        else:
            text = next(pool_iter)
            ctype = "n4_control" if clause_id == N4_POSITION else "wiki_passage"
        clauses.append({"id": clause_id, "type": ctype, "text": text})

    pos = 0
    for c in clauses:
        header = f"Passage {c['id']}. "
        full = header + c["text"]
        c["char_start"] = pos
        c["char_end"] = pos + len(full)
        c["full_text"] = full
        pos = c["char_end"] + 2

    ground_truth = {
        "variant": variant,
        "total_clauses": TOTAL_CLAUSES,
        "reinforced_fact_clause_ids": list(FRAME_POSITIONS),
        "n_contradictions": 1 if variant == "wikiA" else 0,
        "needles": needles,
        "source_articles": SOURCE_ARTICLES,
    }
    return clauses, ground_truth


def main() -> None:
    for variant in ("wikiA", "wikiB"):
        clauses, ground_truth = build_corpus(variant)
        out_corpus = DATA_DIR / f"corpus_seed{variant}.jsonl"
        out_truth = DATA_DIR / f"ground_truth_seed{variant}.json"
        with open(out_corpus, "w", encoding="utf-8") as f:
            for c in clauses:
                f.write(json.dumps({
                    "id": c["id"], "type": c["type"], "text": c["text"],
                    "char_start": c["char_start"], "char_end": c["char_end"],
                }) + "\n")
        out_truth.write_text(json.dumps(ground_truth, indent=2))
        total_chars = sum(len(c["full_text"]) for c in clauses)
        print(f"{variant}: wrote {len(clauses)} clauses, {total_chars} chars -> {out_corpus}")


if __name__ == "__main__":
    main()
