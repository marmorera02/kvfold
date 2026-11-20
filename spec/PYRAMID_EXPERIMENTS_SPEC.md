# Surprisal Pyramid — Experiment Specification

Owner: Christoph Deuel · Date: 2026-07-07 · Target: Windows laptop, everything inside a Docker container (WSL2 backend), AMD GPU, Python, llama.cpp
Purpose: falsify or confirm the load-bearing claims behind surprise-gated hierarchical memory before building the full system.

Priority order for today: **E0 → E1 → E2**. E3 and E4 are stretch goals. E5 is a later phase (outline only).
E2 is the gate: if cross-scale surprisal profiles do not separate the four needle types, stop and rethink.

---

## 0. Shared environment and determinism contract

### 0.1 Stack and container architecture

**CPU-only, single lane, everything inside the container.** All scoring *and* generation run on the CPU (Ryzen 7 Pro) with fixed thread count — bit-deterministic end to end, no GPU passthrough, no host services. This is a deliberate choice: for these experiments, exact reproducibility is worth more than generation speed. Set `n_threads` to physical cores (not SMT threads) and pin it in env.json; leave 1–2 cores headroom if working on the machine during runs.

```
Container: python 3.11 on Ubuntu 24.04 base
llama-cpp-python        # default CPU build (compiles with native AVX flags in the container)
sentence-transformers   # embeddings: BAAI/bge-small-en-v1.5 (CPU)
transformers + torch    # CPU build, only used for NLI
spacy + en_core_web_sm  # atomizer (E3)
zstandard, numpy, matplotlib, tqdm
Volumes: ./models, ./data, ./results mounted from host; container is stateless otherwise
Docker: default WSL2 backend; give the WSL VM enough RAM in .wslconfig (8 GB+ for model + NLI + embeddings)
```

Throughput planning: a 3B Q5 model on this class of CPU does roughly a few hundred tokens/s prefill and ~10–20 tokens/s decode. E2's ~130 summaries at ~50 output tokens each ≈ 10–20 min of generation; scoring passes over the ~20k-token corpus per level are prefill-bound and fast. If generation feels slow, batch summaries sequentially overnight rather than reintroducing GPU complexity.

### 0.2 Model (pinned)

- Primary: `Qwen2.5-3B-Instruct-Q5_K_M.gguf` (fallback: `Llama-3.2-3B-Instruct-Q5_K_M.gguf`).
- Record in `results/env.json`: model filename, SHA256, llama-cpp-python version, n_ctx, n_batch, n_threads, CPU model, container image digest.
- **Determinism pins (all experiments):** `seed=0`, `temperature=0.0`, `top_k=1`, fixed `n_batch=512`, fixed `n_threads` (pin explicitly; don't let it default to core count, which differs between container and host). All "generation" is greedy. All scoring uses `logits_all=True` on the CPU instance.
- NLI model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (CPU).

### 0.3 Core scoring primitive (used everywhere)

`score_tokens(text, context) -> list[TokenScore]` where `TokenScore = {token_id, logprob_true, rank_true, entropy}`:

1. Tokenize `context + text` with the GGUF tokenizer.
2. One eval pass with `logits_all=True` over the concatenation (chunk if > n_ctx, see 0.4).
3. For each position i inside `text`: softmax the logits at position i−1; record log p(true token), the rank of the true token in the sorted distribution, and the distribution entropy.
4. Surprisal of a unit (atom / clause / summary) = mean over its tokens of −log2 p. Also keep max.

### 0.4 Long-text handling

Fixed block scheme: blocks of 2048 tokens with a 512-token left-context prefix carried from the previous block. No sliding per-token window (too slow). Document block boundaries in results.

### 0.5 Repo layout

```
pyramid_experiments/
  spec/PYRAMID_EXPERIMENTS_SPEC.md   (this file)
  src/
    common/ (model.py, scoring.py, io.py, det.py)
    e0_sanity.py
    e1_rankcodec.py
    e2_fourneedle/ (gen_corpus.py, build_levels.py, score_levels.py, nli.py, classify.py, plots.py)
    e3_folding/ (atomize.py, watershed.py, verify.py, run.py)
    e4_speed/ (bench.py, qa.py)
  data/            (generated corpora, ground truth JSON)
  results/         (JSON + PNG per experiment, env.json)
```

Every experiment is a CLI: `python -m src.e1_rankcodec --doc data/x.txt --out results/e1.json`. Every result JSON embeds the env fingerprint.

---

## E0 — Infrastructure sanity (30 min)

**Goal:** prove we can (a) get per-token logprobs/ranks deterministically, (b) reproduce them bit-identically across two runs.

**Procedure:**
1. Load the model with pins from 0.2 inside the container (single instance serves both scoring and generation).
2. Score a fixed 500-token paragraph twice (fresh process each time). Assert identical rank streams.
3. Greedy-generate 100 tokens from a fixed prompt twice (fresh process each time). Assert identical output.
4. Print tokens/sec for prefill and decode separately (llama.cpp reports both), plus wall-clock for the 500-token scoring pass — these numbers calibrate the runtime estimates for E1/E2.

**Pass:** bit-identical rank streams and generations across runs.
**Fail action:** do not proceed to E1 until determinism holds — everything downstream depends on it. First suspects if it fails: unpinned `n_threads`, or a llama-cpp-python build with nondeterministic thread scheduling in the attention kernels (fix by setting `n_threads=1` for scoring as a last resort; scoring is prefill-bound and parallel within the batch, so even this is tolerable for E0–E2 scale).

---

## E1 — Rank-stream lossless codec (1–2 h)

**Claim tested:** storage cost per token under LM-coding equals its surprisal; the bits/token curve spikes exactly at planted surprising content; reconstruction is bit-exact.

**Procedure:**
1. Input: one ~15–20k-token document = the E2 synthetic contract (so needle positions are known ground truth). Run this after `gen_corpus.py` exists; a plain long text works as an interim smoke test.
2. Encode: for every token position, record `rank_true` (0 = model's greedy top-1). Serialize ranks as varints, compress with zstd level 19. This is the residual stream.
3. Baselines: zstd-19 and gzip-9 on the raw UTF-8 text.
4. Decode (verification of losslessness): sequentially greedy-decode; at each step take the token at stored rank r from the sorted distribution; feed it back. Do this for at least the first 2048 tokens plus a 512-token window around each needle. Assert byte-equality with the original.
5. Plot: rolling mean (window 25 tokens) of bits/token across the document, with needle positions marked as vertical lines.

**Metrics (results/e1.json):** bits/token (LM codec vs zstd vs gzip, normalized per LM token and per character), fraction of rank-0 tokens, reconstruction exactness (bool + mismatch positions if any), per-needle local bits/token vs document median.

**Pass:** lossless reconstruction; LM codec < zstd on this text; visible spike at the *weird-but-benign* needle. **Expected and important:** NO spike at the *fluent-but-contradictory* needle — this is the documented failure of token-level surprisal that E2 exists to fix. Record it explicitly; it is the motivating figure for E2.

---

## E2 — Four-needle scale-space experiment (THE GATE, 3–4 h)

**Claim tested:** salience is the *profile of surprisal across pyramid levels*, not its amplitude at one level. The four planted types are separable by profile:

| Type | L0 surprisal | L2 surprisal | NLI contradiction |
|---|---|---|---|
| N4 boilerplate control | low | low | low |
| N1 weird-but-benign | **high** | low (folds away) | low |
| N2 fluent-but-contradictory | low | **high** | **high** |
| N3 genuinely novel | high | high | low |

### E2.1 Corpus generator (`gen_corpus.py`)

Synthetic supply agreement, ~100 numbered clauses, ~18–22k tokens total. Deterministic given `--seed`.

- ~80 boilerplate clauses from 8–10 templates (delivery, warranty, payment, confidentiality, force majeure, IP, compliance, termination) with light paraphrase variation via template slots (party names, durations, amounts). No LLM needed for generation — string templates only, so ground truth stays exact.
- Liability frame established early and reinforced: clauses ~8, ~25, ~50 state variations of "Supplier's aggregate liability shall be capped at 100% of the annual contract value; consequential and indirect damages are excluded."
- **N1 (weird-but-benign), clause ~33:** ordinary content with surface weirdness — e.g. a benign notice clause containing a garbled word ("notices shall be sent by regustered mail") and an unusual but harmless phrasing. Content contradicts nothing.
- **N2 (fluent-but-contradictory), clause ~87:** perfectly fluent legal English, long-range contradiction of the cap frame: "Notwithstanding any other provision of this Agreement, Supplier's liability for line-down damages at Buyer's production facilities shall be unlimited and shall not be subject to the limitations set forth herein."
- **N3 (genuinely novel), clause ~61:** off-topic but coherent new subject never mentioned elsewhere (e.g. a clause about quantum-computing export-control classification of shipped goods).
- **N4 control:** designate one ordinary mid-document boilerplate clause (~45) as the measured control.
- Output: `data/corpus_seed{K}.jsonl` (one clause per line: id, text, char span) + `data/ground_truth_seed{K}.json` (needle ids, types, the frame-clause ids). Generate seeds 0, 1, 2.

### E2.2 Pyramid construction (`build_levels.py`)

Fluent-summary pyramid (per the OOD-scorer decision — levels are generated summaries, not token-deletion skeletons):

- **L0** = raw clauses.
- **L1** = one-sentence summary per clause. Prompt (fixed, temperature 0): "Summarize this contract clause in one sentence. Preserve any negations, exceptions, caps, amounts, and party obligations exactly. Clause: {text}. Summary:"
- **L2** = one summary per section of 10 consecutive clauses, generated from the ten L1 summaries: "Summarize these clause summaries in 2–3 sentences, preserving all exceptions, caps, and contradictions: {L1 block}".
- **L3** = whole-document summary from the ten L2 summaries (same style).
- Store `data/levels_seed{K}.json` with full provenance (which L0 ids feed each L1/L2/L3 node).

### E2.3 Scale-space scoring (`score_levels.py`)

For each level k and each unit u in document order: surprisal(u | all preceding units at level k, concatenated, within the block scheme of 0.4). Then z-score within each level (levels have different baseline perplexities; z makes them comparable).

Output per unit: `{level, unit_id, mean_surprisal, max_surprisal, z}`. For needles, map L0 clause → its L1 summary → its L2 section to get the profile vector `(z_L0, z_L1, z_L2)`.

### E2.4 NLI verifier (`nli.py`)

For each L1 summary u: embed all L1 summaries (bge-small); take the top-5 earlier summaries by cosine; run NLI with each as premise, u as hypothesis; record max contradiction probability. (This implements "high coarse surprisal proposes, NLI disposes" — run NLI on all units, not just surprising ones, so we can measure precision/recall of the proposal step too.)

### E2.5 Classifier and pass criteria (`classify.py`)

Simple fixed-threshold rules (choose thresholds on seed 0, evaluate frozen on seeds 1 and 2):

```
boilerplate:  z_L0 < 1  and z_L2 < 1
weird-benign: z_L0 ≥ 1  and z_L2 < 1  and nli < 0.5
contradiction: z_L2 ≥ 1 and nli ≥ 0.5
novel:        z_L0 ≥ 1  and z_L2 ≥ 1  and nli < 0.5
```

**Metrics (results/e2_seed{K}.json):** profile vectors for all four measured clauses; needle rank within each level's z-ordering; NLI contradiction scores for all L1 units (so false-positive rate is measurable); confusion matrix of the rule classifier over {N1,N2,N3,N4} × 3 seeds.

**Plots:** (a) the quadrant figure — z_L0 on x, z_L2 on y, all clauses grey, four measured clauses colored by type; (b) per-level surprisal curves along the document with needles marked.

**PASS (gate):** 4/4 correct classification on ≥ 2 of 3 held-out-threshold seeds, AND N2 specifically: z_L0 < 1 with z_L2 ≥ 1 (i.e. invisible at token level, visible at concept level). **FAIL:** N2 not separable → the central claim is wrong as formulated; do not build E3+ machinery; revisit whether L2 windows actually make the cap-frame and exception adjacent (check block boundaries first — adjacency is the mechanism).

---

## E3 — Verified folding transactions (stretch, 2–3 h)

**Claim tested:** propose-then-verify folding self-calibrates across text types with no genre detection: compression ratio varies by type, fidelity never does.

**Procedure:**
1. Inputs: four ~1500-token samples in `data/e3/`: (a) legal boilerplate (reuse corpus clauses), (b) news-style prose, (c) a markdown table of numbers, (d) Python code.
2. `atomize.py` (spaCy): merge tokens into atoms — named entities, number/date spans, negation bound to its head (dep == neg → merge with head verb/aux and its particle), modal+verb pairs. Gate list forcing anchor status: {not, no, never, except, unless, notwithstanding, shall, must, may, plus all numerals and party names}. Gated atoms can never be inside a committed fold.
3. Score atoms (left surprisal, mean over atom tokens; single pass).
4. `watershed.py`: anchors = gated atoms ∪ atoms with z ≥ 0.5; every valley atom drains to its nearest left anchor; proposals = maximal contiguous valley runs, ordered by estimated cost (sum of surprisal) ascending.
5. `verify.py` — the transaction: for each proposal in order, context = current text with all previously committed folds removed, up to the span start; greedy-decode len(span) tokens; record rank stream; residual bits = len(zstd(varint(ranks))) × 8. Budget θ = 1.0 bit × span-token-count. Commit (store residual, remove span) if bits ≤ θ, else promote span to anchor. One level only for v1 (no recursion today).
6. Re-verify a random 20% of committed folds against the *final* post-fold context and report any residual growth (checks the sequential-commit consistency argument).

**Metrics per text type (results/e3.json):** fold fraction (tokens folded / total), residual bits/token of committed folds, rejection rate, and reconstruction exactness on 10 randomly sampled committed folds (replay residual, assert byte equality).

**Pass:** boilerplate fold fraction > prose > code/table (expected roughly 60–80% / 30–50% / <15%, but the *ordering* is the claim, not the numbers); zero reconstruction mismatches; zero gated atoms inside folds.

---

## E4 — Decode speed vs. condensation on CPU (stretch, 1–2 h)

**Claim tested:** smaller live KV → measurably faster decode, and condensed contexts answer as well or better than full context — the speed–fidelity frontier. CPU is actually a *favorable* setting for this measurement: decode is memory-bandwidth-bound, and on CPU the KV cache competes directly with model weights for the same DRAM bandwidth, so the decode-slowdown-per-KV-token effect should be clearly visible. The practical stakes are also higher here — on a CPU-only setup, every live token avoided at query time is felt directly in wall-clock.

**Procedure:**
1. `bench.py`: for KV fills {512, 2k, 4k, 8k, 16k} (as n_ctx and RAM allow; 16k prefill on CPU takes a couple of minutes — fine, it runs once): prefill a context of that exact token length, then greedy-generate 256 tokens; record prefill t/s and decode t/s separately, 3 repetitions, median. Plot decode t/s vs live KV tokens. Keep other processes quiet during the run; CPU benchmarks are noise-sensitive.
2. `qa.py`: 10 questions over the E2 corpus with gold key-fact answers in `data/questions.json` (include: "Is Supplier's liability capped in all cases?" — gold: no, line-down damages uncapped; plus cap amount, notice method, payment terms, the novel-clause topic, etc.). Answer under three context regimes: (a) full corpus text (if it fits n_ctx; else sliding-window best effort, note it), (b) all L1 summaries + top-3 L0 clauses retrieved by embedding similarity to the question ("pyramid regime"), (c) L2 summaries only. Score: gold-key-fact containment (case-insensitive substring / manual check file). Record live prompt tokens and wall-clock per answer.
3. Combine: table of regime × {accuracy, live tokens, prefill s, decode t/s, total s}.

**Pass:** decode t/s monotonically decreases with KV fill (sanity of the premise); pyramid regime ≥ full-context accuracy at < 30% of live tokens. The interesting open result: whether question 1 (the N2 contradiction) survives regime (c) — prediction from the design: it survives only if the L2 summary preserved the exception, which measures L2 summary quality.

---

## E5 — LoRA consolidation (outline only, later phase)

Sleep-cycle analogue: periodically fine-tune a small adapter on accumulated stable cloud summaries so consolidated knowledge migrates from context into weights; measure how many pyramid tokens become unnecessary post-consolidation (same QA accuracy at smaller live context). Stack: PEFT/QLoRA on the 3B base, adapters only — realistically a rented GPU hour rather than a CPU run for this one. Not specified further until E2 passes and E4 shows where context pressure actually binds.

---

## Decision tree for today

```
E0 rank streams differ across runs → pin n_threads lower (last resort: 1 for scoring); do not proceed until identical
E0 shows decode < ~8 t/s → E2 summaries will take ~30+ min; start them first and build score_levels.py while they run
E1 shows LM codec worse than zstd → check block scheme / tokenizer mismatch first; not fatal
E1 shows spike at N2 → surprising! token surprisal richer than theorized; E2 still runs
E2 PASSES → the core figure exists; proceed E3/E4; start writeup skeleton
E2 FAILS on N2 adjacency → inspect whether cap-frame and exception share an L2 scoring window; fix windows, rerun once
E2 FAILS structurally → stop building; the scale-space claim needs reformulation
```

## Notes for Claude Code sessions

- Never call cloud APIs; everything local. Never upgrade the pinned model mid-experiment.
- Every results JSON embeds `env.json` fingerprint; refuse to compare results across differing fingerprints.
- Prefer boring, verifiable code over clever code: this is measurement instrumentation, not a product.
- Plots to PNG in results/; no interactive display needed.
