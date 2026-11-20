# kvfold

Surprise-gated folding and KV-state reload for long documents using surprisal to flag fluent contradictions.

A long document is summarised into a pyramid (clauses, one-sentence summaries, section summaries) and a language model
scores the surprisal of every level. Surprisal is used in two ways: as a salience score, to locate clauses that
contradict an earlier rule while reading as fluent text, and as a storage cost (−log₂ p bits per token), to decide which
spans can be folded out of the live context and later reconstructed exactly. Folded spans keep their KV state on disk and
are reloaded instead of re-read.

Everything runs on one CPU laptop, in a container, with greedy sampling and pinned threading. Models are not included.

## Method

**Scale-space surprisal.** Each planted clause type has a predicted profile across levels (`spec/PYRAMID_EXPERIMENTS_SPEC.md`, E2):

| Planted clause | L0 surprisal | L2 surprisal | NLI contradiction |
|---|---|---|---|
| N4 ordinary boilerplate (control) | low | low | low |
| N1 weird-but-benign (typo, odd aside) | high | low | low |
| N2 fluent contradiction of an early liability cap | low | high | high |
| N3 genuinely novel (off-topic but coherent) | high | high | low |

**Verified folding.** A fold is a transaction: propose a run of low-surprisal tokens, verify that the model reconstructs
it exactly from the remaining context (stored as rank residuals), then commit it. Negations, exceptions, numbers and
names are gated and never fold. A three-bin policy (silent / flagged / live) is derived from the document's own
surprisal statistics. KV state is saved before eviction and reloaded on unfold.

## Setup

| Item | Setting |
|---|---|
| Scorers | Qwen2.5-3B-Instruct Q5_K_M (primary); Llama-3.2-3B-Instruct Q5_K_M (cross-model control); Qwen2.5-7B-Instruct Q5_K_M in the speed run |
| Runtime | llama.cpp via llama-cpp-python 0.3.9, CPU only (Ryzen 7 PRO 8840U, 8 pinned threads), `seed=0`, `temperature=0`, `top_k=1`, `n_batch=512`, model SHA-256 recorded in `results/env.json` |
| Corpus | Template-generated supply agreements, 100 clauses, seeds 0–5. N1/N3/N4 at clauses 33/61/45, N2 at clause 87; liability cap stated at clauses 8, 25, 50. Seeds 3 and 4 contain no contradiction, seed 5 contains two (one reinforced frame, one stated once) |
| Real prose | Three Wikipedia articles (Eiffel Tower, Golden Gate Bridge, Titanic) with an authored fact restated three times plus one contradiction (`wikiA`) or none (`wikiB`) |
| Pyramid | L0 clauses → L1 one-sentence summaries → L2 summaries of ten L1 units, generated greedily by Qwen2.5-3B-Instruct |
| Scoring | mean −log₂ p per token of each unit given all preceding units at its level, z-scored within level. Conditioning delta Δ = S(u \| context) − S(u \| ∅) |
| NLI | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, CPU |
| Protocol | predictions written to a file before each run; thresholds fixed on seed 0 and frozen; pass criteria fixed in advance |

Each result below points at a file in `results/`. Experiments are named as in the spec (E0–E11); Phases A–E are the
follow-up passes after the first gate.

## Results: surprisal profiles and contradiction flagging

### E1: token-level surprisal (seed 0)

Local mean surprisal over the planted clauses, bits per token (document median 0.52):

| N1 weird-but-benign | N2 contradiction | N3 novel | N4 control |
|---|---|---|---|
| 3.57 | 2.61 | 3.62 | 2.16 |

### E2: four-needle experiment (`results/E2_FINDINGS.md`)

Fixed-threshold classifier: 1/4 correct on each of three seeds. N2 profile:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| z_L0 | 1.47 | 1.35 | 1.36 |
| z_L2 | −1.63 | −1.58 | −0.67 |
| max NLI contradiction | 0.99 | 0.99 | 0.99 |

65–67% of ordinary clauses score ≥ 0.5 NLI contradiction against their top-5 neighbours (16 templates differing in
numbers and product names).

### Phases A–D (`results/e2_phase{A,B,C,D}_results.md`)

- N2's z_Δ at L2: −1.65, −0.95, −0.83 (seeds 0–2).
- Rescoring the unchanged summaries with Llama-3.2-3B moves every needle's z by under 0.2; N2 goes from −1.65 to −1.46.
- Assumption-set NLI (each clause compared only with extracted standing rules that share a referent): NLI
  flagged-pair rate about 65% → 8–15%, recall 0.989.
- Routing NLI to the lowest-Δ section covers the contradiction at about 20% of the NLI calls. Four-way classification over
  three seeds: NLI alone 6/12, Δ alone 11/12, combined 11/12, Δ-routed NLI 11/12 (one contradiction per document).
  AUC of Δ for novel vs boilerplate: 0.528.

### Documents with zero or two contradictions, and real prose

| Test | Result |
|---|---|
| Task 2, seeds 3/4 (no contradiction) | 3 contradictions reported (two force-majeure clauses, two IP clauses, two insolvency clauses) |
| Task 2, seed 5 (two contradictions) | n2a (frame restated 3×) reported; n2b (frame stated once) not reported: its section ranks 3–6 of 10 for routing and its frame was not extracted as a rule |
| E7: structured SCOPE/ATTRIBUTE/VALUE ledger + deterministic rule | extraction recall 11/12 including the single-mention frame; 71 and 127 conflict pairs reported on seeds 3 and 4 |
| Phase E, real prose, no contradiction (`wikiB`) | 0 contradictions reported |
| Phase E, real prose, one contradiction (`wikiA`) | section z_Δ_L2 = +0.63; raw NLI score against the matching premise 0.007 (max over 13 premises 0.024) |

### E10: minimal-pair contrast Δ± = S(u | A) − S(u | ¬A) (`results/e10_results.md`)

Premise A is contrasted with its negation ¬A so that topic priming is held constant (911 pairs, both scorers; a
hand-picked sign-flip check passed to 1e-6 before the full run).

| | Qwen | Llama |
|---|---|---|
| AUC, true contradictions vs all other pairs | 0.810 | 0.829 |
| True pairs with Δ± > 0 | 15/16 | 14/16 |
| The 3 pairs reported as contradictions in Task 2 have Δ± ≤ 0 | 3/3 | 3/3 |

### Computation shape (E1, `results/e1_seed0.json`)

On identical context, model and seed, batched prefill and single-token decode disagree on the top-1 token at 41 of 200
positions (20.5%), mean absolute logit difference 0.109, maximum 0.92 (Q5_K_M, CPU). All scoring passes use one batched
call. Repeat-determinism on a 4-slot Vulkan server: 2 of 5 prompts differed across identical requests
(`results/g2_results.md`).

## Results: folding and KV reload

(`results/cycle3_*`, `results/e7_*`, `results/e11_*`)

| Measurement | Result |
|---|---|
| Rank-stream codec (E1, seed 0, 17,801 tokens) | 2.23 bits/token vs zstd-19 3.22 and gzip-9 4.04; 67.4% of tokens rank 0 |
| KV save/restore | 100-token greedy continuation byte-identical in a fresh process. Reload vs re-prefill: 380.7× at 4k tokens, 1198× at 16k |
| Fold transaction | 10/10 sampled folds reconstruct byte-exactly; 0 gated atoms folded. Fold fraction 38.3% (boilerplate) and 43.1% (prose) |
| Fold fraction, predicted vs measured | 63.9% → 38.3% and 69.8% → 43.1% (verification runs against the already-thinned context) |
| QA on the assembled system (10 questions) | gold 10/10; skeleton + KV reload 9/10 at 48.1% of gold latency; with a 64-token seam recompute 9/10 at 51.5% |
| E11, naked vs enhanced, 3 questions | 3B: enhanced 1/3, naked 3/3, break-even about 9.9 queries. 7B: enhanced 3/3, naked 3/3, break-even about 8.2 queries (one-time fold cost 265 s and 499 s) |

KV state size fits `311,776,005 + 36,880 × n_tokens` bytes; the per-token term is the Qwen2.5-3B KV size
(36 layers × 2 × 2 KV heads × 128 × 2 bytes = 36,864 bytes, plus 16).

## Reproducing

```bash
docker compose build
# download the GGUF models into ./models (see THIRD_PARTY.md), then for example:
docker compose run --rm pyramid -m src.e0_sanity
docker compose run --rm pyramid -m src.e1_rankcodec --help
docker compose run --rm pyramid -m src.e2_fourneedle.gen_corpus --seed 0
```

Result locations and environment come from `PYRAMID_RESULTS_DIR` and `PYRAMID_MODELS_DIR` (default `/workspace/...`).
Every result JSON embeds the model checksum, thread count and library versions. Corpora are deterministic given the seed
and are included in `data/`.

Not included: the model files; the KV-state pickles (0.3–1.5 GB each); the raw E10 pair files and all outputs derived
from Wikipedia text other than the three source articles (they are regenerated by the scripts).

## Layout

| Path | Contents |
|---|---|
| `spec/` | the original experiment specification (written as "Surprisal Pyramid", the project's first name) |
| `src/common/` | model loader with determinism pins, scoring primitive, result I/O |
| `src/e0_sanity.py`, `src/e1_rankcodec.py` | determinism check; rank-stream codec and token-level baseline |
| `src/e2_fourneedle/` | corpus generators, pyramid builder, scoring, NLI, assumption sets, delta scoring, classifier, plots |
| `src/e3_folding/` | atomizer, watershed proposals, verified fold transaction, fast linear variant |
| `src/e6_kvcache/` | KV save/restore, reload speed, QA iterations 3–4 |
| `src/e7_ledger/` | structured ledger extraction and deterministic conflict rule |
| `src/e10_contrast/` | minimal-pair construction, negation, contrast scoring |
| `src/e11_7b_frontier/` | naked versus enhanced crossover at 3B and 7B |
| `results/` | `*_predictions.md` (written before runs), `*_results.md`, numeric JSON, plots, `runtime_log.csv` |
| `analyze_phaseD.py` | post-hoc analysis of Phase D from stored JSON |

## Provenance

The experiments were run and written up in an AI-assisted workflow: the predictions and results reports were drafted
with Claude Code under written directives, and the commits carry a co-author line.

## License

Code: PolyForm Noncommercial License 1.0.0 (`LICENSE.md`, `NOTICE.txt`). Models and the Wikipedia text keep their own
terms (`THIRD_PARTY.md`).
