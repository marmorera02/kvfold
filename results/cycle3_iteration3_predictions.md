# Cycle 3 — Iteration 3 pre-registration (written before running)

**Question**: when unfolding means reloading frozen KV of folded regions into a live context built by the REAL fold policy (Iteration 2b's skeleton), how much answer quality survives relative to gold?

**Implementation note** (owner-approved simplification): llama.cpp's KV save/load (validated in Iteration 1) operates on whole-sequence state tied to absolute positions — there's no exposed mechanism to splice a saved span's KV state into a different position in a different sequence with correct RoPE encoding. R2/R3 are implemented as: reload the live-skeleton's KV state (the 2b skeleton with folded spans removed, i.e. only "live"/anchor atoms, in original order), then APPEND the folded spans' reconstructed original text at the current sequence end (not their true original position) before the question. This is genuinely naive/positionally-wrong — that mismatch is what's being tested, not hidden. R3 additionally re-prefills (fresh, not reloaded) the last 64 tokens of the skeleton together with the appended content, giving the model a correctly-computed local window right at the seam.

Regimes:
- **R1** (gold): full original text + question, fresh full re-prefill.
- **R2** (naive stitch): reload full-skeleton KV state, append reconstructed folded content + question.
- **R3** (stitch + seam recompute): reload skeleton-minus-last-64-tokens KV state, freshly re-eval those 64 tokens + appended folded content + question.

QA: 10 questions with gold key-facts, `data/e6_qa_set.json`, frozen after this iteration (Iteration 4 reuses it unchanged).

## P-O
R1 ≥ R3 ≥ R2 in QA accuracy (gold-key-fact containment, case-insensitive substring); R3 within 1 question of R1.

## P-P
R2/R3 latency < 25% of R1's latency.

## Gate
- P-O holds → three-tier architecture validated end to end at prototype scale; Iteration 4 measures its frontier.
- R3 materially below R1 → seam repair insufficient at 3B training-free; escalate seam width (128, 256 tokens) once each, logged. If still failing, the honest conclusion is that training-free KV stitching costs real accuracy — quantify the cost, proceed to Iteration 4 anyway with both text-only and KV regimes so the frontier shows the trade explicitly.

Failure anatomy (degenerate generations, seam artifacts) will be reported with quotes and counts regardless of outcome.
