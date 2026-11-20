# Task 1: N1/L1 autopsy

## Finding: the orthographic anomaly is already normalized — flag per directive branch 3

L0 text (identical across all seeds, since N1_TEXT is a fixed constant in `gen_corpus.py`):
> "All notices under this Agreement shall be sent by **regustered** mail to the address..."

L1 summary (identical across all seeds — same input text, greedy decoding):
> "All notices under this Agreement shall be sent by **registered** mail to the address of the receiving party set out in Schedule 1 or by mutual agreement, with notices by facsimile deemed effective only upon confirmed transmission and a courtesy copy dispatched by ordinary post within two Business Days, notwithstanding informal practices of certain regional offices."

The typo is corrected in the summary, in every seed. Per the directive's branch 3 ("if NO: flag for owner review"), this is exactly that case: the anomaly is absent from the summary, yet z_L1(N1) stays at ~2.7 (matching z_L0's ~2.3–2.4) in all three seeds. **Flagging as instructed rather than proceeding with the normalization rerun**, since that experiment only makes sense if the anomaly were still present verbatim.

## My best diagnostic read (not a rerun, just inspection of the existing text)

N1's design wasn't only the misspelling — the clause also contains "an unusual but harmless phrasing" (per the original corpus spec): a description of regional offices informally telephoning notice recipients. That substantive oddity survives into the L1 summary intact: *"notwithstanding informal practices of certain regional offices."* Discussing informal telephone customs is itself unusual, low-frequency content for a legal-clause summary — independent of spelling. The summary is also structurally a single long run-on sentence stitching together several distinct ideas (notice method, fax effectiveness, courtesy-copy timing, regional customs), which is plausibly harder to predict than a more uniformly-structured summary regardless of content.

So my working hypothesis: **z_L1 stays elevated because the substantive weirdness (the telephone-custom aside), not the spelling, is what's actually driving surprisal — and summarization only ever normalized the orthographic layer, not the substantive one.** This would mean the pyramid's "L1 folds away benign surface weirdness" claim was never really about *this* needle's full content to begin with; it was only ever going to fold away the typo specifically.

This is a hypothesis, not a verified mechanism — I have not rerun scoring to isolate the two components (e.g., no clean way to test "elevated surprisal from odd content" vs "elevated surprisal from sentence complexity" without a controlled ablation, which would take more than the 30-minute budget). Flagging for your review rather than spending more time chasing it, per the directive's stop condition.

**Net effect on P-C**: stays refuted, but the refutation is now more precisely characterized — it's not that folding fails in general, it's that folding only removes the *specific* anomaly type it was asked to smooth (spelling), and this needle happened to bundle two anomaly types together (orthographic + substantive). Worth remembering when designing needles for Phase E: keep orthographic and substantive weirdness as separate, independently-testable needle variants rather than bundling them.
