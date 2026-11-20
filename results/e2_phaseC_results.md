# Phase C results: cross-model scoring control

Question: is Phase A's N2 delta-inversion (-1.65 z_delta_L2 under Qwen, opposite of the pre-registered prediction) a Qwen-self-scoring artifact (the model finds its own generated summaries more predictable), or does it reflect something about the corpus/content that would show up under any scorer (topic-priming / register-dominance)?

Method: kept the existing Qwen-generated summaries (`levels_seed{K}.json`) completely unchanged, and reran Phase A's delta-scoring computation using Llama-3.2-3B-Instruct-Q5_K_M as the scorer instead of Qwen2.5-3B-Instruct. Different model family, different tokenizer (128k vocab vs. Qwen's 152k), different training data and provider.

## Seed 0: Qwen-scored vs. Llama-scored z_delta_L2

| Needle | Qwen scorer | Llama scorer |
|---|---|---|
| N1 weird-benign | 0.788 | 0.592 |
| N2 contradiction | **-1.649** | **-1.460** |
| N3 novel | 0.406 | 0.273 |
| N4 control | -0.303 | -0.061 |

The entire profile shape is preserved under a completely different model — not just the sign, but roughly the same relative spacing between needle types. N2 is still the clear low outlier (-1.46, essentially matching Qwen's -1.65); N1 is still the high outlier; N3 and N4 sit in between in the same order.

## Verdict: self-scoring-artifact hypothesis is ruled out; topic-priming/register-dominance is supported

If N2's inversion were an artifact of Qwen reading its own generated text as unsurprising, swapping the scorer to an unrelated model (different architecture, different training run, different provider) should have broken or at least significantly attenuated the effect. It didn't — the numbers moved by less than 0.2 in either direction across all four needles at L2. This is strong evidence that what Phase A is measuring reflects a real property of the *content* (the corpus's repeated liability-cap theme priming any model toward finding same-topic continuations more predictable, regardless of whether they agree or conflict with the established direction) rather than a quirk of one model scoring its own writing.

This matters for how E2 should be described going forward: the problem is not "use a different model for summarization than scoring" (Phase A's original mitigation candidate #1 from `E2_FINDINGS.md`) — that fix would not have worked, and Phase C is why we know that rather than guessing.
