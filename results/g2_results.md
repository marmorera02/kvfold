# G2: GPU generation acceptance test — FAILED, migration stops here

## Result

2 of 5 fixed prompts produced **different output on repeated identical requests** (same prompt, seed=0, temperature=0, top_k=1) against the host Vulkan `llama-server` (AMD Radeon 780M, `-ngl 999 --parallel 4`):

- **Prompt 2** (L2-style multi-clause summarization): completely different completions from the first sentence onward — "Returns must be made within 30 days of **receipt**... liable for damages up to $1000" vs. "...within 30 days of **purchase**... refund policy is 100% refund for defective products." Not a late token flip; the two completions diverge substantively and early.
- **Prompt 4** (open-ended narrative continuation): identical for the first 209 characters, then diverges — "The keeper's **wife**, a woman of strong character, stood by his side..." vs. "The keeper **took a deep breath**, his hands trembling slightly as he continued his meticulous record..." A real mid-generation fork, not trailing noise.

Prompts 0, 1, 3 (short, single-clause summarization / classification-style prompts) were stable across both runs.

Per the directive: **"if ANY pair differs, generation stays on CPU — record the failure and stop the migration here."** Done. GPU generation is not used for Task 2 or Phase E. All generation continues on CPU as before.

## Likely cause (diagnostic, not required to unblock, but worth recording)

The server was started with `--parallel 4` (4 concurrent request slots sharing one GPU context). This is exactly the kind of computation-shape sensitivity the E1 lesson already identified, one level up: batch/request composition on this hardware stack measurably changes numerics (E1 found ~20% rank-mismatch between single-token and batched-prefill paths on the *same* CPU backend). A multi-slot server is inherently exposed to the same class of problem, because slot scheduling and batching of concurrent requests can vary run-to-run depending on server-internal timing, even for what looks like an isolated, sequential test from the client side. The prompts that stayed stable were short and simple; the two that failed were longer, more complex generations — consistent with more opportunity for a mid-sequence numerical divergence to compound into a different greedy path.

Not chasing this further (e.g., retesting with `--parallel 1`) right now — the gate is unambiguous and the directive doesn't ask for a retry loop. If revisiting this later, `--parallel 1` (single in-flight request, no cross-request batching interference) would be the natural next thing to try, since it removes the suspected variable directly.

## Consequence for G1/G3

- G1 infrastructure (host Vulkan server, `genserver.py` client, SHA256-verified same GGUF) works technically — connectivity, correct model load, and real GPU engagement (100% GPU utilization during prefill, confirmed via live counter sampling) are all verified. It's the *determinism* requirement that fails, not the setup.
- **G3 (scoring migration) was not attempted.** It's gated on G1/G2 succeeding for generation first per the directive's own ordering, and separately the directive explicitly sanctions skipping G3 given the toolchain complexity of a Vulkan llama-cpp-python build inside the Linux container. Given G2 already failed, there's no basis to proceed to G3 regardless.
- **Net effect on the ongoing E2 program: none.** Task 2 (seeds 3/4/5) and Phase E continue exactly as planned, 100% CPU, matching everything already committed for seeds 0-3 and (in progress) seed 4. No mixed-backend risk was introduced — the failure was caught in an isolated acceptance test before any real corpus work touched the GPU path.
