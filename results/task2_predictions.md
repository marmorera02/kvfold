# Task 2 pre-registration (written before running the open-world seeds)

Threat being tested: Phase D's 11/12 may be inflated by the closed-world guarantee of exactly one contradiction per seed. Seeds 3/4 (zero contradictions) and seed 5 (two contradictions) test whether the pipeline knows when to say "none" and can find more than one.

Configs evaluated: (ii) delta-only, (iii) combined, (iv) delta-routed NLI — using Phase D's frozen rules (bottom-15th-percentile-by-z_delta_L2 = contradiction-candidate section; z_L0≥1 & |z_delta_L2|<0.5 = novel; z_L0≥1 & |z_delta_L2|≥0.5 = weird-benign; else boilerplate). **No threshold retuning.**

- **P-G**: Delta-only FAILS open-world by construction — it will flag a "contradiction" in seeds 3/4 (there is always a lowest-ranked section, whether or not anything is actually wrong there) and will find at most one contradiction in seed 5 (the rule as built has no mechanism to flag a second, non-extreme section). This is the expected, documented reason delta alone must never be the verdict — only NLI confirmation should determine an actual contradiction call.
- **P-H**: Combined and delta-routed configs report ZERO contradictions on seeds 3/4 (the NLI gate, checked against the assumption set with the referent guard, must not confirm any flagged candidate as a genuine contradiction). Allowed slack: at most one false contradiction across both zero-seeds combined.
- **P-I**: Seed 5 recall — both N2a and N2b caught by config (iv). Specifically checking whether N2b's section falls inside the bottom-15th-percentile routing set; if routing misses it, report its actual delta rank and compute a recall-vs-cost curve at 15th/25th/40th percentile cutoffs.

**Gate**: P-H and P-I hold → E3/E4 unlocked, pipeline graduates to "locator + verifier, open-world validated." P-H fails → pipeline is a candidate-ranker only; report false-positive anatomy (which premise/hypothesis pairs confirmed falsely) and stop for owner decision before E3/E4.
