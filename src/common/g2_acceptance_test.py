"""G2 acceptance test (compute-migration directive): stability gate for GPU
generation. 5 fixed prompts, each sent twice (fresh requests) to the host
Vulkan llama-server; asserts byte-identical output both times. If ANY pair
differs, generation stays on CPU and the migration stops there.

Also runs the same 5 prompts through local CPU generation for an
informational (non-gating) cross-backend diff -- differences here are
EXPECTED and not a failure; logged only.
"""
import json

from src.common import genserver
from src.common import model as model_mod

FIXED_PROMPTS = [
    "Summarize this contract clause in one sentence. Preserve any negations, exceptions, caps, amounts, and party obligations exactly. Clause: Supplier's aggregate liability under this Agreement shall be capped at 100% of the annual Contract Value; in no event shall Supplier be liable for consequential, incidental, or indirect damages of any kind. Summary:",
    "Summarize this contract clause in one sentence. Preserve any negations, exceptions, caps, amounts, and party obligations exactly. Clause: Buyer shall pay all undisputed invoices within sixty (60) days of the invoice date, by wire transfer to the bank account designated by Supplier in writing. Summary:",
    "Summarize these clause summaries in 2-3 sentences, preserving all exceptions, caps, and contradictions: The delivery terms require notice within 14 days. The warranty period is 90 days from delivery. Payment is due within 60 days of invoice.",
    "If this clause summary states a general rule, cap, exclusion, or obligation that applies beyond a single transaction, restate it as one declarative sentence. If it is a routine transactional detail, output NONE. Summary: Buyer shall pay invoices within 60 days of receipt.",
    "Continue the following story in the same voice, for a few sentences.\n\nThe lighthouse keeper opened his second notebook and wrote:",
]
MAX_TOKENS = 80


def run() -> dict:
    results = {"prompts": []}
    all_stable = True

    for i, prompt in enumerate(FIXED_PROMPTS):
        gpu_1 = genserver.generate(prompt, MAX_TOKENS)
        gpu_2 = genserver.generate(prompt, MAX_TOKENS)
        stable = gpu_1 == gpu_2
        all_stable = all_stable and stable

        results["prompts"].append({
            "index": i,
            "gpu_run1": gpu_1,
            "gpu_run2": gpu_2,
            "gpu_stable": stable,
        })
        print(f"prompt {i}: GPU stable={stable}")

    results["all_gpu_stable"] = all_stable

    # Informational cross-backend diff (not a gate)
    llm = model_mod.load_model(logits_all=False, n_ctx=2048)
    for i, prompt in enumerate(FIXED_PROMPTS):
        cpu_out = llm(
            prompt, max_tokens=MAX_TOKENS,
            temperature=model_mod.TEMPERATURE, top_k=model_mod.TOP_K, seed=model_mod.SEED,
        )["choices"][0]["text"]
        gpu_out = results["prompts"][i]["gpu_run1"]
        results["prompts"][i]["cpu_output"] = cpu_out
        results["prompts"][i]["cpu_gpu_match"] = (cpu_out == gpu_out)
        print(f"prompt {i}: CPU==GPU (informational only) = {cpu_out == gpu_out}")

    return results


if __name__ == "__main__":
    payload = run()
    with open("/workspace/results/g2_acceptance_test.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nALL GPU STABLE: {payload['all_gpu_stable']}")
