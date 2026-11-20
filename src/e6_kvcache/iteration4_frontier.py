"""Cycle 3 Iteration 4 (E4): the speed-fidelity frontier of condensation,
across all regimes validated so far. Reuses Iteration 3's R1 (full context)
and R2/R3 (KV-reload unfold) results rather than rerunning them; adds (b)
skeleton+text-unfold and (c) coarse-only as new regimes.
"""
import json

from src.common import io as io_mod
from src.common import model as model_mod
from src.e3_folding.run import _load_boilerplate_text
from src.e6_kvcache.iteration3_qa import (
    DATA_DIR, MAX_ANSWER_TOKENS, QA_PATH, _greedy_continue, build_skeleton, check_answer,
)
import time


def run_fresh(llm, context_text: str, question: str):
    prompt = f"Context: {context_text}\n\nQuestion: {question}\nAnswer:"
    t0 = time.perf_counter()
    llm.reset()
    ctx_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    llm.eval(ctx_tokens)
    answer_tokens = _greedy_continue(llm, len(ctx_tokens), MAX_ANSWER_TOKENS)
    elapsed = time.perf_counter() - t0
    return llm.detokenize(answer_tokens).decode("utf-8", errors="ignore"), elapsed, len(ctx_tokens)


def main() -> None:
    llm = model_mod.load_model(n_ctx=4096, logits_all=True)
    full_text = _load_boilerplate_text()
    qa = json.loads(QA_PATH.read_text())["questions"]

    print("rebuilding 2b skeleton...")
    live_text, folded_text, fold_fraction = build_skeleton(llm, full_text)
    stitched_text = live_text + "\n\n" + folded_text  # (b): same content as R2, plain fresh prefill

    full_tokens_n = len(llm.tokenize(full_text.encode("utf-8"), add_bos=True))
    live_tokens_n = len(llm.tokenize(live_text.encode("utf-8"), add_bos=True))

    results_b, results_c = [], []
    for q in qa:
        ans_b, lat_b, ntok_b = run_fresh(llm, stitched_text, q["question"])
        correct_b = check_answer(ans_b, q["gold_key_facts"])
        results_b.append({"id": q["id"], "answer": ans_b, "correct": correct_b,
                           "latency_s": lat_b, "live_tokens": ntok_b})

        ans_c, lat_c, ntok_c = run_fresh(llm, live_text, q["question"])
        correct_c = check_answer(ans_c, q["gold_key_facts"])
        results_c.append({"id": q["id"], "answer": ans_c, "correct": correct_c,
                           "latency_s": lat_c, "live_tokens": ntok_c})
        print(f"  Q{q['id']}: (b)={'OK' if correct_b else 'MISS'} (c)={'OK' if correct_c else 'MISS'}")

    n = len(qa)
    b_acc = sum(r["correct"] for r in results_b) / n
    c_acc = sum(r["correct"] for r in results_c) / n
    b_lat = sum(r["latency_s"] for r in results_b) / n
    c_lat = sum(r["latency_s"] for r in results_c) / n

    # Reuse Iteration 3's R1/R2/R3 (not rerun)
    i3 = json.loads((io_mod.RESULTS_DIR / "cycle3_iteration3_run.json").read_text())

    frontier = {
        "a_full_context": {
            "accuracy": i3["R1_accuracy"], "mean_latency_s": i3["R1_mean_latency_s"],
            "live_tokens": full_tokens_n, "live_token_fraction": 1.0,
        },
        "b_skeleton_text_unfold": {
            "accuracy": b_acc, "mean_latency_s": b_lat,
            "live_tokens": int(sum(r["live_tokens"] for r in results_b) / n),
            "live_token_fraction": float(sum(r["live_tokens"] for r in results_b) / n) / full_tokens_n,
        },
        "c_coarse_only": {
            "accuracy": c_acc, "mean_latency_s": c_lat,
            "live_tokens": int(sum(r["live_tokens"] for r in results_c) / n),
            "live_token_fraction": float(sum(r["live_tokens"] for r in results_c) / n) / full_tokens_n,
        },
        "d_skeleton_kv_reload_R2": {
            "accuracy": i3["R2_accuracy"], "mean_latency_s": i3["R2_mean_latency_s"],
            "live_tokens": live_tokens_n,  # only the freshly-appended folded content is "live" prefill beyond the reload
            "live_token_fraction": (1 - fold_fraction),
        },
        "d_skeleton_kv_reload_R3": {
            "accuracy": i3["R3_accuracy"], "mean_latency_s": i3["R3_mean_latency_s"],
            "live_tokens": live_tokens_n,
            "live_token_fraction": (1 - fold_fraction),
        },
    }

    p_u_holds = False
    p_u_regime = None
    for name, row in frontier.items():
        if name == "a_full_context":
            continue
        if row["accuracy"] >= frontier["a_full_context"]["accuracy"] and row["live_token_fraction"] < 0.30:
            p_u_holds = True
            p_u_regime = name
            break

    summary = {
        "fold_fraction": fold_fraction,
        "full_text_tokens": full_tokens_n,
        "frontier": frontier,
        "P_U_holds": p_u_holds,
        "P_U_matching_regime": p_u_regime,
        "results_b_per_question": results_b,
        "results_c_per_question": results_c,
    }
    out_path = io_mod.save_result("cycle3_iteration4_run.json", summary)

    print("\n=== Frontier table ===")
    for name, row in frontier.items():
        print(f"  {name}: accuracy={row['accuracy']:.2f} latency={row['mean_latency_s']:.2f}s "
              f"live_tokens~{row['live_tokens']} ({row['live_token_fraction']:.1%} of full)")
    print(f"\nP_U_holds: {p_u_holds} (regime: {p_u_regime})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
