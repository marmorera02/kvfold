"""E11 -- naked-vs-enhanced crossover, 3B vs 7B (owner request, delivered
alongside Cycle 3 / E10).

Question: given the enhanced method's one-time setup cost (real
fold-detection: atomize + score + watershed + verify-transaction, i.e. the
actual "surprise-gated" analysis -- NOT just a plain prefill) is larger
than a single naked prefill, after how many consecutive prompts against
the SAME document does the enhanced method's cheaper per-query cost
(KV-reload + shorter remainder prefill + decode) pay back that upfront
cost, relative to the naked method (fresh full-context prefill + decode,
every single time, no reuse)?

Reuses existing Cycle 3 numbers where already measured:
- 3B fold fraction, full pipeline validity (P-T/P-O), 10-question accuracy
  and full-sample latencies: cycle3_iteration{2b,3}_run.json.
This script adds what wasn't previously isolated: the fold-detection
setup cost measured on its own (not bundled into a QA latency number), and
a prefill/decode SPLIT (Iteration 3's R1/R2 timers measured combined
latency only) -- for both 3B (re-timed fresh, small question sample, for
an apples-to-apples split against 7B) and 7B (new model, not previously
run in this project).

Deliberately scoped as "a very short sample test" (owner's words): 3 of
the 10 frozen QA questions, not all 10. The one-time fold-detection cost
is genuinely large (real analysis work, not a formality) and is measured
faithfully, in full, for both model sizes -- that part is not shortened.

Uses the FAST fold-detection pipeline (verify_2b_fast.py,
results/verify_2b_fast_results.md) rather than Iteration 2b/3's original
cost-ascending run_transaction_2b: the original took 60+ minutes on this
document and never finished during initial testing for this experiment,
which is exactly the "ingest algorithm is far too compute expensive" issue
that prompted the optimization. The fast path is document-order rather
than cost-ascending, so its fold fraction differs from Iterations 2b-4's
own recorded numbers (disclosed, not hidden, in that results file) --
this experiment's 3B numbers are therefore fresh measurements against the
fast pipeline, not a reuse of Iteration 3/4's exact prior figures.
"""
import argparse
import json
import time

from src.common import io as io_mod
from src.common import model as model_mod
from src.e3_folding.run import _load_boilerplate_text
from src.e3_folding.run_2b import _derive_thresholds, _document_data
from src.e3_folding.verify_2b_fast import run_transaction_2b_fast
from src.e3_folding.watershed import build_proposals
from src.e6_kvcache.iteration3_qa import MAX_ANSWER_TOKENS, QA_PATH, _greedy_continue, check_answer


def build_skeleton_fast(llm, text: str):
    """Fast-pipeline equivalent of iteration3_qa.build_skeleton: same
    inputs/outputs (live_text, folded_text, fold_fraction), but via
    verify_2b_fast's document-order, incrementally-cached transaction
    instead of the original's cost-ascending, full-reprefill-per-candidate
    one. See verify_2b_fast_results.md for why and the resulting
    fold-fraction difference."""
    data = _document_data(llm, text)
    atoms = data["atoms"]
    theta_fold, theta_flag = _derive_thresholds(data["per_token_costs"])
    proposals, is_anchor, z = build_proposals(atoms, data["atom_surprisals"])
    committed, folded = run_transaction_2b_fast(
        llm, atoms, is_anchor, data["rank_probs"], llm.n_vocab(), theta_fold, theta_flag,
    )
    live_text = "".join(atoms[i]["text"] for i in range(len(atoms)) if not folded[i])
    folded_text = "".join(atoms[i]["text"] for i in range(len(atoms)) if folded[i])
    fold_fraction = sum(c["n_tokens"] for c in committed) / data["total_tokens"]
    return live_text, folded_text, fold_fraction

N_SAMPLE_QUESTIONS = 3


def timed_naked(llm, full_text: str, question: str) -> dict:
    prompt = f"Context: {full_text}\n\nQuestion: {question}\nAnswer:"
    llm.reset()
    t0 = time.perf_counter()
    ctx_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    llm.eval(ctx_tokens)
    t_prefill = time.perf_counter() - t0

    t1 = time.perf_counter()
    answer_tokens = _greedy_continue(llm, len(ctx_tokens), MAX_ANSWER_TOKENS)
    t_decode = time.perf_counter() - t1

    answer = llm.detokenize(answer_tokens).decode("utf-8", errors="ignore")
    return {
        "answer": answer, "t_prefill": t_prefill, "t_decode": t_decode,
        "decode_tps": (MAX_ANSWER_TOKENS / t_decode) if t_decode > 0 else None,
        "ctx_tokens": len(ctx_tokens),
    }


def timed_enhanced(llm, skeleton_state, skeleton_len: int, folded_text: str, question: str) -> dict:
    t0 = time.perf_counter()
    llm.load_state(skeleton_state)
    t_reload = time.perf_counter() - t0

    tail = f"{folded_text}\n\nQuestion: {question}\nAnswer:"
    t1 = time.perf_counter()
    tail_tokens = llm.tokenize(tail.encode("utf-8"), add_bos=False)
    llm.eval(tail_tokens)
    t_prefill_remainder = time.perf_counter() - t1
    total = skeleton_len + len(tail_tokens)

    t2 = time.perf_counter()
    answer_tokens = _greedy_continue(llm, total, MAX_ANSWER_TOKENS)
    t_decode = time.perf_counter() - t2

    answer = llm.detokenize(answer_tokens).decode("utf-8", errors="ignore")
    return {
        "answer": answer, "t_reload": t_reload, "t_prefill_remainder": t_prefill_remainder,
        "t_ingest_total": t_reload + t_prefill_remainder, "t_decode": t_decode,
        "decode_tps": (MAX_ANSWER_TOKENS / t_decode) if t_decode > 0 else None,
        "remainder_tokens": len(tail_tokens), "total_tokens": total,
    }


def run_for_model(model_label: str) -> dict:
    llm = model_mod.load_model(n_ctx=4096, logits_all=True)
    full_text = _load_boilerplate_text()
    qa = json.loads(QA_PATH.read_text())["questions"][:N_SAMPLE_QUESTIONS]

    print(f"[{model_label}] running fold-detection (build_skeleton) -- this is the real one-time cost...")
    t0 = time.perf_counter()
    live_text, folded_text, fold_fraction = build_skeleton_fast(llm, full_text)
    t_fold_detection = time.perf_counter() - t0
    print(f"[{model_label}] fold-detection: {t_fold_detection:.1f}s, fold_fraction={fold_fraction:.3f}")

    skeleton_tokens = llm.tokenize(live_text.encode("utf-8"), add_bos=True)
    t0 = time.perf_counter()
    llm.reset()
    llm.eval(skeleton_tokens)
    skeleton_state = llm.save_state()
    t_skeleton_prefill_and_save = time.perf_counter() - t0
    skeleton_len = len(skeleton_tokens)
    t_setup_enhanced = t_fold_detection + t_skeleton_prefill_and_save
    print(f"[{model_label}] skeleton prefill+save: {t_skeleton_prefill_and_save:.2f}s -- "
          f"total one-time enhanced setup cost: {t_setup_enhanced:.1f}s")

    naked_results, enhanced_results = [], []
    for q in qa:
        nr = timed_naked(llm, full_text, q["question"])
        nr["correct"] = check_answer(nr["answer"], q["gold_key_facts"])
        naked_results.append(nr)

        er = timed_enhanced(llm, skeleton_state, skeleton_len, folded_text, q["question"])
        er["correct"] = check_answer(er["answer"], q["gold_key_facts"])
        enhanced_results.append(er)

        print(f"[{model_label}] Q{q['id']}: "
              f"naked={'OK' if nr['correct'] else 'MISS'} (ingest={nr['t_prefill']:.2f}s decode={nr['t_decode']:.2f}s) | "
              f"enhanced={'OK' if er['correct'] else 'MISS'} (ingest={er['t_ingest_total']:.2f}s decode={er['t_decode']:.2f}s)")

    n = len(qa)
    mean_naked_ingest = sum(r["t_prefill"] for r in naked_results) / n
    mean_naked_decode = sum(r["t_decode"] for r in naked_results) / n
    mean_naked_tps = sum(r["decode_tps"] for r in naked_results) / n
    mean_enhanced_ingest = sum(r["t_ingest_total"] for r in enhanced_results) / n
    mean_enhanced_decode = sum(r["t_decode"] for r in enhanced_results) / n
    mean_enhanced_tps = sum(r["decode_tps"] for r in enhanced_results) / n

    per_query_savings = (mean_naked_ingest + mean_naked_decode) - (mean_enhanced_ingest + mean_enhanced_decode)
    crossover_n = (t_setup_enhanced / per_query_savings) if per_query_savings and per_query_savings > 0 else None

    return {
        "model_label": model_label,
        "model_filename": model_mod.MODEL_FILENAME,
        "n_sample_questions": n,
        "fold_fraction": fold_fraction,
        "t_fold_detection_s": t_fold_detection,
        "t_skeleton_prefill_and_save_s": t_skeleton_prefill_and_save,
        "t_setup_enhanced_total_s": t_setup_enhanced,
        "mean_naked_ingest_s": mean_naked_ingest,
        "mean_naked_decode_s": mean_naked_decode,
        "mean_naked_decode_tps": mean_naked_tps,
        "mean_enhanced_ingest_s": mean_enhanced_ingest,
        "mean_enhanced_decode_s": mean_enhanced_decode,
        "mean_enhanced_decode_tps": mean_enhanced_tps,
        "per_query_savings_s": per_query_savings,
        "crossover_n_queries": crossover_n,
        "naked_accuracy": sum(r["correct"] for r in naked_results) / n,
        "enhanced_accuracy": sum(r["correct"] for r in enhanced_results) / n,
        "naked_results": naked_results,
        "enhanced_results": enhanced_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="e.g. 3B or 7B, for output filenames")
    args = parser.parse_args()
    summary = run_for_model(args.label)
    out_path = io_mod.save_result(f"e11_7b_frontier_{args.label}.json", summary)

    print(f"\n=== {args.label} summary ===")
    print(f"fold_fraction: {summary['fold_fraction']:.3f}")
    print(f"one-time enhanced setup cost: {summary['t_setup_enhanced_total_s']:.1f}s "
          f"(fold-detection {summary['t_fold_detection_s']:.1f}s + "
          f"skeleton prefill/save {summary['t_skeleton_prefill_and_save_s']:.2f}s)")
    print(f"naked:    ingest={summary['mean_naked_ingest_s']:.2f}s  decode={summary['mean_naked_decode_s']:.2f}s "
          f"({summary['mean_naked_decode_tps']:.1f} tok/s)  accuracy={summary['naked_accuracy']:.2f}")
    print(f"enhanced: ingest={summary['mean_enhanced_ingest_s']:.2f}s  decode={summary['mean_enhanced_decode_s']:.2f}s "
          f"({summary['mean_enhanced_decode_tps']:.1f} tok/s)  accuracy={summary['enhanced_accuracy']:.2f}")
    print(f"per-query savings: {summary['per_query_savings_s']:.2f}s")
    print(f"crossover (queries needed to break even): {summary['crossover_n_queries']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
