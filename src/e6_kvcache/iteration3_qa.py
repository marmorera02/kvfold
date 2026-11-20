"""Cycle 3 Iteration 3 (E6.c, real skeleton): does the assembled system
preserve truth? Reloads frozen KV of the 2b skeleton into a live context
and measures QA accuracy relative to gold, per
`cycle3_iteration3_predictions.md`.

R2/R3 use an owner-approved simplification: llama.cpp's KV save/load
(validated in Iteration 1) operates on whole-sequence state tied to
absolute positions, with no exposed mechanism to splice a span's KV state
into a different position with correct RoPE encoding. So: reload the
live-skeleton's KV state, then APPEND the folded content's reconstructed
text at the current sequence end (not its true original position) before
the question -- genuinely naive/positionally-wrong, which is the point
being tested, not hidden. R3 additionally re-prefills (fresh) the last 64
skeleton tokens together with the appended content, giving the model a
correctly-computed local window right at the seam.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.common import io as io_mod
from src.common import model as model_mod
from src.e3_folding.run import _load_boilerplate_text
from src.e3_folding.run_2b import _derive_thresholds, _document_data
from src.e3_folding.verify_2b import run_transaction_2b
from src.e3_folding.watershed import build_proposals

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
QA_PATH = DATA_DIR / "e6_qa_set.json"
SEAM_TOKENS = 64
MAX_ANSWER_TOKENS = 60


def _greedy_continue(llm, total_so_far: int, n_new: int) -> list[int]:
    tokens = []
    total = total_so_far
    for _ in range(n_new):
        logits = np.asarray(llm.scores[total - 1])
        next_id = int(np.argmax(logits))
        tokens.append(next_id)
        llm.eval([next_id])
        total += 1
    return tokens


def build_skeleton(llm, text: str):
    """Reruns Iteration 2b's exact pipeline (deterministic, so this
    reproduces results/cycle3_iteration2b_run.json's boilerplate result
    identically) to recover which atoms are live vs. folded."""
    data = _document_data(llm, text)
    atoms = data["atoms"]
    theta_fold, theta_flag = _derive_thresholds(data["per_token_costs"])
    proposals, is_anchor, z = build_proposals(atoms, data["atom_surprisals"])
    committed, folded = run_transaction_2b(
        llm, atoms, proposals, data["rank_probs"], llm.n_vocab(), theta_fold, theta_flag,
    )
    live_text = "".join(atoms[i]["text"] for i in range(len(atoms)) if not folded[i])
    folded_text = "".join(atoms[i]["text"] for i in range(len(atoms)) if folded[i])
    fold_fraction = sum(c["n_tokens"] for c in committed) / data["total_tokens"]
    return live_text, folded_text, fold_fraction


def check_answer(answer: str, gold_key_facts: list[str]) -> bool:
    lower = answer.lower()
    return any(fact.lower() in lower for fact in gold_key_facts)


def run_r1(llm, full_text: str, question: str):
    prompt = f"Context: {full_text}\n\nQuestion: {question}\nAnswer:"
    t0 = time.perf_counter()
    llm.reset()
    ctx_tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
    llm.eval(ctx_tokens)
    answer_tokens = _greedy_continue(llm, len(ctx_tokens), MAX_ANSWER_TOKENS)
    elapsed = time.perf_counter() - t0
    return llm.detokenize(answer_tokens).decode("utf-8", errors="ignore"), elapsed


def run_r2(llm, skeleton_state, skeleton_len: int, folded_text: str, question: str):
    t0 = time.perf_counter()
    llm.load_state(skeleton_state)
    tail = f"{folded_text}\n\nQuestion: {question}\nAnswer:"
    tail_tokens = llm.tokenize(tail.encode("utf-8"), add_bos=False)
    llm.eval(tail_tokens)
    total = skeleton_len + len(tail_tokens)
    answer_tokens = _greedy_continue(llm, total, MAX_ANSWER_TOKENS)
    elapsed = time.perf_counter() - t0
    return llm.detokenize(answer_tokens).decode("utf-8", errors="ignore"), elapsed


def run_r3(llm, trunc_state, trunc_len: int, seam_tokens: list[int], folded_text: str, question: str):
    t0 = time.perf_counter()
    llm.load_state(trunc_state)
    llm.eval(seam_tokens)
    total = trunc_len + len(seam_tokens)
    tail = f"{folded_text}\n\nQuestion: {question}\nAnswer:"
    tail_tokens = llm.tokenize(tail.encode("utf-8"), add_bos=False)
    llm.eval(tail_tokens)
    total += len(tail_tokens)
    answer_tokens = _greedy_continue(llm, total, MAX_ANSWER_TOKENS)
    elapsed = time.perf_counter() - t0
    return llm.detokenize(answer_tokens).decode("utf-8", errors="ignore"), elapsed


def main() -> None:
    llm = model_mod.load_model(n_ctx=4096, logits_all=True)
    full_text = _load_boilerplate_text()
    qa = json.loads(QA_PATH.read_text())["questions"]

    print("building 2b skeleton (rerunning deterministic pipeline)...")
    live_text, folded_text, fold_fraction = build_skeleton(llm, full_text)
    print(f"  fold_fraction={fold_fraction:.3f}, live_text tokens~{len(live_text.split())}w, "
          f"folded_text tokens~{len(folded_text.split())}w")

    # Build the two skeleton KV states once (setup cost, amortized across all 10 questions).
    skeleton_tokens = llm.tokenize(live_text.encode("utf-8"), add_bos=True)
    llm.reset()
    llm.eval(skeleton_tokens)
    skeleton_state = llm.save_state()
    skeleton_len = len(skeleton_tokens)

    trunc_len = max(skeleton_len - SEAM_TOKENS, 1)
    llm.reset()
    llm.eval(skeleton_tokens[:trunc_len])
    trunc_state = llm.save_state()
    seam_tokens = skeleton_tokens[trunc_len:skeleton_len]

    results = []
    for q in qa:
        r1_answer, r1_latency = run_r1(llm, full_text, q["question"])
        r1_correct = check_answer(r1_answer, q["gold_key_facts"])

        r2_answer, r2_latency = run_r2(llm, skeleton_state, skeleton_len, folded_text, q["question"])
        r2_correct = check_answer(r2_answer, q["gold_key_facts"])

        r3_answer, r3_latency = run_r3(llm, trunc_state, trunc_len, seam_tokens, folded_text, q["question"])
        r3_correct = check_answer(r3_answer, q["gold_key_facts"])

        results.append({
            "id": q["id"], "question": q["question"], "gold_key_facts": q["gold_key_facts"],
            "R1": {"answer": r1_answer, "correct": r1_correct, "latency_s": r1_latency},
            "R2": {"answer": r2_answer, "correct": r2_correct, "latency_s": r2_latency},
            "R3": {"answer": r3_answer, "correct": r3_correct, "latency_s": r3_latency},
        })
        print(f"  Q{q['id']}: R1={'OK' if r1_correct else 'MISS'} "
              f"R2={'OK' if r2_correct else 'MISS'} R3={'OK' if r3_correct else 'MISS'}")

    n = len(results)
    r1_acc = sum(r["R1"]["correct"] for r in results) / n
    r2_acc = sum(r["R2"]["correct"] for r in results) / n
    r3_acc = sum(r["R3"]["correct"] for r in results) / n
    r1_lat = sum(r["R1"]["latency_s"] for r in results) / n
    r2_lat = sum(r["R2"]["latency_s"] for r in results) / n
    r3_lat = sum(r["R3"]["latency_s"] for r in results) / n

    n_r1_correct = sum(r["R1"]["correct"] for r in results)
    n_r3_correct = sum(r["R3"]["correct"] for r in results)
    p_o_ordering = (r1_acc >= r3_acc >= r2_acc)
    p_o_within_1 = (n_r1_correct - n_r3_correct) <= 1
    p_o_holds = p_o_ordering and p_o_within_1
    p_p_holds = (r2_lat < 0.25 * r1_lat) and (r3_lat < 0.25 * r1_lat)

    summary = {
        "fold_fraction": fold_fraction,
        "n_questions": n,
        "R1_accuracy": r1_acc, "R2_accuracy": r2_acc, "R3_accuracy": r3_acc,
        "R1_mean_latency_s": r1_lat, "R2_mean_latency_s": r2_lat, "R3_mean_latency_s": r3_lat,
        "R2_latency_fraction_of_R1": r2_lat / r1_lat if r1_lat else None,
        "R3_latency_fraction_of_R1": r3_lat / r1_lat if r1_lat else None,
        "P_O_ordering_holds": p_o_ordering,
        "P_O_within_1_question": p_o_within_1,
        "P_O_holds": p_o_holds,
        "P_P_holds": p_p_holds,
        "gate_holds": bool(p_o_holds and p_p_holds),
        "per_question_results": results,
    }
    out_path = io_mod.save_result("cycle3_iteration3_run.json", summary)
    print(f"\nR1 accuracy: {r1_acc:.2f}  R2 accuracy: {r2_acc:.2f}  R3 accuracy: {r3_acc:.2f}")
    print(f"R1 latency: {r1_lat:.3f}s  R2 latency: {r2_lat:.3f}s ({summary['R2_latency_fraction_of_R1']:.1%} of R1)  "
          f"R3 latency: {r3_lat:.3f}s ({summary['R3_latency_fraction_of_R1']:.1%} of R1)")
    print(f"P_O_holds: {p_o_holds}  P_P_holds: {p_p_holds}  gate_holds: {summary['gate_holds']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
