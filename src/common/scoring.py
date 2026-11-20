"""Core scoring primitive shared across experiments (spec 0.3, 0.4).

score_tokens(text, context) -> list[TokenScore], where TokenScore carries
token_id, logprob_true (bits, i.e. log2 p), rank_true (0 = greedy top-1),
and the full-distribution entropy (bits) at that position.
"""
import math
from dataclasses import dataclass

import numpy as np

# Fixed block scheme (spec 0.4): each eval covers at most BLOCK_SIZE tokens,
# of which up to LEFT_CTX are carried left-context from the previous block.
BLOCK_SIZE = 2048
LEFT_CTX = 512

_LOG2 = math.log(2)


@dataclass
class TokenScore:
    token_id: int
    logprob_true: float  # bits: log2 p(true token)
    rank_true: int  # 0 = model's greedy top-1
    entropy: float  # bits


def _logprobs_and_entropy_bits(logits_row: np.ndarray):
    m = np.max(logits_row)
    shifted = logits_row - m
    log_sum_exp = m + np.log(np.sum(np.exp(shifted)))
    log_probs_nats = logits_row - log_sum_exp
    probs = np.exp(log_probs_nats)
    entropy_nats = -np.sum(probs * log_probs_nats)
    return log_probs_nats / _LOG2, entropy_nats / _LOG2


def _score_positions(llm, tokens: list[int], score_from: int) -> list[TokenScore]:
    """Score tokens[score_from:] using logits already computed by llm.eval(tokens)."""
    scores = np.asarray(llm.scores)[: len(tokens)]
    out = []
    for i in range(score_from, len(tokens)):
        logits_row = scores[i - 1]
        true_id = tokens[i]
        log_probs_bits, entropy_bits = _logprobs_and_entropy_bits(logits_row)
        rank_true = int(np.sum(log_probs_bits > log_probs_bits[true_id]))
        out.append(
            TokenScore(
                token_id=true_id,
                logprob_true=float(log_probs_bits[true_id]),
                rank_true=rank_true,
                entropy=float(entropy_bits),
            )
        )
    return out


def _score_blocked(llm, full: list[int], score_from: int) -> list[TokenScore]:
    results: list[TokenScore] = []
    pos = score_from
    while pos < len(full):
        left = max(0, pos - LEFT_CTX)
        end = min(len(full), left + BLOCK_SIZE)
        end = max(end, pos + 1)
        block = full[left:end]
        llm.reset()
        llm.eval(block)
        local_score_from = pos - left
        results.extend(_score_positions(llm, block, local_score_from))
        pos = end
    return results


def score_tokens(llm, text: str, context: str = "") -> list[TokenScore]:
    ctx_bytes = context.encode("utf-8")
    text_bytes = text.encode("utf-8")
    ctx_tokens = llm.tokenize(ctx_bytes, add_bos=True)
    text_tokens = llm.tokenize(text_bytes, add_bos=False)
    full = ctx_tokens + text_tokens
    score_from = len(ctx_tokens)

    if len(full) <= llm.n_ctx():
        llm.reset()
        llm.eval(full)
        return _score_positions(llm, full, score_from)

    return _score_blocked(llm, full, score_from)


def unit_token_spans(llm, unit_texts: list[str], joiner: str = "\n\n") -> tuple[str, list[tuple[int, int]]]:
    """Join unit_texts with joiner; return (full_text, [(token_start, token_end), ...])
    giving each unit's token range within full_text's tokenization (no leading BOS).
    Boundaries are derived from cumulative tokenization of growing prefixes, so they
    are exact at whitespace joins (BPE does not merge across the joiner)."""
    full_text = joiner.join(unit_texts)
    spans = []
    prefix = ""
    prev_tok_count = 0
    for i, t in enumerate(unit_texts):
        prefix += (joiner if i > 0 else "") + t
        tok_count = len(llm.tokenize(prefix.encode("utf-8"), add_bos=False))
        spans.append((prev_tok_count, tok_count))
        prev_tok_count = tok_count
    return full_text, spans
