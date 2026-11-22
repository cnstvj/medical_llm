"""
BLEU (Bilingual Evaluation Understudy) Score Implementation
===========================================================
A complete, mathematically rigorous implementation of the BLEU evaluation metric
from Papineni et al. (2002): "BLEU: a Method for Automatic Evaluation of Machine
Translation", with support for:
  - Modified n-gram precision with clipping (prevents repetition gaming)
  - Brevity Penalty (BP) with closest-reference selection
  - Configurable n-gram weights (BLEU-1, BLEU-2, BLEU-3, BLEU-4, etc.)
  - Sentence-level BLEU and Corpus-level BLEU
  - Multiple reference translations per hypothesis
  - Standard smoothing techniques (Chen & Cherry Method 1, Epsilon/Floor, Laplace Add-1)
  - Pre-tokenized sequences and raw text strings
  - Detailed diagnostic breakdown (n-gram precisions, BP, effective lengths)
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, Optional, Dict, Any
from collections import Counter
import math
import re


@dataclass
class BLEUResult:
    """Structured result containing BLEU score and diagnostic components."""
    score: float
    precisions: List[float]
    brevity_penalty: float
    candidate_length: int
    reference_length: int
    n_gram_order: int
    weights: Tuple[float, ...]
    smoothed: bool
    details: Dict[str, Any]

    def __repr__(self) -> str:
        p_str = ", ".join(f"p_{i+1}={p:.4f}" for i, p in enumerate(self.precisions))
        return (
            f"BLEUResult(score={self.score:.4f}, BP={self.brevity_penalty:.4f}, "
            f"precisions=[{p_str}], c={self.candidate_length}, r={self.reference_length})"
        )


def _tokenize(text_or_tokens: Union[str, Sequence[str]], lowercase: bool = False) -> List[str]:
    """
    Normalizes input to a list of token strings.
    If already a sequence of strings, preserves or lowercases them.
    If a string, tokenizes on alphanumeric words and punctuation symbols.
    """
    if isinstance(text_or_tokens, str):
        if lowercase:
            text_or_tokens = text_or_tokens.lower()
        # Regex extracts words, numbers, and individual punctuation marks
        return re.findall(r"\w+|[^\w\s]", text_or_tokens)
    
    if lowercase:
        return [str(tok).lower() for tok in text_or_tokens]
    return [str(tok) for tok in text_or_tokens]


def compute_ngrams(tokens: Sequence[str], n: int) -> Counter:
    """
    Extracts n-grams of size `n` from a token sequence and returns their counts.
    
    Example:
        >>> compute_ngrams(["the", "cat", "sat"], 2)
        Counter({('the', 'cat'): 1, ('cat', 'sat'): 1})
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    
    num_ngrams = len(tokens) - n + 1
    if num_ngrams <= 0:
        return Counter()
    
    return Counter(tuple(tokens[i:i + n]) for i in range(num_ngrams))


def modified_precision(
    candidate_tokens: Sequence[str],
    reference_tokens_list: Sequence[Sequence[str]],
    n: int,
) -> Tuple[int, int]:
    """
    Computes modified n-gram clipped match count and total candidate n-gram count for a sentence.
    
    Clipped count:
      For each n-gram in candidate, clip its count by the maximum count observed
      in any single reference translation.
      
    Returns:
      (clipped_matches, total_candidate_ngrams)
    """
    cand_counts = compute_ngrams(candidate_tokens, n)
    total_cand_ngrams = sum(cand_counts.values())
    
    if total_cand_ngrams == 0:
        return 0, 0
    
    ref_counters = [compute_ngrams(ref, n) for ref in reference_tokens_list]
    
    clipped_matches = 0
    for ngram, count in cand_counts.items():
        max_ref_count = max((ref_c.get(ngram, 0) for ref_c in ref_counters), default=0)
        clipped_matches += min(count, max_ref_count)
        
    return clipped_matches, total_cand_ngrams


def closest_reference_length(candidate_len: int, reference_lengths: Sequence[int]) -> int:
    """
    Selects the reference length closest to candidate length.
    If multiple references have equal distance, breaks tie by choosing the shorter length
    (following Papineni et al., 2002).
    """
    if not reference_lengths:
        return 0
    # Sort criteria: smallest absolute difference, then smallest length
    return min(reference_lengths, key=lambda r: (abs(r - candidate_len), r))


def calculate_brevity_penalty(candidate_len: int, reference_len: int) -> float:
    """
    Calculates the Brevity Penalty (BP):
      BP = 1.0               if c > r
      BP = exp(1 - r / c)   if c <= r and c > 0
      BP = 0.0               if c == 0
    """
    if candidate_len <= 0:
        return 0.0
    if candidate_len > reference_len:
        return 1.0
    return math.exp(1.0 - (reference_len / candidate_len))


def _normalize_weights(weights: Sequence[float], n: int) -> Tuple[float, ...]:
    """Ensures weights are of length n and sum to 1.0."""
    if not weights:
        w = tuple(1.0 / n for _ in range(n))
        return w
    if len(weights) != n:
        raise ValueError(f"Length of weights ({len(weights)}) does not match n-gram order ({n})")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"Sum of weights must be positive, got {total}")
    return tuple(w / total for w in weights)


def sentence_bleu(
    candidate: Union[str, Sequence[str]],
    references: Union[Union[str, Sequence[str]], Sequence[Union[str, Sequence[str]]]],
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    smoothing: Optional[str] = "method1",
    lowercase: bool = False,
) -> float:
    """
    Calculates sentence-level BLEU score.
    
    Args:
        candidate: Candidate sentence (string or token sequence).
        references: One reference or a list of references.
        weights: Weights for each n-gram order (defaults to uniform 4-gram: BLEU-4).
        smoothing: Smoothing technique ('none', 'method1' / 'chen_cherry_1', 'floor', 'add-1').
                   Default is 'method1' which is recommended for sentence BLEU to avoid 0.0
                   from zero higher-order matches on short sentences.
        lowercase: Whether to case-fold tokens before evaluation.
        
    Returns:
        BLEU score as a float in [0.0, 1.0].
    """
    result = sentence_bleu_detailed(
        candidate=candidate,
        references=references,
        weights=weights,
        smoothing=smoothing,
        lowercase=lowercase,
    )
    return result.score


def _normalize_references_for_sentence(
    references: Union[str, Sequence[Union[str, Sequence[str]]]],
    lowercase: bool = False,
) -> List[List[str]]:
    """
    Normalizes reference input for a single candidate sentence.
    
    Accepts:
      - str: "A single reference sentence."
      - Sequence of str:
          e.g. ["Ref sentence 1", "Ref sentence 2"]
          OR if user passed a single pre-tokenized reference: ["the", "cat", "sat"]
      - Sequence of sequences:
          e.g. [["the", "cat", "sat"], ["a", "cat", "sat"]]
    """
    if isinstance(references, str):
        return [_tokenize(references, lowercase=lowercase)]

    if not isinstance(references, (list, tuple)):
        raise TypeError(f"Unsupported references type: {type(references)}")

    if len(references) == 0:
        raise ValueError("references list cannot be empty")

    first = references[0]
    if isinstance(first, (list, tuple)):
        # List of tokenized references: [ ["the", "cat"], ["a", "cat"] ]
        return [_tokenize(r, lowercase=lowercase) for r in references]

    if isinstance(first, str):
        # Determine if references is a list of sentences or tokens of a single reference
        has_whitespace = any(" " in s for s in references)
        if has_whitespace:
            return [_tokenize(r, lowercase=lowercase) for r in references]
        # Single pre-tokenized reference word list
        return [[s.lower() if lowercase else s for s in references]]

    return [[str(x).lower() if lowercase else str(x) for x in references]]


def sentence_bleu_detailed(
    candidate: Union[str, Sequence[str]],
    references: Union[str, Sequence[Union[str, Sequence[str]]]],
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    smoothing: Optional[str] = "method1",
    lowercase: bool = False,
) -> BLEUResult:
    """
    Calculates sentence-level BLEU score with full breakdown of precisions and brevity penalty.
    """
    cand_tokens = _tokenize(candidate, lowercase=lowercase)
    ref_list = _normalize_references_for_sentence(references, lowercase=lowercase)

    max_n = len(weights)
    norm_weights = _normalize_weights(weights, max_n)
    
    c = len(cand_tokens)
    ref_lengths = [len(r) for r in ref_list]
    r = closest_reference_length(c, ref_lengths)
    
    bp = calculate_brevity_penalty(c, r)
    
    precisions: List[float] = []
    clipped_matches_list: List[int] = []
    total_cand_ngrams_list: List[int] = []
    has_smoothed = False

    for n in range(1, max_n + 1):
        clipped, total = modified_precision(cand_tokens, ref_list, n)
        clipped_matches_list.append(clipped)
        total_cand_ngrams_list.append(total)
        
        if total == 0:
            p_n = 0.0
        else:
            p_n = clipped / total
        
        # Apply smoothing if requested and precision is zero
        if p_n == 0.0 and smoothing and smoothing.lower() != "none":
            has_smoothed = True
            sm = smoothing.lower()
            if sm in ("method1", "chen_cherry_1"):
                # Chen & Cherry Method 1: replace 0 with epsilon / denominator
                epsilon = 0.1
                denom = max(total, 1)
                p_n = epsilon / denom
            elif sm in ("floor", "epsilon"):
                p_n = 1e-4
            elif sm in ("add-1", "laplace"):
                p_n = (clipped + 1.0) / (total + 1.0)
            else:
                # Default fallback
                p_n = 1e-4
        elif smoothing and smoothing.lower() in ("add-1", "laplace"):
            # Laplace smoothing adds 1 to all orders
            has_smoothed = True
            p_n = (clipped + 1.0) / (total + 1.0)

        precisions.append(p_n)

    # Compute weighted geometric mean
    if any(p == 0.0 for p in precisions) or bp == 0.0:
        final_score = 0.0
    else:
        log_sum = sum(w * math.log(p) for w, p in zip(norm_weights, precisions))
        final_score = bp * math.exp(log_sum)

    # Clamp score to [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))

    return BLEUResult(
        score=final_score,
        precisions=precisions,
        brevity_penalty=bp,
        candidate_length=c,
        reference_length=r,
        n_gram_order=max_n,
        weights=norm_weights,
        smoothed=has_smoothed,
        details={
            "clipped_matches": clipped_matches_list,
            "total_candidate_ngrams": total_cand_ngrams_list,
            "reference_lengths": ref_lengths,
        },
    )



def corpus_bleu(
    candidates: Sequence[Union[str, Sequence[str]]],
    references: Sequence[Sequence[Union[str, Sequence[str]]]],
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    smoothing: Optional[str] = "none",
    lowercase: bool = False,
) -> float:
    """
    Calculates corpus-level BLEU score across multiple sentences.
    
    Note:
      Corpus BLEU sums clipped matches and total n-grams across all candidate sentences
      before computing precision (micro-average), rather than averaging sentence scores.
      
    Args:
        candidates: List of candidate sentences.
        references: List of reference lists, one list of references per candidate sentence.
        weights: Weights for each n-gram order.
        smoothing: Smoothing technique ('none', 'method1', 'floor', 'add-1').
        lowercase: Whether to case-fold tokens.
        
    Returns:
        Corpus BLEU score in [0.0, 1.0].
    """
    result = corpus_bleu_detailed(
        candidates=candidates,
        references=references,
        weights=weights,
        smoothing=smoothing,
        lowercase=lowercase,
    )
    return result.score


def corpus_bleu_detailed(
    candidates: Sequence[Union[str, Sequence[str]]],
    references: Sequence[Sequence[Union[str, Sequence[str]]]],
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    smoothing: Optional[str] = "none",
    lowercase: bool = False,
) -> BLEUResult:
    """
    Calculates corpus-level BLEU score with detailed diagnostic metrics.
    """
    if len(candidates) != len(references):
        raise ValueError(
            f"Number of candidates ({len(candidates)}) must match number of reference sets ({len(references)})"
        )
    if not candidates:
        return BLEUResult(
            score=0.0,
            precisions=[0.0] * len(weights),
            brevity_penalty=0.0,
            candidate_length=0,
            reference_length=0,
            n_gram_order=len(weights),
            weights=_normalize_weights(weights, len(weights)),
            smoothed=False,
            details={},
        )

    max_n = len(weights)
    norm_weights = _normalize_weights(weights, max_n)

    total_clipped = [0] * max_n
    total_cand_ngrams = [0] * max_n
    total_c = 0
    total_r = 0

    for cand, refs in zip(candidates, references):
        cand_toks = _tokenize(cand, lowercase=lowercase)
        ref_tok_list = _normalize_references_for_sentence(refs, lowercase=lowercase)

        c_len = len(cand_toks)
        r_lengths = [len(r) for r in ref_tok_list]
        r_len = closest_reference_length(c_len, r_lengths)

        total_c += c_len
        total_r += r_len

        for n in range(1, max_n + 1):
            clipped, cand_ng = modified_precision(cand_toks, ref_tok_list, n)
            total_clipped[n - 1] += clipped
            total_cand_ngrams[n - 1] += cand_ng

    bp = calculate_brevity_penalty(total_c, total_r)

    precisions: List[float] = []
    has_smoothed = False

    for n in range(1, max_n + 1):
        clipped = total_clipped[n - 1]
        cand_ng = total_cand_ngrams[n - 1]

        if cand_ng == 0:
            p_n = 0.0
        else:
            p_n = clipped / cand_ng

        if p_n == 0.0 and smoothing and smoothing.lower() != "none":
            has_smoothed = True
            sm = smoothing.lower()
            if sm in ("method1", "chen_cherry_1"):
                epsilon = 0.1
                denom = max(cand_ng, 1)
                p_n = epsilon / denom
            elif sm in ("floor", "epsilon"):
                p_n = 1e-4
            elif sm in ("add-1", "laplace"):
                p_n = (clipped + 1.0) / (cand_ng + 1.0)
            else:
                p_n = 1e-4
        elif smoothing and smoothing.lower() in ("add-1", "laplace"):
            has_smoothed = True
            p_n = (clipped + 1.0) / (cand_ng + 1.0)

        precisions.append(p_n)

    if any(p == 0.0 for p in precisions) or bp == 0.0:
        final_score = 0.0
    else:
        log_sum = sum(w * math.log(p) for w, p in zip(norm_weights, precisions))
        final_score = bp * math.exp(log_sum)

    final_score = max(0.0, min(1.0, final_score))

    return BLEUResult(
        score=final_score,
        precisions=precisions,
        brevity_penalty=bp,
        candidate_length=total_c,
        reference_length=total_r,
        n_gram_order=max_n,
        weights=norm_weights,
        smoothed=has_smoothed,
        details={
            "corpus_clipped_matches": total_clipped,
            "corpus_candidate_ngrams": total_cand_ngrams,
        },
    )
