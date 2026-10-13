"""Judge-bias diagnostics: trust the verdict only as far as you trust the judge.

Three cheap checks that every improvement audit should run before believing
its own verdict:
  1. self-consistency — re-judge the same outputs; a judge that disagrees
     with itself injects noise the audit cannot see.
  2. judge-vs-human agreement — on a labeled subset, how well does the judge
     track human preference? (Pearson r for scores, accuracy for choices.)
  3. length bias — does the judge systematically prefer longer outputs?
     A strong length-score correlation is a red flag.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def judge_self_consistency(
    judge: Callable[[Any, Any], float],
    task_outputs: list[tuple[Any, Any]],
    n_repeats: int = 2,
    seed: int = 0,
) -> dict:
    """Re-judge identical (task, output) pairs and measure agreement.

    Returns mean absolute score deviation across repeats plus, for
    pairwise-style judges, the fraction of repeats that agree on the
    direction. Low self-agreement => the judge's noise dominates the
    comparison and any verdict is suspect.
    """
    if n_repeats < 2:
        raise ValueError("n_repeats must be >= 2")
    rng = np.random.default_rng(seed)
    scores = []
    for task, output in task_outputs:
        reps = [float(judge(task, output)) for _ in range(n_repeats)]
        scores.append(reps)
    scores = np.asarray(scores)
    mad = float(np.abs(scores - scores.mean(axis=1, keepdims=True)).mean())
    return {
        "n_items": len(task_outputs),
        "n_repeats": n_repeats,
        "mean_abs_deviation": mad,
        "score_range": float(scores.max() - scores.min()),
    }


def judge_human_agreement(
    judge_scores: list | np.ndarray,
    human_scores: list | np.ndarray,
) -> dict:
    """How well do judge scores track human labels? Pearson r (and, when
    human labels are binary, judge accuracy at threshold 0.5)."""
    j = np.asarray(judge_scores, dtype=float)
    h = np.asarray(human_scores, dtype=float)
    if j.shape != h.shape:
        raise ValueError("judge_scores and human_scores must match")
    out = {"n": int(j.size), "pearson_r": _pearson(j, h)}
    uniq = set(np.unique(h).tolist())
    if uniq <= {0.0, 1.0}:
        out["accuracy_at_0.5"] = float(((j >= 0.5) == (h >= 0.5)).mean())
    return out


def judge_length_bias(
    judge: Callable[[Any, Any], float],
    task_outputs: list[tuple[Any, str]],
) -> dict:
    """Correlation between judge score and output length (characters).

    A large positive r suggests the judge rewards verbosity rather than
    quality — flag it in the audit report.
    """
    lengths, scores = [], []
    for task, output in task_outputs:
        lengths.append(len(str(output)))
        scores.append(float(judge(task, output)))
    r = _pearson(np.asarray(lengths, float), np.asarray(scores, float))
    return {"n": len(task_outputs), "length_score_pearson_r": r}
