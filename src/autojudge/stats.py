"""Core statistics: the significance gate that hype must pass through.

Decision rule (WIN) requires ALL THREE:
  1. paired permutation-test p < alpha (default 0.05)
  2. bootstrap 95% CI of the mean paired difference excludes zero
  3. practical effect size: |Cohen's d_z| >= d_min (default 0.2)

A naive mean-comparison would declare a win from (3) failing alone; the
audit exists to catch exactly those cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ALPHA = 0.05
D_MIN = 0.2


def _as_array(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size < 2:
        raise ValueError(f"{name} must be a 1-D array with >= 2 scores")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite scores")
    return a


def paired_permutation_test(
    a: list | np.ndarray,
    b: list | np.ndarray,
    n_perm: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation p-value on the mean difference.

    Under the null (no systematic difference), the sign of each paired
    difference is exchangeable; we flip signs at random and compare the
    observed |mean| against the permutation distribution.
    """
    a, b = _as_array(a, "a"), _as_array(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length")
    diff = b - a
    obs = abs(diff.mean())
    if obs == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    perm_means = np.abs((signs * diff).mean(axis=1))
    # +1 smoothing: never report a literal zero p-value
    return float((1 + np.sum(perm_means >= obs)) / (1 + n_perm))


def bootstrap_ci(
    a: list | np.ndarray,
    b: list | np.ndarray,
    alpha: float = ALPHA,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference (b - a)."""
    a, b = _as_array(a, "a"), _as_array(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length")
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def cohens_d_paired(a: list | np.ndarray, b: list | np.ndarray) -> float:
    """Cohen's d_z for paired samples: mean(diff) / sd(diff)."""
    a, b = _as_array(a, "a"), _as_array(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length")
    diff = b - a
    sd = diff.std(ddof=1)
    if sd == 0.0:
        return 0.0
    return float(diff.mean() / sd)


@dataclass(frozen=True)
class AuditResult:
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    p_value: float
    ci_lo: float
    ci_hi: float
    cohens_d: float
    alpha: float
    d_min: float
    verdict: str  # WIN | REGRESSION | NOT_PRACTICALLY_SIGNIFICANT | NO_SIGNIFICANT_DIFFERENCE
    win_components: tuple[bool, bool, bool]  # (p_ok, ci_ok, d_ok) for B>A

    @property
    def naive_winner(self) -> str:
        return "B" if self.mean_diff > 0 else ("A" if self.mean_diff < 0 else "tie")


def audit(
    a: list | np.ndarray,
    b: list | np.ndarray,
    alpha: float = ALPHA,
    d_min: float = D_MIN,
    n_perm: int = 10_000,
    n_boot: int = 10_000,
    seed: int = 0,
) -> AuditResult:
    """Run the full significance audit on paired judge scores.

    Verdicts:
      WIN — statistically significant AND practically meaningful improvement (B > A)
      REGRESSION — statistically significant AND practically meaningful drop (B < A)
      NOT_PRACTICALLY_SIGNIFICANT — p < alpha and CI excludes 0, but |d| < d_min
      NO_SIGNIFICANT_DIFFERENCE — none of the above (could not reject the null)
    """
    a, b = _as_array(a, "a"), _as_array(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same length")
    p = paired_permutation_test(a, b, n_perm=n_perm, seed=seed)
    ci_lo, ci_hi = bootstrap_ci(a, b, alpha=alpha, n_boot=n_boot, seed=seed + 1)
    d = cohens_d_paired(a, b)
    mean_diff = float(b.mean() - a.mean())

    p_ok = p < alpha
    ci_ok = ci_lo > 0 or ci_hi < 0
    d_ok = abs(d) >= d_min

    if p_ok and ci_ok and d_ok:
        verdict = "WIN" if mean_diff > 0 else "REGRESSION"
    elif p_ok and ci_ok:
        verdict = "NOT_PRACTICALLY_SIGNIFICANT"
    else:
        verdict = "NO_SIGNIFICANT_DIFFERENCE"

    return AuditResult(
        n=int(a.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=mean_diff,
        p_value=p,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        cohens_d=d,
        alpha=alpha,
        d_min=d_min,
        verdict=verdict,
        win_components=(p_ok, ci_ok, d_ok),
    )


def min_detectable_d(n: int, alpha: float = ALPHA, power: float = 0.8) -> float:
    """Back-of-envelope minimum Cohen's d_z detectable with given power.

    Useful for sizing a comparison BEFORE running it: if the effect you care
    about is smaller than this, collect more tasks.
    """
    from statistics import NormalDist

    nd = NormalDist()
    z_a = nd.inv_cdf(1 - alpha / 2)
    z_b = nd.inv_cdf(power)
    return float((z_a + z_b) / math.sqrt(n))
