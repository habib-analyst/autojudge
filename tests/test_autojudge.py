"""Unit tests for autojudge (fast, deterministic)."""

import numpy as np
import pytest

from autojudge import (
    audit,
    bootstrap_ci,
    cohens_d_paired,
    judge_human_agreement,
    judge_length_bias,
    judge_self_consistency,
    paired_permutation_test,
    PairedComparison,
    render_audit,
)

RNG = np.random.default_rng(7)


def test_permutation_null_p_uniformish():
    # under the null, p-values should be roughly uniform: mostly not tiny
    ps = [paired_permutation_test(RNG.normal(size=30), RNG.normal(size=30), n_perm=500, seed=i) for i in range(20)]
    assert sum(p < 0.05 for p in ps) <= 3  # not rejecting wildly


def test_permutation_detects_real_effect():
    a = RNG.normal(0, 1, 40)
    b = RNG.normal(0.8, 1, 40)
    assert paired_permutation_test(a, b, n_perm=2000, seed=0) < 0.01


def test_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0.5, 1.0, 200)
    lo, hi = bootstrap_ci(np.zeros(200), diffs, n_boot=2000, seed=1)
    assert lo < 0.5 < hi


def test_cohens_d_paired():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = a + 1.0  # constant shift => d = mean(1)/std(0)... guard: sd=0 => 0
    assert cohens_d_paired(a, b) == 0.0
    b = a + np.array([0.5, 1.5, 0.5, 1.5])
    d = cohens_d_paired(a, b)
    assert d > 0 and np.isfinite(d)


def test_audit_verdicts():
    rng = np.random.default_rng(1)
    # real improvement
    r = audit(rng.normal(0, 1, 80), rng.normal(0.6, 1, 80), seed=0)
    assert r.verdict == "WIN", r.verdict
    # pure noise
    r = audit(rng.normal(0, 1, 80), rng.normal(0, 1, 80), seed=2)
    assert r.verdict == "NO_SIGNIFICANT_DIFFERENCE", r.verdict
    # tiny effect: significant-ish but d < 0.2 -> not practical
    r = audit(rng.normal(0, 0.2, 400), rng.normal(0.02, 0.2, 400), seed=3)
    assert r.verdict in ("NOT_PRACTICALLY_SIGNIFICANT", "NO_SIGNIFICANT_DIFFERENCE"), r.verdict
    # regression
    r = audit(rng.normal(0.6, 1, 80), rng.normal(0, 1, 80), seed=4)
    assert r.verdict == "REGRESSION", r.verdict


def test_audit_rejects_bad_input():
    with pytest.raises(ValueError):
        audit([1.0], [2.0])
    with pytest.raises(ValueError):
        audit([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        audit([1.0, float("nan")], [1.0, 2.0])


def test_paired_comparison_harness():
    tasks = [{"id": i} for i in range(12)]
    comp = PairedComparison(name_a="base", name_b="tuned", tasks=tasks)
    res = comp.run(
        system_a=lambda t: "x" * (5 + t["id"] % 3),
        system_b=lambda t: "x" * (25 + (t["id"] * 7) % 9),
        judge=lambda t, out: float(len(out)),
        n_perm=500,
        n_boot=500,
        seed=0,
    )
    assert res.n_scored == 12 and res.n_failures == 0
    assert res.verdict == "WIN"
    assert "VERDICT: WIN" in render_audit(res)


def test_judge_self_consistency_deterministic_judge():
    items = [(i, f"output {i}") for i in range(5)]
    rep = judge_self_consistency(lambda t, o: len(o), items, n_repeats=3)
    assert rep["mean_abs_deviation"] == 0.0


def test_judge_self_consistency_noisy_judge():
    rng = np.random.default_rng(0)
    items = [(i, f"output {i}") for i in range(20)]
    rep = judge_self_consistency(lambda t, o: len(o) + rng.normal(0, 5), items, n_repeats=4, seed=0)
    assert rep["mean_abs_deviation"] > 0


def test_judge_human_agreement():
    j = [0.1, 0.4, 0.6, 0.9]
    h = [0.0, 0.0, 1.0, 1.0]
    rep = judge_human_agreement(j, h)
    assert rep["pearson_r"] > 0.8
    assert rep["accuracy_at_0.5"] == 1.0


def test_judge_length_bias_detects_verbosity_lover():
    items = [(i, "x" * (10 * (i + 1))) for i in range(10)]
    rep = judge_length_bias(lambda t, o: float(len(o)), items)
    assert rep["length_score_pearson_r"] > 0.99


def test_report_contains_gates():
    rng = np.random.default_rng(5)
    r = audit(rng.normal(0, 1, 50), rng.normal(0.7, 1, 50), seed=0)
    from autojudge.evaluate import ComparisonResult

    res = ComparisonResult("A", "B", 50, 50, 0, r, np.zeros(50), np.zeros(50))
    doc = render_audit(res)
    assert "## Gates" in doc and "p-value" in doc


def test_min_detectable_d_sanity():
    from autojudge.stats import min_detectable_d

    assert min_detectable_d(25) > min_detectable_d(400) > 0
