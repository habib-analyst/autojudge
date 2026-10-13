"""Kill-gate experiments for autojudge.

Simulated judge scores (no LLM needed — the statistics are the invention).
Judge model: score(task, system) = quality(system) + task_noise + judge_noise.

Kill-gate (ALL must hold):
  1. scenario (a) real improvement (d ~ 0.5)       -> WIN
  2. scenario (b) pure noise (A == B)               -> NO_SIGNIFICANT_DIFFERENCE
  3. scenario (c) tiny effect (d ~ 0.1, n = 500)    -> NOT_PRACTICALLY_SIGNIFICANT
     (a naive mean-comparison declares this a win; the audit must refuse)
  4. type-I calibration: >= 200 null sims, rejection rate of p < 0.05 in [0.02, 0.08]
"""

import json
import sys

import numpy as np

from autojudge import audit

N_PERM = 5000
N_BOOT = 5000
ALPHA = 0.05
D_MIN = 0.2


def simulate(quality_a, quality_b, n, task_sd=1.0, judge_sd=0.5, seed=0):
    """Paired judge scores for two systems over n tasks."""
    rng = np.random.default_rng(seed)
    task_effect = rng.normal(0, task_sd, n)  # some tasks are harder
    a = quality_a + task_effect + rng.normal(0, judge_sd, n)
    b = quality_b + task_effect + rng.normal(0, judge_sd, n)
    return a, b


def scenario(name, quality_a, quality_b, n, expected, seed):
    a, b = simulate(quality_a, quality_b, n, seed=seed)
    r = audit(a, b, alpha=ALPHA, d_min=D_MIN, n_perm=N_PERM, n_boot=N_BOOT, seed=seed)
    naive = "B" if r.mean_diff > 0 else "A"
    ok = r.verdict == expected
    print(f"[{name}] verdict={r.verdict} (expected {expected}) "
          f"p={r.p_value:.4f} CI=[{r.ci_lo:+.4f},{r.ci_hi:+.4f}] d={r.cohens_d:+.3f} "
          f"naive-would-say={naive} -> {'PASS' if ok else 'FAIL'}")
    return ok, r


def main():
    results = {}
    ok_all = True

    # (a) real improvement: B better by ~0.5 sd -> WIN
    ok, r = scenario("a-real", 0.0, 0.5, 60, "WIN", seed=11)
    results["a"] = {"verdict": r.verdict, "p": r.p_value, "d": r.cohens_d, "pass": ok}
    ok_all &= ok

    # (b) identical systems, judge noise only -> NO_SIGNIFICANT_DIFFERENCE
    ok, r = scenario("b-noise", 0.0, 0.0, 60, "NO_SIGNIFICANT_DIFFERENCE", seed=22)
    results["b"] = {"verdict": r.verdict, "p": r.p_value, "d": r.cohens_d, "pass": ok}
    ok_all &= ok

    # (c) tiny but real effect d~0.1, n=500: significant, but not practical.
    # A naive mean-comparison declares a win (mean gap > 0); the audit refuses.
    ok, r = scenario("c-tiny", 0.0, 0.1, 500, "NOT_PRACTICALLY_SIGNIFICANT", seed=33)
    results["c"] = {"verdict": r.verdict, "p": r.p_value, "d": r.cohens_d, "pass": ok,
                    "naive_mean_gap": r.mean_diff}
    ok_all &= ok

    # (d) type-I calibration: 200 null simulations, rejection rate of the p-gate
    n_sims = 200
    rejects = 0
    for i in range(n_sims):
        a, b = simulate(0.0, 0.0, 50, seed=1000 + i)
        r = audit(a, b, alpha=ALPHA, d_min=D_MIN, n_perm=2000, n_boot=2000, seed=1000 + i)
        rejects += (r.p_value < ALPHA)
    type1 = rejects / n_sims
    ok_type1 = 0.02 <= type1 <= 0.08
    print(f"[d-typeI] null sims={n_sims} rejection-rate={type1:.4f} (bar [0.02, 0.08]) "
          f"-> {'PASS' if ok_type1 else 'FAIL'}")
    results["typeI"] = {"n_sims": n_sims, "rejection_rate": type1, "pass": ok_type1}
    ok_all &= ok_type1

    results["kill_gate"] = "PASS" if ok_all else "FAIL"
    print(f"\nKILL-GATE: {results['kill_gate']}")
    with open("results/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
