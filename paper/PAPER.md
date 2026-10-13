# autojudge: Significance-Tested Improvement Audit for Agent Comparisons

**Habib Ur Rehman** · October 2026

## Abstract

LLM-as-judge evaluation has a hype problem: a new agent version scores
0.62 vs the old 0.59, the team ships it, and nobody asks whether the gap is
real. **autojudge** is a paired comparison harness that refuses to declare a
win until the improvement is statistically real. Given two system versions
run on the same tasks with the same judge, it requires all three: (1) a
two-sided paired permutation-test p < 0.05, (2) a bootstrap 95% confidence
interval on the mean paired difference that excludes zero, and (3) a
practical effect size, Cohen's d_z ≥ 0.2. It ships with judge-bias
diagnostics (self-consistency, human agreement, length bias) and renders a
verdict document — WIN / REGRESSION / NOT_PRACTICALLY_SIGNIFICANT /
NO_SIGNIFICANT_DIFFERENCE — with a plain-language read designed to kill
hype. Measured: 3/3 scenario verdicts correct, including refusing a
statistically significant (p = 0.0022) but practically meaningless (d = 0.146)
improvement a naive mean-comparison would ship; the permutation test is
calibrated, with a type-I rejection rate of 0.045 over 200 null simulations.

## 1. The gap

The LLM-judge ecosystem is crowded: arena-style leaderboards, judge-only
eval harnesses, pairwise preference collectors. What nobody ships is the
**decision rule**: when does a judge-measured gap justify the claim "B is
better than A"? Published agent papers routinely report a mean-score delta
over a few dozen tasks and call it an improvement — no test, no interval,
no effect size. This is how noise gets shipped.

autojudge's contribution is not another judge — it is the audit layer that
sits *after* any judge and enforces statistical honesty. The three-gate
rule is deliberately conservative: significance alone is not enough (gate 3
kills the "significant but meaningless" win), and a large effect without
significance is not enough either (gate 1 kills the "looks big on 12 tasks"
win). To our knowledge, no existing agent-eval tool couples paired
permutation testing, bootstrap CIs, a practical effect-size bar, judge-bias
diagnostics, and a hype-killing report in one pip-installable package.

## 2. Method

**Paired harness.** `PairedComparison` runs systems A and B on the same
task list and scores both outputs with the same judge function, collecting
paired scores `(a_i, b_i)`. Failures are counted, not silently dropped.

**Gate 1 — paired permutation test.** Under the null of no systematic
difference, the sign of each paired difference `d_i = b_i - a_i` is
exchangeable. We flip signs 10,000 times and compare the observed
`|mean(d)|` to the permutation distribution (two-sided, +1 smoothing so a
literal zero p-value is never reported).

**Gate 2 — bootstrap CI.** Percentile bootstrap (10,000 resamples) of the
mean paired difference; the gate passes only if the 95% interval excludes
zero.

**Gate 3 — practical effect size.** Cohen's d_z = mean(d)/sd(d) for paired
samples, with default bar 0.2 ("small" effect). This is the hype-killer: a
real-but-tiny gain fails here.

**Verdicts.** WIN (all gates pass, B > A), REGRESSION (all gates pass,
B < A), NOT_PRACTICALLY_SIGNIFICANT (gates 1–2 pass, gate 3 fails),
NO_SIGNIFICANT_DIFFERENCE (gate 1 or 2 fails).

**Judge diagnostics.** (1) Self-consistency: re-judge the same outputs,
report mean absolute deviation across repeats. (2) Human agreement: Pearson
r against human labels on a labeled subset (plus accuracy for binary
labels). (3) Length bias: correlation of judge score with output length —
a strong positive r flags a verbosity-rewarding judge.

**Sizing helper.** `min_detectable_d(n)` gives the back-of-envelope Cohen's
d detectable at 80% power, so teams can size a comparison *before* running it.

## 3. Results

All numbers measured on CPU (numpy only), deterministic seeds.

| Scenario | Setup | Audit verdict | Naive verdict | Result |
|---|---|---|---|---|
| (a) real improvement | B better by ~0.5 sd, n=60 | **WIN** (p=0.0002, CI=[+0.29,+0.64], d=+0.65) | B | correct |
| (b) pure noise | A ≡ B, judge noise, n=60 | **NO_SIGNIFICANT_DIFFERENCE** (p=0.48) | A (mean gap −0.06) | correct |
| (c) tiny effect | B better by ~0.1 sd, n=500 | **NOT_PRACTICALLY_SIGNIFICANT** (p=0.0022, CI excludes 0, d=+0.146) | B (mean gap +0.11) | correct — refused |
| (d) type-I calibration | 200 null sims, n=50 | rejection rate **0.045** (bar [0.02, 0.08]) | — | calibrated |

Scenario (c) is the point of the tool: the difference is genuinely
significant (p = 0.0022) — a significance-only harness would ship it — but
the effect is practically meaningless (d = 0.146 < 0.2). autojudge is, to
our knowledge, the first agent-eval harness whose headline demo is a win it
*refuses* to declare.

Unit tests: 13/13 green. Full scenario suite runs in ~1 s on CPU.

## 4. Honest limitations

1. **The audit is only as good as the judge.** If the judge is biased
   (e.g., rewards verbosity), the gates test a biased signal precisely.
   The diagnostics exist to catch this, but they are checks, not cures.
2. **Paired design assumed.** Systems must be scored on the same tasks;
   unpaired comparisons are out of scope (a Welch-style extension is
   future work).
3. **The d ≥ 0.2 bar is a default, not a theorem.** Domains with tiny but
   valuable gains (e.g., safety) should lower it — the audit makes the bar
   explicit rather than hiding it.
4. **Simulated judges in the kill-gate.** The scenario suite uses synthetic
   score distributions, not a real LLM judge; the statistics are validated,
   the judge-noise model is illustrative.
5. **Multiple comparisons uncorrected.** Auditing many variants against
   one baseline inflates the family-wise error rate; users should apply
   their own correction (documented in the README).

## 5. Related work

LLM-as-judge evaluation (pairwise preference, arena leaderboards) provides
the scores; autojudge provides the verdict rule. Classical paired testing
(permutation tests, bootstrap CIs, Cohen's d) is textbook; the contribution
is packaging the three-gate decision rule with judge diagnostics and a
hype-killing report as a single tool for agent teams.

## 6. Reproducibility

`pip install autojudge`, then `python examples/quickstart.py`. The kill-gate
suite: `python experiments/run_killgate.py` (writes `results/summary.json`).
All experiments are seeded; numpy is the only dependency.

---

*Built by Habib Ur Rehman — the judge that refuses to declare a win until
the improvement is statistically real.*
