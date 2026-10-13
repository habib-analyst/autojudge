[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/autojudge.svg)](https://github.com/habib-analyst/autojudge/stargazers)

# autojudge

**The judge that refuses to declare a win until the improvement is statistically real.**

Your new agent scores 0.62 vs the old 0.59. Ship it? A naive eval says yes.
autojudge runs the paired comparison through three gates — permutation-test
significance, a bootstrap confidence interval, and a practical effect-size
bar — and tells you when the "win" is just noise. Its headline demo is a
win it *correctly refuses*: p = 0.0022 (significant!) but d = 0.146 (meaningless).

## Quickstart

```bash
pip install autojudge
```

```python
from autojudge import PairedComparison, render_audit

tasks = [{"ticket": i} for i in range(40)]
comp = PairedComparison(name_a="support-v1", name_b="support-v2", tasks=tasks)

res = comp.run(
    system_a=lambda t: old_agent(t),   # your system: task -> output
    system_b=lambda t: new_agent(t),
    judge=lambda t, out: llm_judge(t, out),  # your judge: (task, output) -> score
    seed=0,
)
print(render_audit(res))
```

Real output:

```
# Improvement audit: support-v1 vs support-v2

**VERDICT: WIN** — real improvement.

| metric | value |
|---|---|
| tasks scored | 40 |
| mean difference (B - A) | +0.6187 |
| permutation p-value | 0.0001 (alpha = 0.05) |
| bootstrap 95% CI of mean diff | [+0.4327, +0.7978] |
| Cohen's d_z | +1.0354 (practical bar 0.2) |

## Gates (all three must pass for a WIN)
| gate | result |
|---|---|
| p < 0.05 | PASS |
| CI excludes zero | PASS |
| |d| >= 0.2 | PASS |
```

And the skeptic's case — a tiny tweak a naive eval would ship:

```python
from autojudge import audit
a = rng.normal(0, 1, 500)
b = rng.normal(0.08, 1, 500)   # d ~ 0.08: real, but meaningless
r = audit(a, b, seed=1)
# naive mean gap = +0.0924 (a naive eval ships it)
# autojudge verdict: NO_SIGNIFICANT_DIFFERENCE -> correctly refused
```

## Measured results

All measured on CPU (numpy only), deterministic seeds. Kill-gate suite:
`python experiments/run_killgate.py`.

| Scenario | Audit verdict | Naive verdict | Correct? |
|---|---|---|---|
| Real improvement (B +0.5 sd, n=60) | **WIN** (p=0.0002, d=+0.65) | B | ✅ |
| Pure noise (A ≡ B, n=60) | **NO_SIGNIFICANT_DIFFERENCE** (p=0.48) | A | ✅ |
| Tiny effect (B +0.1 sd, n=500) | **NOT_PRACTICALLY_SIGNIFICANT** (p=0.0022, d=+0.146) | B | ✅ refused |
| Type-I calibration (200 null sims) | rejection rate **0.045** ∈ [0.02, 0.08] | — | ✅ calibrated |

13/13 unit tests green. Full suite runs in ~1 s.

## How it works

```
tasks ──► system A ──► judge ──► scores a_i ──┐
        └► system B ──► judge ──► scores b_i ──┤──► paired differences d_i = b_i − a_i
                                              │
                    ┌── gate 1: paired permutation test, p < 0.05
                    ├── gate 2: bootstrap 95% CI of mean(d) excludes 0
                    └── gate 3: |Cohen's d_z| ≥ 0.2   ◄── the hype-killer
                                              │
                    verdict: WIN / REGRESSION / NOT_PRACTICALLY_SIGNIFICANT
                             / NO_SIGNIFICANT_DIFFERENCE
```

Plus judge-bias diagnostics (`judge_self_consistency`,
`judge_human_agreement`, `judge_length_bias`) and `min_detectable_d(n)` to
size a comparison *before* running it.

## Citation

```bibtex
@software{rehman2026autojudge,
  author  = {Habib Ur Rehman},
  title   = {autojudge: Significance-Tested Improvement Audit for Agent Comparisons},
  year    = {2026},
  url     = {https://github.com/habib-analyst/autojudge}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
