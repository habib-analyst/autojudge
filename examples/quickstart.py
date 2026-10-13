"""autojudge quickstart: audit a "new and improved" agent version.

Two versions of a support agent answer the same 40 tickets; an LLM-judge
stand-in scores them. The audit tells you whether the improvement is real —
or just noise a naive mean-comparison would celebrate.
"""

import numpy as np

from autojudge import PairedComparison, audit, render_audit

rng = np.random.default_rng(42)

# 40 support tickets of varying difficulty
tasks = [{"ticket": i, "difficulty": rng.normal(0, 1)} for i in range(40)]

# v1: baseline agent. v2: "improved" — but is it? (quality bump of 0.55 sd)
def v1(task):
    return 3.0 + 0.4 * task["difficulty"] + rng.normal(0, 0.6)

def v2(task):
    return 3.0 + 0.55 + 0.4 * task["difficulty"] + rng.normal(0, 0.6)

# pointwise judge: scores the raw answer quality
judge = lambda task, answer: answer

comp = PairedComparison(name_a="support-v1", name_b="support-v2", tasks=tasks)
res = comp.run(v1, v2, judge, seed=0)
print(render_audit(res))

# --- the skeptic's case: a tiny tweak with a positive mean gap ---
a = rng.normal(0, 1, 500)
b = rng.normal(0.08, 1, 500)  # d ~ 0.08: real, but practically meaningless
r = audit(a, b, seed=1)
print()
print(f"Tiny tweak: naive mean gap = {r.mean_diff:+.4f} (a naive eval ships it).")
print(f"autojudge verdict: {r.verdict}  |d|={abs(r.cohens_d):.3f} < 0.2 -> correctly refused.")
