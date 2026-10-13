"""autojudge — significance-tested improvement audit.

A comparison harness for agent/system versions that refuses to declare a win
until the improvement is statistically real: paired permutation test +
bootstrap confidence interval + practical effect-size gate.
"""

from .stats import (
    audit,
    bootstrap_ci,
    cohens_d_paired,
    paired_permutation_test,
)
from .evaluate import PairedComparison, JudgeFn, SystemFn
from .diagnostics import (
    judge_self_consistency,
    judge_human_agreement,
    judge_length_bias,
)
from .report import render_audit

__all__ = [
    "audit",
    "bootstrap_ci",
    "cohens_d_paired",
    "paired_permutation_test",
    "PairedComparison",
    "JudgeFn",
    "SystemFn",
    "judge_self_consistency",
    "judge_human_agreement",
    "judge_length_bias",
    "render_audit",
]

__version__ = "0.1.0"
