"""Paired evaluation harness: run systems A and B on the SAME tasks,
score both with the SAME judge, collect paired scores for the audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .stats import AuditResult, audit

# A system maps a task -> an output (any object the judge can score).
SystemFn = Callable[[Any], Any]
# A judge maps (task, output) -> a numeric score (higher = better).
JudgeFn = Callable[[Any, Any], float]


@dataclass
class PairedComparison:
    """One audited comparison of system A vs system B."""

    name_a: str = "A"
    name_b: str = "B"
    tasks: list = field(default_factory=list)

    def run(
        self,
        system_a: SystemFn,
        system_b: SystemFn,
        judge: JudgeFn,
        *,
        n_perm: int = 10_000,
        n_boot: int = 10_000,
        seed: int = 0,
    ) -> "ComparisonResult":
        if not self.tasks:
            raise ValueError("no tasks to evaluate")
        scores_a, scores_b, failures = [], [], 0
        for t in self.tasks:
            try:
                out_a, out_b = system_a(t), system_b(t)
                scores_a.append(float(judge(t, out_a)))
                scores_b.append(float(judge(t, out_b)))
            except Exception:
                failures += 1
        if len(scores_a) < 2:
            raise ValueError("fewer than 2 successful paired evaluations")
        audit_result = audit(
            scores_a, scores_b, n_perm=n_perm, n_boot=n_boot, seed=seed
        )
        return ComparisonResult(
            name_a=self.name_a,
            name_b=self.name_b,
            n_tasks=len(self.tasks),
            n_scored=len(scores_a),
            n_failures=failures,
            audit=audit_result,
            scores_a=np.asarray(scores_a),
            scores_b=np.asarray(scores_b),
        )


@dataclass
class ComparisonResult:
    name_a: str
    name_b: str
    n_tasks: int
    n_scored: int
    n_failures: int
    audit: AuditResult
    scores_a: np.ndarray
    scores_b: np.ndarray

    @property
    def verdict(self) -> str:
        return self.audit.verdict
