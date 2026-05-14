"""Evaluation metrics for the date-generator problem.

All metrics work on plain Python objects — no torch — so they can be reused
inside training loops, in eval scripts, and in pure-Python unit tests.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .format import (
    YEAR_MIN,
    YEAR_MAX,
    DAYS_OF_WEEK,
    Condition,
    is_leap,
)


@dataclass
class EvalReport:
    """Aggregated condition-compliance + validity numbers."""
    n: int = 0
    acc_dow: float = 0.0
    acc_month: float = 0.0
    acc_leap: float = 0.0
    acc_decade: float = 0.0
    acc_joint: float = 0.0
    validity: float = 0.0
    diversity_entropy_mean: float | None = None
    per_condition_counts: dict[str, int] = field(default_factory=dict)

    def as_table(self) -> str:
        """One-line-per-metric table for human reading."""
        lines = [
            f"n                       : {self.n}",
            f"validity (in-range, legal): {self.validity:.4f}",
            f"acc_dow  (day-of-week)  : {self.acc_dow:.4f}",
            f"acc_month               : {self.acc_month:.4f}",
            f"acc_leap                : {self.acc_leap:.4f}",
            f"acc_decade              : {self.acc_decade:.4f}",
            f"acc_joint (all 4 + valid): {self.acc_joint:.4f}",
        ]
        if self.diversity_entropy_mean is not None:
            lines.append(f"diversity entropy (mean): {self.diversity_entropy_mean:.4f}")
        return "\n".join(lines)


def date_validity(date: _dt.date | None) -> bool:
    """True if ``date`` is non-None and within the supported range."""
    if date is None:
        return False
    return _dt.date(YEAR_MIN, 1, 1) <= date <= _dt.date(YEAR_MAX, 12, 31)


def condition_compliance(cond: Condition, date: _dt.date) -> dict[str, bool]:
    """Per-condition checks. Caller is responsible for ensuring date is valid."""
    return {
        "dow": DAYS_OF_WEEK[cond.dow] == DAYS_OF_WEEK[date.weekday()],
        "month": cond.month + 1 == date.month,
        "leap": bool(cond.leap) == is_leap(date.year),
        "decade": cond.decade_int == date.year // 10,
    }


def joint_compliance(cond: Condition, date: _dt.date | None) -> bool:
    """True iff ``date`` is valid and satisfies all 4 conditions."""
    if not date_validity(date):
        return False
    assert date is not None  # narrowing for type-checkers
    return all(condition_compliance(cond, date).values())


def evaluate(pairs: Iterable[tuple[Condition, _dt.date | None]]) -> EvalReport:
    """Aggregate compliance numbers across many (condition, generated_date) pairs."""
    n = 0
    valid_n = 0
    counts = {"dow": 0, "month": 0, "leap": 0, "decade": 0, "joint": 0}
    for cond, date in pairs:
        n += 1
        if not date_validity(date):
            continue
        assert date is not None
        valid_n += 1
        checks = condition_compliance(cond, date)
        for k in counts:
            if k == "joint":
                continue
            if checks[k]:
                counts[k] += 1
        if all(checks.values()):
            counts["joint"] += 1
    if n == 0:
        return EvalReport()
    return EvalReport(
        n=n,
        validity=valid_n / n,
        acc_dow=counts["dow"] / n,
        acc_month=counts["month"] / n,
        acc_leap=counts["leap"] / n,
        acc_decade=counts["decade"] / n,
        acc_joint=counts["joint"] / n,
        per_condition_counts=counts,
    )


def diversity_entropy(samples_per_condition: Sequence[Sequence[_dt.date]]) -> float:
    """Mean Shannon entropy of generated dates per condition.

    ``samples_per_condition[i]`` is the list of N dates generated for the i-th
    condition tuple. Higher entropy ⇒ less mode-collapse. Returned in nats.
    """
    if not samples_per_condition:
        return 0.0
    entropies = []
    for samples in samples_per_condition:
        if not samples:
            continue
        counts = Counter(samples)
        total = sum(counts.values())
        entropies.append(
            -sum((c / total) * math.log(c / total) for c in counts.values())
        )
    return sum(entropies) / max(1, len(entropies))
