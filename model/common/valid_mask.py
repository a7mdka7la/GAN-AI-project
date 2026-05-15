"""Compute boolean validity masks over (day, year_digit) given a Condition.

A model's *learnable* job is the day-of-week and leap-year conditions. Month
and decade are determined directly by the input, so we can rule out impossible
outputs at sample time without learning them.

For a given condition ``c``:

* ``valid_year_digit_mask(c)`` returns a boolean mask of size 10 indicating
  which last digits of the year produce a year in the requested decade with
  the requested leap-or-not status, and within ``[YEAR_MIN, YEAR_MAX]``.

* ``valid_day_mask(c, year_digit)`` returns a boolean mask of size 31
  indicating which day-of-month values exist for the (month, full_year) pair.

The per-model code first samples a year_digit using a logit-mask of
``valid_year_digit_mask``, then a day using ``valid_day_mask`` for that year.
Day-of-week is NOT masked — that's what the model has to learn.
"""
from __future__ import annotations

import calendar

from .format import (
    YEAR_MIN,
    YEAR_MAX,
    Condition,
    is_leap,
)
from .tokenizer import N_DAY, N_YEAR_DIGIT


def valid_year_digit_mask(cond: Condition) -> list[bool]:
    """Boolean list of length 10 — True if (decade*10 + digit) is a valid year
    matching the leap-year condition and within [YEAR_MIN, YEAR_MAX]."""
    base = cond.decade_int * 10
    leap_required = bool(cond.leap)
    mask = []
    for digit in range(N_YEAR_DIGIT):
        year = base + digit
        ok = (YEAR_MIN <= year <= YEAR_MAX) and (is_leap(year) == leap_required)
        mask.append(ok)
    return mask


def valid_day_mask(cond: Condition, year_digit: int) -> list[bool]:
    """Boolean list of length 31 — True if day (1..31) exists for the month
    of ``cond`` in year ``decade*10 + year_digit``."""
    year = cond.decade_int * 10 + year_digit
    if not (YEAR_MIN <= year <= YEAR_MAX):
        return [False] * N_DAY
    month_idx = cond.month + 1  # 1..12
    days_in_month = calendar.monthrange(year, month_idx)[1]
    return [d < days_in_month for d in range(N_DAY)]


def full_valid_mask(cond: Condition) -> list[list[bool]]:
    """31x10 boolean matrix. ``m[d][y]`` is True iff (day=d+1, ydigit=y) is a
    valid (in-range, month-legal, leap-respecting) date for ``cond``.

    The model still has to learn day-of-week. This mask only constrains the
    deterministic conditions (month, decade, leap, calendar legality)."""
    ydigit_mask = valid_year_digit_mask(cond)
    out = [[False] * N_YEAR_DIGIT for _ in range(N_DAY)]
    for y, y_ok in enumerate(ydigit_mask):
        if not y_ok:
            continue
        day_mask = valid_day_mask(cond, y)
        for d, d_ok in enumerate(day_mask):
            out[d][y] = d_ok
    return out


def any_valid(cond: Condition) -> bool:
    """True iff at least one (day, year_digit) pair satisfies the deterministic
    conditions. (All condition tuples in the real dataset should pass this.)"""
    return any(any(row) for row in full_valid_mask(cond))


def compliance_matrix(cond: Condition) -> list[list[float]]:
    """31x10 matrix; ``m[d][y] = 1.0`` iff (day=d+1, year_digit=y) yields a
    valid, in-range date that satisfies ALL four conditions of ``cond`` —
    including day-of-week. Used as an exact day-of-week oracle for the cGAN's
    auxiliary compliance loss."""
    out = [[0.0] * N_YEAR_DIGIT for _ in range(N_DAY)]
    for y in range(N_YEAR_DIGIT):
        year = cond.decade_int * 10 + y
        if not (YEAR_MIN <= year <= YEAR_MAX):
            continue
        if is_leap(year) != bool(cond.leap):
            continue
        month_idx = cond.month + 1
        days_in_month = calendar.monthrange(year, month_idx)[1]
        for d in range(N_DAY):
            if d >= days_in_month:
                continue
            # calendar.weekday: Monday=0 .. Sunday=6, matching our dow indices.
            if calendar.weekday(year, month_idx, d + 1) == cond.dow:
                out[d][y] = 1.0
    return out


def build_compliance_table() -> list[list[list[float]]]:
    """Compliance matrices for every joint-condition id, indexed
    ``table[joint_id][day][year_digit]``. Built once; the cGAN converts it to
    a tensor and gathers per batch."""
    from .format import N_JOINT_CONDITIONS, N_DECADES

    table: list[list[list[float]]] = []
    for jid in range(N_JOINT_CONDITIONS):
        decade = jid % N_DECADES
        rest = jid // N_DECADES
        leap = rest % 2
        rest //= 2
        month = rest % 12
        dow = rest // 12
        table.append(compliance_matrix(Condition(dow=dow, month=month, leap=leap, decade=decade)))
    return table
