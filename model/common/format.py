"""Parsing and formatting for the dates-generator data format.

The on-disk format is, per line:

    [DOW] [MON] [True|False] [DEC] D-M-YYYY

where day and month are unpadded (e.g. ``1-1-1800``, not ``01-01-1800``).
Decade is the integer ``year // 10`` formatted as a 3-digit string with no
zero-padding inside the brackets (i.e. ``180`` for the 1800s through ``220``
for "the 2200s" — which in this dataset contains only the year 2200).
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Final

DAYS_OF_WEEK: Final[tuple[str, ...]] = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
MONTHS: Final[tuple[str, ...]] = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)

YEAR_MIN: Final[int] = 1800
YEAR_MAX: Final[int] = 2200
DECADE_MIN: Final[int] = 180
DECADE_MAX: Final[int] = 220
N_DECADES: Final[int] = DECADE_MAX - DECADE_MIN + 1  # 41

# A single integer id for the full (dow, month, leap, decade) condition tuple.
# Models embed this directly so they get a sharp, memorisation-friendly handle
# on each distinct condition rather than having to reconstruct the joint from
# four separate embeddings.
N_JOINT_CONDITIONS: Final[int] = 7 * 12 * 2 * N_DECADES  # 6888


def joint_condition_id(dow, month, leap, decade):
    """Map the 4 condition indices to one id in ``[0, N_JOINT_CONDITIONS)``.

    Works on plain ints or on broadcastable integer tensors (the arithmetic is
    elementwise), so it can be called inside a model's ``forward``.
    """
    return ((dow * 12 + month) * 2 + leap) * N_DECADES + decade


@dataclass(frozen=True)
class Condition:
    """A single set of input conditions.

    Stored as integer indices so it can be used as a key, hashed, and fed to
    the model without further parsing.
    """
    dow: int      # 0..6  (MON..SUN)
    month: int    # 0..11 (JAN..DEC)
    leap: int     # 0=False, 1=True
    decade: int   # 0..40 (DECADE_MIN..DECADE_MAX)

    @property
    def decade_int(self) -> int:
        """Raw decade integer, e.g. 180, 181, ..., 220."""
        return self.decade + DECADE_MIN

    def to_text(self) -> str:
        """Render the condition prefix in the exact dataset format."""
        return (
            f"[{DAYS_OF_WEEK[self.dow]}] "
            f"[{MONTHS[self.month]}] "
            f"[{'True' if self.leap else 'False'}] "
            f"[{self.decade_int}]"
        )


_LINE_RE = re.compile(
    r"^\[(?P<dow>[A-Z]{3})\]\s+"
    r"\[(?P<mon>[A-Z]{3})\]\s+"
    r"\[(?P<leap>True|False)\]\s+"
    r"\[(?P<dec>\d{3})\]"
    r"(?:\s+(?P<date>\d{1,2}-\d{1,2}-\d{4}))?\s*$"
)


def parse_input_line(line: str) -> tuple[Condition, _dt.date | None]:
    """Parse one line from ``data.txt`` or ``example_input.txt``.

    Returns ``(condition, date_or_None)``. ``date`` is ``None`` for input-only
    lines (``example_input.txt``).
    """
    match = _LINE_RE.match(line.strip())
    if not match:
        raise ValueError(f"Bad line format: {line!r}")
    dow = DAYS_OF_WEEK.index(match["dow"])
    month = MONTHS.index(match["mon"])
    leap = 1 if match["leap"] == "True" else 0
    decade_int = int(match["dec"])
    if not (DECADE_MIN <= decade_int <= DECADE_MAX):
        raise ValueError(f"decade {decade_int} out of supported range")
    decade = decade_int - DECADE_MIN
    cond = Condition(dow=dow, month=month, leap=leap, decade=decade)

    date: _dt.date | None = None
    if match["date"] is not None:
        d_str, m_str, y_str = match["date"].split("-")
        date = _dt.date(int(y_str), int(m_str), int(d_str))
    return cond, date


def format_output_line(cond: Condition, date: _dt.date) -> str:
    """Format a (condition, date) pair as one output line, unpadded date."""
    return f"{cond.to_text()} {date.day}-{date.month}-{date.year}"


def is_leap(year: int) -> bool:
    """Gregorian leap year rule."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
