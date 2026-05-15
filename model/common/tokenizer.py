"""Tokenizer for the date-generator problem.

Vocabulary layout (single shared vocab, used by the Transformer; the other
models can ignore the ``transformer_*`` ids and use only the categorical sizes):

    0..6     day-of-week tokens (MON..SUN)
    7..18    month tokens       (JAN..DEC)
    19,20    leap tokens        (False, True)
    21..61   decade tokens      (180..220)
    62..92   day tokens         (1..31)
    93..102  year-last-digit    (0..9)
    103      <BOS>
    104      <PAD>
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .format import (
    DAYS_OF_WEEK,
    MONTHS,
    DECADE_MIN,
    DECADE_MAX,
    N_DECADES,
    Condition,
)

N_DOW: Final[int] = 7
N_MONTH: Final[int] = 12
N_LEAP: Final[int] = 2
N_DAY: Final[int] = 31           # day-of-month, classes 1..31 → index 0..30
N_YEAR_DIGIT: Final[int] = 10    # last digit of year, 0..9


@dataclass(frozen=True)
class TokenIds:
    """Pre-computed start indices for each vocab block."""
    dow_start: int = 0
    month_start: int = N_DOW
    leap_start: int = N_DOW + N_MONTH
    decade_start: int = N_DOW + N_MONTH + N_LEAP
    day_start: int = N_DOW + N_MONTH + N_LEAP + N_DECADES
    year_digit_start: int = N_DOW + N_MONTH + N_LEAP + N_DECADES + N_DAY
    bos: int = N_DOW + N_MONTH + N_LEAP + N_DECADES + N_DAY + N_YEAR_DIGIT
    pad: int = N_DOW + N_MONTH + N_LEAP + N_DECADES + N_DAY + N_YEAR_DIGIT + 1
    vocab_size: int = N_DOW + N_MONTH + N_LEAP + N_DECADES + N_DAY + N_YEAR_DIGIT + 2


class ConditionTokenizer:
    """Encodes conditions and (day, year-digit) outputs as token id sequences.

    Designed for the autoregressive Transformer. Other models read structured
    indices off the same object via :py:meth:`encode_condition_indices`.
    """

    def __init__(self) -> None:
        self.ids = TokenIds()

    # ---- structured indices (used by cGAN / cVAE / Diffusion) -------------
    def encode_condition_indices(self, cond: Condition) -> tuple[int, int, int, int]:
        """Return raw 0-based indices ``(dow, month, leap, decade)``."""
        return cond.dow, cond.month, cond.leap, cond.decade

    # ---- token sequence form (used by Transformer) ------------------------
    def encode_prompt(self, cond: Condition) -> list[int]:
        """Returns the input prompt token sequence ``[BOS, DOW, MON, LEAP, DEC]``."""
        return [
            self.ids.bos,
            self.ids.dow_start + cond.dow,
            self.ids.month_start + cond.month,
            self.ids.leap_start + cond.leap,
            self.ids.decade_start + cond.decade,
        ]

    def encode_target(self, day: int, year_digit: int) -> list[int]:
        """Encode the two-token target ``[ydigit_tok, day_tok]``.

        Year-digit comes first: predicting it from the condition alone is the
        easier sub-problem, and conditioning the day on an already-chosen year
        means the model only has to place the day on the right weekday for a
        known year. ``day`` is 1..31 and ``year_digit`` is 0..9.
        """
        if not 1 <= day <= 31:
            raise ValueError(f"day {day} out of range 1..31")
        if not 0 <= year_digit <= 9:
            raise ValueError(f"year_digit {year_digit} out of range 0..9")
        return [self.ids.year_digit_start + year_digit, self.ids.day_start + (day - 1)]

    def decode_day_token(self, tok: int) -> int:
        """Map a model-emitted day token id back to the integer 1..31."""
        return tok - self.ids.day_start + 1

    def decode_year_digit_token(self, tok: int) -> int:
        """Map a model-emitted year-digit token id back to the integer 0..9."""
        return tok - self.ids.year_digit_start

    @property
    def day_token_range(self) -> tuple[int, int]:
        return self.ids.day_start, self.ids.day_start + N_DAY

    @property
    def year_digit_token_range(self) -> tuple[int, int]:
        return self.ids.year_digit_start, self.ids.year_digit_start + N_YEAR_DIGIT
