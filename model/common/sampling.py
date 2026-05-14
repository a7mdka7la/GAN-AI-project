"""Shared sampling utilities used by all 4 models at inference time.

The same pattern is used everywhere: produce ``(logits_day [B,31], logits_yd
[B,10])`` from the model, then sample year-digit first (using the year-digit
mask), then sample the day conditioned on the chosen year digit (using the
appropriate day mask). This guarantees every emitted sample respects the
deterministic conditions: month, decade, leap-year, and calendar legality.

Day-of-week is *not* masked — the model has to learn it.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .format import Condition
from .valid_mask import valid_day_mask, valid_year_digit_mask

_NEG_INF = -1e9


def _build_mask(mask_list: Sequence[bool], device: torch.device) -> torch.Tensor:
    """Convert a Python boolean list to a float tensor of additive log-masks."""
    t = torch.zeros(len(mask_list), device=device)
    for i, ok in enumerate(mask_list):
        if not ok:
            t[i] = _NEG_INF
    return t


def _sample_categorical(
    logits: torch.Tensor, generator: torch.Generator | None
) -> int:
    """Sample one index from a 1-D logits tensor."""
    probs = torch.softmax(logits, dim=-1)
    # torch.multinomial requires non-negative probabilities; softmax guarantees this.
    return int(torch.multinomial(probs, 1, generator=generator).item())


def sample_day_and_digit(
    logits_day: torch.Tensor,
    logits_yd: torch.Tensor,
    conditions: Sequence[Condition],
    generator: torch.Generator | None = None,
) -> list[tuple[int, int]]:
    """Mask + sample one (day, year_digit) per condition.

    ``logits_day`` is ``[B, 31]``, ``logits_yd`` is ``[B, 10]``. Returns the
    integer day (1..31) and year_digit (0..9) for each item in the batch.
    """
    device = logits_day.device
    out: list[tuple[int, int]] = []
    for i, cond in enumerate(conditions):
        yd_mask = _build_mask(valid_year_digit_mask(cond), device)
        yd_idx = _sample_categorical(logits_yd[i] + yd_mask, generator)
        day_mask = _build_mask(valid_day_mask(cond, yd_idx), device)
        day_idx_zero_based = _sample_categorical(logits_day[i] + day_mask, generator)
        out.append((day_idx_zero_based + 1, yd_idx))
    return out


def argmax_day_and_digit(
    logits_day: torch.Tensor,
    logits_yd: torch.Tensor,
    conditions: Sequence[Condition],
) -> list[tuple[int, int]]:
    """Deterministic variant — used for ablations / debugging."""
    device = logits_day.device
    out: list[tuple[int, int]] = []
    for i, cond in enumerate(conditions):
        yd_mask = _build_mask(valid_year_digit_mask(cond), device)
        yd_idx = int(torch.argmax(logits_yd[i] + yd_mask).item())
        day_mask = _build_mask(valid_day_mask(cond, yd_idx), device)
        day_idx_zero_based = int(torch.argmax(logits_day[i] + day_mask).item())
        out.append((day_idx_zero_based + 1, yd_idx))
    return out
