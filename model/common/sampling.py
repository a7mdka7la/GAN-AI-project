"""Shared sampling utilities used by all 4 models at inference time.

All sampling goes through ``sample_day_and_digit``:

1. Build per-row additive log-masks for the year-digit head; sample one
   year-digit per row from masked logits.
2. Build per-row additive log-masks for the day head, *conditioned on the
   sampled year-digit* (so e.g. February 29 is only allowed when the year
   was actually picked as a leap year), and sample the day.

The whole batch is processed in two GPU calls per head — no per-row
``.item()`` syncs — and the actual draw is done with the **Gumbel-max**
trick rather than ``torch.multinomial``. Gumbel-max is numerically robust:
it never asserts on degenerate / NaN probability rows, and we explicitly
``nan_to_num`` the model's logits before adding the mask so that an
unstable early-training generator (cGAN especially) can't crash validation.
Day-of-week is *not* masked — that's what the model has to learn.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .format import Condition
from .tokenizer import N_DAY, N_YEAR_DIGIT
from .valid_mask import valid_day_mask, valid_year_digit_mask

_NEG_INF: float = -1e9
_LOGIT_CLAMP: float = 30.0


def _build_yd_mask_batch(
    conditions: Sequence[Condition], device: torch.device
) -> torch.Tensor:
    """Stacked year-digit masks, shape ``[B, 10]``."""
    rows: list[list[float]] = []
    for cond in conditions:
        rows.append([0.0 if ok else _NEG_INF for ok in valid_year_digit_mask(cond)])
    return torch.tensor(rows, device=device, dtype=torch.float32)


def _build_day_mask_batch(
    conditions: Sequence[Condition],
    year_digits: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Stacked day masks, shape ``[B, 31]``, given the year-digit chosen for each row."""
    rows: list[list[float]] = []
    for cond, yd in zip(conditions, year_digits):
        rows.append([0.0 if ok else _NEG_INF for ok in valid_day_mask(cond, yd)])
    return torch.tensor(rows, device=device, dtype=torch.float32)


def _clean_logits(logits: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with finite values and clamp to a safe range.

    Necessary because an unstable cGAN generator early in training can emit
    NaNs or values large enough to make the subsequent softmax degenerate.
    Clamping to ``±_LOGIT_CLAMP`` is harmless: ``softmax`` differences of
    more than ~30 are already saturated.
    """
    return torch.nan_to_num(
        logits, nan=0.0, posinf=_LOGIT_CLAMP, neginf=-_LOGIT_CLAMP
    ).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)


def _gumbel_max_sample(
    logits: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    """Sample one index per row using the Gumbel-max trick.

    Equivalent in distribution to ``Categorical(logits=...).sample()`` but
    has no all-zero / NaN failure mode and avoids the
    generator-device-mismatch quirk that ``torch.multinomial`` has when the
    generator is on CPU and the logits are on CUDA.
    """
    u = torch.empty_like(logits)
    if generator is not None and generator.device == u.device:
        u.uniform_(generator=generator)
    else:
        # generator on a different device (or None) — fall back to default RNG.
        u.uniform_()
    # Two clamp_min(1e-9) calls keep both -log(u) and -log(-log(u)) finite.
    g = -torch.log(-torch.log(u.clamp_min(1e-9)).clamp_min(1e-9))
    return (logits + g).argmax(dim=-1)


def sample_day_and_digit(
    logits_day: torch.Tensor,
    logits_yd: torch.Tensor,
    conditions: Sequence[Condition],
    generator: torch.Generator | None = None,
) -> list[tuple[int, int]]:
    """Mask + sample one ``(day, year_digit)`` per condition (batched).

    ``logits_day`` is ``[B, 31]``, ``logits_yd`` is ``[B, 10]``. Returns a
    list of length ``B`` of ``(day, year_digit)`` where ``day ∈ {1..31}``
    and ``year_digit ∈ {0..9}``.
    """
    device = logits_day.device
    assert logits_yd.shape[-1] == N_YEAR_DIGIT
    assert logits_day.shape[-1] == N_DAY

    yd_mask = _build_yd_mask_batch(conditions, device)
    yd_logits = _clean_logits(logits_yd) + yd_mask
    yd_idx = _gumbel_max_sample(yd_logits, generator)  # [B]
    yd_list: list[int] = yd_idx.cpu().tolist()

    day_mask = _build_day_mask_batch(conditions, yd_list, device)
    day_logits = _clean_logits(logits_day) + day_mask
    day_idx = _gumbel_max_sample(day_logits, generator)  # [B]
    day_list: list[int] = day_idx.cpu().tolist()

    return [(day_list[i] + 1, yd_list[i]) for i in range(len(conditions))]


def argmax_day_and_digit(
    logits_day: torch.Tensor,
    logits_yd: torch.Tensor,
    conditions: Sequence[Condition],
) -> list[tuple[int, int]]:
    """Deterministic variant — argmax instead of sampling."""
    device = logits_day.device
    yd_mask = _build_yd_mask_batch(conditions, device)
    yd_idx = (_clean_logits(logits_yd) + yd_mask).argmax(dim=-1)
    yd_list: list[int] = yd_idx.cpu().tolist()
    day_mask = _build_day_mask_batch(conditions, yd_list, device)
    day_idx = (_clean_logits(logits_day) + day_mask).argmax(dim=-1)
    day_list: list[int] = day_idx.cpu().tolist()
    return [(day_list[i] + 1, yd_list[i]) for i in range(len(conditions))]
