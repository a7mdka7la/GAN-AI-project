"""Conditional GAN (AC-GAN flavour) over (day, year-digit) categoricals.

* Generator: ``z, c -> (logits_day, logits_yd)``. At training time we draw a
  Gumbel-softmax sample for each head and pass *soft one-hots* into the
  discriminator. At sampling time we apply the deterministic-condition mask
  and draw a categorical.
* Discriminator: ``(x_day, x_yd, c) -> real/fake + 4 auxiliary classifiers``.
  The auxiliary heads (AC-GAN) force the generator to actually condition on
  the inputs rather than producing arbitrary valid dates.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.condition_encoder import ConditionEncoder
from ..common.format import N_DECADES
from ..common.tokenizer import N_DAY, N_DOW, N_LEAP, N_MONTH, N_YEAR_DIGIT


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Differentiable soft one-hots. ``logits`` is ``[..., K]``."""
    # Sample Gumbel noise: -log(-log(U)), U ~ Uniform(0,1)
    u = torch.empty_like(logits)
    u.uniform_(generator=generator) if generator is not None else u.uniform_()
    g = -torch.log(-torch.log(u.clamp_min(1e-9)).clamp_min(1e-9))
    return F.softmax((logits + g) / max(1e-6, tau), dim=-1)


class Generator(nn.Module):
    """``z, c -> (logits_day [B,31], logits_yd [B,10])``."""

    def __init__(self, z_dim: int = 64, cond_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.cond_encoder = ConditionEncoder(dim=cond_dim)
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head_day = nn.Linear(hidden, N_DAY)
        self.head_yd = nn.Linear(hidden, N_YEAR_DIGIT)

    def sample_z(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch, self.z_dim, device=device)

    def forward(
        self,
        z: torch.Tensor,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cond_encoder(dow, month, leap, decade)
        h = self.net(torch.cat([z, c], dim=-1))
        return self.head_day(h), self.head_yd(h)


def _sn(layer: nn.Module) -> nn.Module:
    """Wrap a Linear layer in spectral normalization.

    Spectral norm bounds the largest singular value of the weight matrix to 1,
    which keeps the discriminator 1-Lipschitz and prevents the exploding
    gradients that turn the generator's weights into NaN within a few steps.
    """
    return nn.utils.parametrizations.spectral_norm(layer)


class Discriminator(nn.Module):
    """``(x_day [B,31], x_yd [B,10], c) -> (rf [B], aux_dow, aux_mon, aux_leap, aux_dec)``.

    Every Linear in the trunk + the real/fake head is spectral-normalised. The
    auxiliary classification heads do NOT need spectral norm because they're
    only optimised on real data — they don't feed back into the generator.
    """

    def __init__(self, cond_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.cond_encoder = ConditionEncoder(dim=cond_dim)
        self.trunk = nn.Sequential(
            _sn(nn.Linear(N_DAY + N_YEAR_DIGIT + cond_dim, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head_rf = _sn(nn.Linear(hidden, 1))
        self.head_dow = nn.Linear(hidden, N_DOW)
        self.head_mon = nn.Linear(hidden, N_MONTH)
        self.head_leap = nn.Linear(hidden, N_LEAP)
        self.head_dec = nn.Linear(hidden, N_DECADES)

    def forward(
        self,
        x_day: torch.Tensor,
        x_yd: torch.Tensor,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.cond_encoder(dow, month, leap, decade)
        h = self.trunk(torch.cat([x_day, x_yd, c], dim=-1))
        return (
            self.head_rf(h).squeeze(-1),
            self.head_dow(h),
            self.head_mon(h),
            self.head_leap(h),
            self.head_dec(h),
        )


class CGAN(nn.Module):
    """Container holding G + D. Sampling interface used by ``predict.py``."""

    def __init__(self, z_dim: int = 64, cond_dim: int = 64, hidden: int = 256) -> None:
        super().__init__()
        self.G = Generator(z_dim=z_dim, cond_dim=cond_dim, hidden=hidden)
        self.D = Discriminator(cond_dim=cond_dim, hidden=hidden)

    @torch.no_grad()
    def sample_logits(
        self,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.G.sample_z(dow.shape[0], dow.device)
        return self.G(z, dow, month, leap, decade)
