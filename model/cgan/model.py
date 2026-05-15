"""Conditional GAN over (day, year-digit) categoricals, with a projection
discriminator (Miyato & Koyama, 2018).

* **Generator** ``z, c -> (day_logits, yd_logits)`` is FiLM-conditioned on the
  condition encoding. Training feeds the discriminator Gumbel-softmax samples.
* **Discriminator** is a *projection* critic:

      D(x, c) = psi(phi(x)) + < y_c , phi(x) >

  where ``phi`` is a spectral-normalised feature extractor over the date,
  ``psi`` an unconditional critic head, and ``y_c`` a learned embedding of the
  full condition tuple. The bilinear projection term lets the discriminator
  memorise which dates are valid for each condition (a low-rank factorisation
  of the condition x date validity matrix), so fooling it forces the
  generator to respect *all* conditions — including day-of-week.

Spectral normalisation + a hinge loss keep the adversarial game stable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.blocks import FiLMMLP
from ..common.condition_encoder import ConditionEncoder
from ..common.format import N_JOINT_CONDITIONS, joint_condition_id
from ..common.tokenizer import N_DAY, N_YEAR_DIGIT

X_DIM = N_DAY + N_YEAR_DIGIT  # 41


def gumbel_softmax(
    logits: torch.Tensor,
    tau: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Differentiable soft one-hots. ``logits`` is ``[..., K]``."""
    u = torch.empty_like(logits)
    u.uniform_(generator=generator) if generator is not None else u.uniform_()
    # g = -log(-log(u)); two explicit steps to avoid the precedence trap where
    # postfix .clamp_min() would bind tighter than the unary minus.
    neg_log_u = -torch.log(u.clamp_min(1e-9))
    g = -torch.log(neg_log_u.clamp_min(1e-9))
    return F.softmax((logits + g) / max(1e-6, tau), dim=-1)


def _sn(layer: nn.Linear) -> nn.Module:
    """Spectral-normalise a linear layer (bounds the critic's Lipschitz
    constant so it cannot overpower the generator)."""
    return nn.utils.parametrizations.spectral_norm(layer)


class Generator(nn.Module):
    """``z, c -> (logits_day [B,31], logits_yd [B,10])``."""

    def __init__(self, z_dim: int = 64, cond_dim: int = 96, hidden: int = 384) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.cond_encoder = ConditionEncoder(dim=cond_dim)
        self.net = FiLMMLP(
            in_dim=z_dim, cond_dim=cond_dim, hidden=hidden, out_dim=X_DIM, n_layers=4
        )

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
        out = self.net(z, c)
        return out[:, :N_DAY], out[:, N_DAY:]


class Discriminator(nn.Module):
    """Projection critic: ``psi(phi(x)) + <y_c, phi(x)>``."""

    def __init__(self, hidden: int = 384) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            _sn(nn.Linear(X_DIM, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Linear(hidden, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.psi = _sn(nn.Linear(hidden, 1))
        # Per-condition projection embedding y_c.
        self.cond_proj = nn.Embedding(N_JOINT_CONDITIONS, hidden)
        nn.init.zeros_(self.cond_proj.weight)

    def forward(
        self,
        x_day: torch.Tensor,
        x_yd: torch.Tensor,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.phi(torch.cat([x_day, x_yd], dim=-1))
        uncond = self.psi(feat).squeeze(-1)
        joint = joint_condition_id(dow, month, leap, decade)
        proj = (self.cond_proj(joint) * feat).sum(dim=-1)
        return uncond + proj


class CGAN(nn.Module):
    """Container holding G + D. Sampling interface used by ``predict.py``."""

    def __init__(self, z_dim: int = 64, cond_dim: int = 96, hidden: int = 384) -> None:
        super().__init__()
        self.G = Generator(z_dim=z_dim, cond_dim=cond_dim, hidden=hidden)
        self.D = Discriminator(hidden=hidden)

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
