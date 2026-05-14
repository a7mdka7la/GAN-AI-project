"""Conditional VAE with two categorical decoder heads (day, year-digit)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.condition_encoder import ConditionEncoder
from ..common.tokenizer import N_DAY, N_YEAR_DIGIT


class CVAE(nn.Module):
    def __init__(
        self,
        z_dim: int = 16,
        cond_dim: int = 64,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.cond_encoder = ConditionEncoder(dim=cond_dim)

        # Encoder reads one-hot day + one-hot year-digit + condition.
        self.encoder = nn.Sequential(
            nn.Linear(N_DAY + N_YEAR_DIGIT + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)

        # Decoder maps (z, cond) -> two logit heads.
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head_day = nn.Linear(hidden, N_DAY)
        self.head_yd = nn.Linear(hidden, N_YEAR_DIGIT)

    # ----- training-time forward ------------------------------------------
    def encode(
        self,
        day_idx: torch.Tensor,
        yd_idx: torch.Tensor,
        cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_day = F.one_hot(day_idx, N_DAY).float()
        x_yd = F.one_hot(yd_idx, N_YEAR_DIGIT).float()
        h = self.encoder(torch.cat([x_day, x_yd, cond], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(torch.cat([z, cond], dim=-1))
        return self.head_day(h), self.head_yd(h)

    def forward(
        self,
        day_idx: torch.Tensor,
        yd_idx: torch.Tensor,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cond = self.cond_encoder(dow, month, leap, decade)
        mu, logvar = self.encode(day_idx, yd_idx, cond)
        z = self.reparameterize(mu, logvar)
        logits_day, logits_yd = self.decode(z, cond)
        return {
            "logits_day": logits_day,
            "logits_yd": logits_yd,
            "mu": mu,
            "logvar": logvar,
        }

    # ----- sampling-time forward ------------------------------------------
    @torch.no_grad()
    def sample_logits(
        self,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.cond_encoder(dow, month, leap, decade)
        z = torch.randn(dow.shape[0], self.z_dim, device=dow.device)
        return self.decode(z, cond)
