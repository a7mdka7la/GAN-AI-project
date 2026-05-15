"""Conditional diffusion over the (day, year-digit) one-hots.

Design choices that make diffusion work on this *discrete* output:

* **x0-prediction with cross-entropy.** Instead of predicting the Gaussian
  noise with an MSE loss, the denoiser predicts the clean date directly as
  ``(day_logits, yd_logits)`` and is trained with cross-entropy against the
  true day / year-digit. This is a direct likelihood objective on the
  discrete target (the same kind of signal the Transformer gets) rather than
  an MSE on a continuous relaxation.
* **FiLM conditioning.** The condition encoding and the timestep embedding
  modulate every layer of the denoiser.
* **Classifier-free guidance** with a learned null-condition embedding.

The forward (noising) process still runs on the +/-1 one-hot vectors; only
the training target and the loss change.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..common.blocks import FiLMMLP
from ..common.condition_encoder import ConditionEncoder
from ..common.tokenizer import N_DAY, N_YEAR_DIGIT

X_DIM = N_DAY + N_YEAR_DIGIT  # 41


def make_betas(num_timesteps: int, schedule: str = "cosine") -> torch.Tensor:
    """Cosine ``beta`` schedule (Nichol & Dhariwal, 2021)."""
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, num_timesteps)
    if schedule != "cosine":
        raise ValueError(f"unknown schedule {schedule!r}")
    s = 0.008
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 1e-5, 0.999)


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    """``x_t, t, cond -> (day_logits, yd_logits)`` — the predicted clean date."""

    def __init__(self, cond_dim: int = 96, hidden: int = 384, t_emb_dim: int = 64) -> None:
        super().__init__()
        self.t_embed = SinusoidalTimeEmbed(t_emb_dim)
        self.t_proj = nn.Linear(t_emb_dim, cond_dim)
        self.cond_encoder = ConditionEncoder(dim=cond_dim)
        # Learned null-condition embedding for classifier-free guidance.
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        self.net = FiLMMLP(
            in_dim=X_DIM, cond_dim=cond_dim, hidden=hidden, out_dim=X_DIM, n_layers=4
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
        drop_cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.cond_encoder(dow, month, leap, decade)
        if drop_cond_mask is not None:
            drop = drop_cond_mask.bool().unsqueeze(-1)
            c = torch.where(drop, self.null_cond.expand_as(c), c)
        cond_vec = c + self.t_proj(self.t_embed(t))
        return self.net(x, cond_vec)


class CondDiffusion(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 200,
        cond_dim: int = 96,
        hidden: int = 384,
        cond_dropout_p: float = 0.1,
        guidance_w: float = 3.0,
    ) -> None:
        super().__init__()
        self.T = num_timesteps
        self.cond_dropout_p = cond_dropout_p
        self.guidance_w = guidance_w
        self.net = Denoiser(cond_dim=cond_dim, hidden=hidden)

        betas = make_betas(num_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_acp", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1 - alphas_cumprod))

    @staticmethod
    def encode_x0(day_idx: torch.Tensor, yd_idx: torch.Tensor) -> torch.Tensor:
        """Concat one-hots scaled to {-1, +1} -> ``[B, 41]`` zero-mean vector."""
        x_day = F.one_hot(day_idx, N_DAY).float()
        x_yd = F.one_hot(yd_idx, N_YEAR_DIGIT).float()
        return 2.0 * torch.cat([x_day, x_yd], dim=-1) - 1.0

    @staticmethod
    def logits_to_x0(day_logits: torch.Tensor, yd_logits: torch.Tensor) -> torch.Tensor:
        """Soft {-1,+1} one-hot reconstruction from predicted logits."""
        soft = torch.cat(
            [F.softmax(day_logits, dim=-1), F.softmax(yd_logits, dim=-1)], dim=-1
        )
        return 2.0 * soft - 1.0

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_acp[t][:, None] * x0 + self.sqrt_one_minus_acp[t][:, None] * noise

    def loss(
        self,
        day_idx: torch.Tensor,
        yd_idx: torch.Tensor,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.encode_x0(day_idx, yd_idx)
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t = self.q_sample(x0, t, torch.randn_like(x0))
        drop = (torch.rand(B, device=x0.device) < self.cond_dropout_p).long()
        pred = self.net(x_t, t, dow, month, leap, decade, drop_cond_mask=drop)
        day_logits, yd_logits = pred[:, :N_DAY], pred[:, N_DAY:]
        # x0-prediction trained with cross-entropy on the discrete target.
        return F.cross_entropy(day_logits, day_idx) + F.cross_entropy(yd_logits, yd_idx)

    def _predict_x0_logits(
        self, x_t, t, dow, month, leap, decade
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Classifier-free-guided x0 logits at timestep t."""
        B = x_t.shape[0]
        zero = torch.zeros(B, device=x_t.device, dtype=torch.long)
        one = torch.ones(B, device=x_t.device, dtype=torch.long)
        pred_c = self.net(x_t, t, dow, month, leap, decade, drop_cond_mask=zero)
        pred_u = self.net(x_t, t, dow, month, leap, decade, drop_cond_mask=one)
        guided = pred_u + self.guidance_w * (pred_c - pred_u)
        return guided[:, :N_DAY], guided[:, N_DAY:]

    @torch.no_grad()
    def sample_logits(
        self,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
        ddim_steps: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the DDIM reverse process; return the final ``(day, yd)`` logits."""
        B = dow.shape[0]
        device = dow.device
        x_t = torch.randn(B, X_DIM, device=device)
        ts = torch.linspace(self.T - 1, 0, ddim_steps, device=device).long()

        day_logits = yd_logits = None
        for i in range(len(ts)):
            t = ts[i].repeat(B)
            day_logits, yd_logits = self._predict_x0_logits(
                x_t, t, dow, month, leap, decade
            )
            x0 = self.logits_to_x0(day_logits, yd_logits)
            a_t = self.alphas_cumprod[t][:, None]
            if i + 1 < len(ts):
                a_next = self.alphas_cumprod[ts[i + 1].repeat(B)][:, None]
                eps = (x_t - torch.sqrt(a_t) * x0) / torch.sqrt(1 - a_t).clamp_min(1e-8)
                x_t = torch.sqrt(a_next) * x0 + torch.sqrt(1 - a_next) * eps
        assert day_logits is not None and yd_logits is not None
        return day_logits, yd_logits
