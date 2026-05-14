"""Conditional DDPM on the concatenated one-hots ``[day(31) | year_digit(10)]``.

Continuous diffusion on the 41-dim one-hot space (the "analog bits" trick: a
discrete sample is a one-hot vector in {0,1}^K — we just diffuse it as if it
were a real-valued vector and argmax-split it at the end). Classifier-free
guidance is implemented by dropping the condition embedding with probability
``cond_dropout_p`` during training.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..common.condition_encoder import ConditionEncoder
from ..common.tokenizer import N_DAY, N_YEAR_DIGIT

X_DIM = N_DAY + N_YEAR_DIGIT  # 41


def make_betas(num_timesteps: int, schedule: str = "cosine") -> torch.Tensor:
    """Standard cosine ``β`` schedule (Nichol & Dhariwal, 2021)."""
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
    """MLP ``ε_θ(x_t, t, c)``."""

    def __init__(self, cond_dim: int = 64, hidden: int = 256, t_emb_dim: int = 64) -> None:
        super().__init__()
        self.t_emb = SinusoidalTimeEmbed(t_emb_dim)
        self.cond_encoder = ConditionEncoder(dim=cond_dim)
        self.net = nn.Sequential(
            nn.Linear(X_DIM + cond_dim + t_emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, X_DIM),
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
            # drop_cond_mask is [B] with 1 = drop (use null conditioning).
            c = c * (1 - drop_cond_mask.float()).unsqueeze(-1)
        te = self.t_emb(t)
        return self.net(torch.cat([x, c, te], dim=-1))


class CondDiffusion(nn.Module):
    def __init__(
        self,
        num_timesteps: int = 200,
        cond_dim: int = 64,
        hidden: int = 256,
        cond_dropout_p: float = 0.1,
        guidance_w: float = 2.0,
    ) -> None:
        super().__init__()
        self.T = num_timesteps
        self.cond_dropout_p = cond_dropout_p
        self.guidance_w = guidance_w
        self.net = Denoiser(cond_dim=cond_dim, hidden=hidden)

        betas = make_betas(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Register as non-trainable buffers so they move with .to(device).
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod)
        )

    @staticmethod
    def encode_x0(day_idx: torch.Tensor, yd_idx: torch.Tensor) -> torch.Tensor:
        """Concat one-hots scaled to {-1, +1} → ``[B, 41]`` zero-mean vector.

        Zero-centring matches the N(0, I) noise prior, which speeds
        convergence vs. plain {0,1} one-hots. The sample-time code reads
        the unscaled x_t and argmax-splits the same way (sign is irrelevant
        for argmax)."""
        x_day = F.one_hot(day_idx, N_DAY).float()
        x_yd = F.one_hot(yd_idx, N_YEAR_DIGIT).float()
        return 2.0 * torch.cat([x_day, x_yd], dim=-1) - 1.0

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.sqrt_alphas_cumprod[t][:, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return a * x0 + b * noise

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
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        drop = (torch.rand(B, device=x0.device) < self.cond_dropout_p).long()
        pred = self.net(x_t, t, dow, month, leap, decade, drop_cond_mask=drop)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample_logits(
        self,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
        ddim_steps: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x0 with classifier-free guidance, return split logits.

        We use a DDIM-style fast sampler over a uniform subset of timesteps.
        """
        B = dow.shape[0]
        device = dow.device
        x_t = torch.randn(B, X_DIM, device=device)

        ts = torch.linspace(self.T - 1, 0, ddim_steps, device=device).long()
        for i in range(len(ts)):
            t = ts[i].repeat(B)
            # Classifier-free guidance: condition prediction + null prediction.
            null_mask = torch.ones(B, device=device, dtype=torch.long)
            zero_mask = torch.zeros(B, device=device, dtype=torch.long)
            eps_uncond = self.net(x_t, t, dow, month, leap, decade, drop_cond_mask=null_mask)
            eps_cond = self.net(x_t, t, dow, month, leap, decade, drop_cond_mask=zero_mask)
            eps = eps_uncond + self.guidance_w * (eps_cond - eps_uncond)

            a_t = self.alphas_cumprod[t][:, None]
            a_next = self.alphas_cumprod[ts[i + 1].repeat(B)][:, None] if i + 1 < len(ts) else torch.ones_like(a_t)
            x0_pred = (x_t - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            # DDIM update (eta=0 deterministic)
            x_t = torch.sqrt(a_next) * x0_pred + torch.sqrt(1 - a_next) * eps

        # Final x0 estimate -> split + interpret as logits (the one-hot scores).
        day_logits = x_t[:, :N_DAY]
        yd_logits = x_t[:, N_DAY:]
        return day_logits, yd_logits
