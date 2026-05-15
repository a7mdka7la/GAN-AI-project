"""Shared neural building blocks.

``FiLMMLP`` is a residual MLP whose every hidden layer is FiLM-modulated
(Feature-wise Linear Modulation) by a conditioning vector. Injecting the
condition at every layer — rather than only concatenating it at the input —
makes the network actually use the condition, which is what the cGAN
generator, the cVAE decoder and the diffusion denoiser all need here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FiLMMLP(nn.Module):
    """Residual MLP with per-layer FiLM conditioning.

    Each layer computes ``h <- h + Linear(GELU(FiLM(h; cond)))`` where
    ``FiLM(h) = (1 + gamma) * h + beta`` and ``gamma, beta`` are produced
    from the conditioning vector.
    """

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        hidden: int,
        out_dim: int,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.films = nn.ModuleList(
            [nn.Linear(cond_dim, 2 * hidden) for _ in range(n_layers)]
        )
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for film, layer, norm in zip(self.films, self.layers, self.norms):
            gamma, beta = film(cond).chunk(2, dim=-1)
            modulated = (1.0 + gamma) * norm(h) + beta
            h = h + layer(F.gelu(modulated))
        return self.out(h)
