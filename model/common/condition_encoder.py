"""Shared condition encoder: embeds the 4 condition indices into a single vector.

Used by cGAN, cVAE, and Diffusion (the Transformer treats condition tokens
through its main embedding table instead).
"""
from __future__ import annotations

import torch
from torch import nn

from .tokenizer import N_DOW, N_MONTH, N_LEAP
from .format import N_DECADES


class ConditionEncoder(nn.Module):
    """4 embeddings concatenated then passed through a small MLP.

    Output dim is fixed at construction. Default ``dim=64`` keeps the cond
    vector small relative to the latent / hidden sizes of the per-model
    networks.
    """

    def __init__(self, dim: int = 64, embed_dim: int = 16) -> None:
        super().__init__()
        self.dow_emb = nn.Embedding(N_DOW, embed_dim)
        self.month_emb = nn.Embedding(N_MONTH, embed_dim)
        self.leap_emb = nn.Embedding(N_LEAP, embed_dim)
        self.decade_emb = nn.Embedding(N_DECADES, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(4 * embed_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.dim = dim

    def forward(
        self,
        dow: torch.Tensor,
        month: torch.Tensor,
        leap: torch.Tensor,
        decade: torch.Tensor,
    ) -> torch.Tensor:
        e = torch.cat(
            [
                self.dow_emb(dow),
                self.month_emb(month),
                self.leap_emb(leap),
                self.decade_emb(decade),
            ],
            dim=-1,
        )
        return self.proj(e)
