"""Shared condition encoder: embeds the 4 condition indices into a single vector.

Used by cGAN, cVAE, and Diffusion (the Transformer has its own embedding path
but uses the same joint-condition id).

The encoder combines two views of the condition:

* four *factored* embeddings (dow, month, leap, decade) — these carry the
  shared structure between conditions, and
* one *joint* embedding keyed on the full ``(dow, month, leap, decade)`` tuple
  id — this gives the model a sharp, per-tuple handle so it can memorise the
  set of valid dates for each individual condition (the factored embeddings
  alone make the network reconstruct the joint, which it does poorly on a
  modular task like day-of-week).
"""
from __future__ import annotations

import torch
from torch import nn

from .format import N_DECADES, N_JOINT_CONDITIONS, joint_condition_id
from .tokenizer import N_DOW, N_MONTH, N_LEAP


class ConditionEncoder(nn.Module):
    """4 factored embeddings + 1 joint embedding, fused by a small MLP."""

    def __init__(self, dim: int = 96, embed_dim: int = 16) -> None:
        super().__init__()
        self.dow_emb = nn.Embedding(N_DOW, embed_dim)
        self.month_emb = nn.Embedding(N_MONTH, embed_dim)
        self.leap_emb = nn.Embedding(N_LEAP, embed_dim)
        self.decade_emb = nn.Embedding(N_DECADES, embed_dim)
        self.joint_emb = nn.Embedding(N_JOINT_CONDITIONS, dim)
        self.proj = nn.Sequential(
            nn.Linear(4 * embed_dim + dim, dim),
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
        joint = joint_condition_id(dow, month, leap, decade)
        e = torch.cat(
            [
                self.dow_emb(dow),
                self.month_emb(month),
                self.leap_emb(leap),
                self.decade_emb(decade),
                self.joint_emb(joint),
            ],
            dim=-1,
        )
        return self.proj(e)
