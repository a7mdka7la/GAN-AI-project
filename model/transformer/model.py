"""Tiny GPT-style conditional decoder.

Input sequence (length 7):
    [BOS, dow_tok, mon_tok, leap_tok, dec_tok, ydigit_tok, day_tok]

Year-digit is predicted before day (see ``ConditionTokenizer.encode_target``).
On top of the per-token embeddings, a single joint-condition embedding —
keyed on the full ``(dow, month, leap, decade)`` tuple — is added to every
position, giving the autoregressive heads a sharp handle on which condition
they are generating for.

We compute logits over the full vocabulary at every position; the training
loss only looks at the two output positions. At inference we feed the
5-token prompt, sample ydigit_tok at position 5, append it, then sample
day_tok at position 6.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..common.format import N_JOINT_CONDITIONS, joint_condition_id
from ..common.tokenizer import ConditionTokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)            # each (B, T, H, D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 160,
        n_heads: int = 4,
        n_layers: int = 5,
        d_ff: int = 384,
        max_seq_len: int = 8,
        tokenizer: ConditionTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.tok = tokenizer or ConditionTokenizer()
        self.token_emb = nn.Embedding(self.tok.ids.vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.joint_emb = nn.Embedding(N_JOINT_CONDITIONS, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.tok.ids.vocab_size, bias=False)
        # Tie head to embedding for parameter efficiency.
        self.head.weight = self.token_emb.weight

    def _joint_from_prompt(self, x: torch.Tensor) -> torch.Tensor:
        """Recover the joint-condition id from prompt tokens (positions 1-4)."""
        ids = self.tok.ids
        dow = x[:, 1] - ids.dow_start
        month = x[:, 2] - ids.month_start
        leap = x[:, 3] - ids.leap_start
        decade = x[:, 4] - ids.decade_start
        return joint_condition_id(dow, month, leap, decade)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, T]`` token ids (T >= 5). Returns ``[B, T, vocab]``."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)[None]
        # Add the joint-condition embedding to every position.
        joint = self._joint_from_prompt(x)               # [B]
        h = h + self.joint_emb(joint)[:, None, :]         # broadcast over T
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return self.head(h)

    @torch.no_grad()
    def sample_logits(
        self,
        prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a prompt ``[B, 5]`` (BOS + 4 condition tokens), return
        ``(day_logits [B,31], yd_logits [B,10])`` for the shared sampler.

        Year-digit is produced first from the prompt; the day logits are then
        produced conditioned on the *argmax* year-digit. Because the shared
        sampler later draws year-digit from a validity-masked distribution,
        the day logits are conditioned on a near-certain (but not identical)
        year-digit — a minor approximation for a small categorical.
        """
        logits = self.forward(prompt)                    # [B, 5, V]
        yd_lo, yd_hi = self.tok.year_digit_token_range
        day_lo, day_hi = self.tok.day_token_range
        # Position 4 (last prompt token) predicts the year-digit token.
        yd_logits = logits[:, -1, yd_lo:yd_hi]            # [B, 10]
        argmax_yd_tok = yd_lo + yd_logits.argmax(dim=-1, keepdim=True)
        full = torch.cat([prompt, argmax_yd_tok], dim=1)  # [B, 6]
        full_logits = self.forward(full)
        day_logits = full_logits[:, -1, day_lo:day_hi]    # [B, 31]
        return day_logits, yd_logits
