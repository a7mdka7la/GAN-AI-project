"""Tiny GPT-style conditional decoder.

Input sequence (length 7):
    [BOS, dow_tok, mon_tok, leap_tok, dec_tok, day_tok, ydigit_tok]

We compute logits over the *full* vocabulary at every position; the training
loss only looks at the two output positions (predicting day_tok from index 5
input and ydigit_tok from index 6 input — standard left-shift). At inference
we feed the 5-token prompt, sample day_tok at position 5, append, then sample
ydigit_tok at position 6.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from ..common.tokenizer import ConditionTokenizer, N_DAY, N_YEAR_DIGIT


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
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        max_seq_len: int = 8,
        tokenizer: ConditionTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.tok = tokenizer or ConditionTokenizer()
        self.token_emb = nn.Embedding(self.tok.ids.vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.tok.ids.vocab_size, bias=False)
        # Tie head to embedding for parameter efficiency.
        self.head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, T]`` token ids. Returns ``[B, T, vocab]`` logits."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return self.head(h)

    @torch.no_grad()
    def sample_logits(
        self,
        prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given a prompt of shape ``[B, 5]`` (BOS+4 cond tokens), produce
        slimmed logits over (day, year-digit) value blocks.

        Implementation strategy: feed prompt, read position-4 logits over
        the day vocabulary block; greedily-but-temperature-sampled
        intermediate day token is NOT chosen here — predict.py is responsible
        for that. We return the two heads' restricted logit views so the
        downstream sampler can mask + sample exactly like the other models.

        Note that the second-step (year_digit) logits are conditioned on the
        argmax of the day distribution, NOT a sampled day — to keep this
        interface batched and parallel. For a small categorical with strong
        decade prior this is a minor approximation; production-quality code
        would sample autoregressively.
        """
        # 1) Run prompt through the model once.
        logits = self.forward(prompt)  # [B, 5, V]
        day_lo, day_hi = self.tok.day_token_range
        yd_lo, yd_hi = self.tok.year_digit_token_range
        # Logits for day_tok come from the LAST prompt position (predicts next token).
        day_logits = logits[:, -1, day_lo:day_hi]   # [B, 31]

        # 2) Build the augmented sequence with the *argmax* day token, then
        # read the year-digit logits at position 5.
        argmax_day_tok = day_lo + day_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
        full = torch.cat([prompt, argmax_day_tok], dim=1)  # [B, 6]
        full_logits = self.forward(full)
        yd_logits = full_logits[:, -1, yd_lo:yd_hi]  # [B, 10]
        return day_logits, yd_logits
