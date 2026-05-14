"""PyTorch dataset wrapping ``data/data.txt``.

Each row yields a dict of integer tensors that the per-model code can use
directly. We keep the *encoding* here so models stay simple.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .format import Condition, parse_input_line
from .tokenizer import ConditionTokenizer


def load_dataset_lines(path: str | Path) -> list[tuple[Condition, int, int]]:
    """Parse all data rows. Returns ``[(condition, day, year_last_digit), ...]``."""
    out: list[tuple[Condition, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cond, date = parse_input_line(line)
            if date is None:
                raise ValueError(f"data row is missing a date: {line!r}")
            out.append((cond, date.day, date.year % 10))
    return out


def split_indices(
    n: int,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Reproducible 90/5/5 random split (override via fractions)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = perm[:n_val]
    test = perm[n_val : n_val + n_test]
    train = perm[n_val + n_test :]
    return train, val, test


class DatesDataset(Dataset):
    """Wraps a parsed list and exposes the encoded fields each model needs."""

    def __init__(
        self,
        rows: Sequence[tuple[Condition, int, int]],
        tokenizer: ConditionTokenizer | None = None,
    ) -> None:
        self.rows = list(rows)
        self.tok = tokenizer or ConditionTokenizer()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cond, day, year_digit = self.rows[idx]
        dow, month, leap, decade = self.tok.encode_condition_indices(cond)
        prompt = self.tok.encode_prompt(cond)
        target = self.tok.encode_target(day, year_digit)
        return {
            "dow":        torch.tensor(dow,        dtype=torch.long),
            "month":      torch.tensor(month,      dtype=torch.long),
            "leap":       torch.tensor(leap,       dtype=torch.long),
            "decade":     torch.tensor(decade,     dtype=torch.long),
            "day":        torch.tensor(day - 1,    dtype=torch.long),   # 0..30
            "year_digit": torch.tensor(year_digit, dtype=torch.long),   # 0..9
            "prompt":     torch.tensor(prompt,     dtype=torch.long),   # (5,)
            "target":     torch.tensor(target,     dtype=torch.long),   # (2,)
        }
