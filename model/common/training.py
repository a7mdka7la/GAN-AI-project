"""Shared training helpers: validation, sample-based eval, checkpoint saving.

Per-model ``train.py`` modules build their own loop (one optimizer for VAE,
two for GAN, etc.) but reuse this for validation and checkpointing.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from .dataset import DatesDataset
from .format import Condition
from .metrics import evaluate
from .sampling import sample_day_and_digit
from .tokenizer import ConditionTokenizer

import datetime as _dt


# A model is anything exposing ``sample_logits(dow, month, leap, decade) -> (logits_day, logits_yd)``.
SampleLogitsModel = Callable[..., tuple[torch.Tensor, torch.Tensor]]


@dataclass
class TrainStats:
    """One row of training-log output. Persisted to ``runs/<model>/log.jsonl``."""
    epoch: int
    step: int
    train_loss: float
    val_joint_compliance: float | None = None
    val_validity: float | None = None
    val_acc_dow: float | None = None
    val_acc_leap: float | None = None
    secs: float | None = None
    extra: dict | None = None


def conditions_from_batch(batch: dict[str, torch.Tensor]) -> list[Condition]:
    """Reconstruct ``Condition`` objects from a DataLoader batch."""
    dow = batch["dow"].cpu().tolist()
    month = batch["month"].cpu().tolist()
    leap = batch["leap"].cpu().tolist()
    decade = batch["decade"].cpu().tolist()
    return [Condition(dow=d, month=m, leap=l, decade=de)
            for d, m, l, de in zip(dow, month, leap, decade)]


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    sample_logits_fn: SampleLogitsModel | None = None,
    n_samples: int = 5000,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Sample from ``model`` for up to ``n_samples`` validation rows and
    compute compliance metrics.

    ``sample_logits_fn`` defaults to ``model.sample_logits``.
    """
    model.eval()
    fn = sample_logits_fn or model.sample_logits  # type: ignore[attr-defined]
    seen = 0
    pairs: list[tuple[Condition, _dt.date | None]] = []
    for batch in val_loader:
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        logits_day, logits_yd = fn(
            dow=batch["dow"], month=batch["month"],
            leap=batch["leap"], decade=batch["decade"],
        )
        conds = conditions_from_batch(batch)
        samples = sample_day_and_digit(logits_day, logits_yd, conds, generator=generator)
        for cond, (day, yd) in zip(conds, samples):
            year = cond.decade_int * 10 + yd
            try:
                date = _dt.date(year, cond.month + 1, day)
            except ValueError:
                # Should never happen — valid_mask guarantees a legal date.
                date = None
            pairs.append((cond, date))
            seen += 1
        if seen >= n_samples:
            break
    rep = evaluate(pairs)
    return {
        "val_joint_compliance": rep.acc_joint,
        "val_validity": rep.validity,
        "val_acc_dow": rep.acc_dow,
        "val_acc_leap": rep.acc_leap,
    }


def make_loaders(
    data_path: str | Path,
    batch_size: int = 1024,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, ConditionTokenizer]:
    """Build train/val/test DataLoaders with a deterministic 90/5/5 split."""
    from .dataset import load_dataset_lines, split_indices

    rows = load_dataset_lines(data_path)
    train_idx, val_idx, test_idx = split_indices(len(rows), seed=seed)
    tok = ConditionTokenizer()
    train_ds = DatesDataset([rows[i] for i in train_idx], tokenizer=tok)
    val_ds = DatesDataset([rows[i] for i in val_idx], tokenizer=tok)
    test_ds = DatesDataset([rows[i] for i in test_idx], tokenizer=tok)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, tok


class CheckpointWriter:
    """Save the best model by validation joint compliance.

    Stores both the latest and the best checkpoints; ``log.jsonl`` accumulates
    one ``TrainStats`` row per logged step for the report.
    """

    def __init__(self, run_dir: str | Path, model_name: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.log_path = self.run_dir / "log.jsonl"
        self.best_path = self.run_dir / f"{model_name}_best.pt"
        self.last_path = self.run_dir / f"{model_name}_last.pt"
        self.best_score: float = float("-inf")

    def log(self, stats: TrainStats) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(stats)) + "\n")

    def maybe_save(self, model: torch.nn.Module, score: float) -> bool:
        """Save ``model`` if ``score`` improves the best so far."""
        torch.save(model.state_dict(), self.last_path)
        if score > self.best_score:
            self.best_score = score
            torch.save(model.state_dict(), self.best_path)
            return True
        return False


def time_block(start: float) -> float:
    """Convenience wrapper for elapsed time."""
    return time.time() - start
