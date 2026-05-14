"""Tiny conditional Transformer training loop."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..common.seed import set_seed
from ..common.training import (
    CheckpointWriter,
    TrainStats,
    make_loaders,
    validate,
)
from ..common.tokenizer import ConditionTokenizer
from .model import ConditionalTransformer


def train(
    data_path: str,
    out_dir: str,
    *,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 3e-4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 256,
    grad_clip: float = 1.0,
    val_n_samples: int = 4000,
    device: str = "cuda",
    seed: int = 42,
    log_every: int = 50,
) -> Path:
    set_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    train_loader, val_loader, _, tok = make_loaders(data_path, batch_size=batch_size, seed=seed)

    model = ConditionalTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
        max_seq_len=8, tokenizer=tok,
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    ckpt = CheckpointWriter(out_dir, model_name="transformer")
    step = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device_t) for k, v in batch.items()}
            # Build [prompt(5) | day_tok | yd_tok] = 7-token sequence
            seq = torch.cat([batch["prompt"], batch["target"]], dim=1)  # [B, 7]
            logits = model(seq[:, :-1])  # predict positions 1..6 from inputs 0..5
            # Targets are seq[:, 1:]. We only count loss at the last two positions
            # (day_tok at output index 4, yd_tok at output index 5).
            day_lo, day_hi = tok.day_token_range
            yd_lo, yd_hi = tok.year_digit_token_range
            # Position 4 of output predicts seq[:, 5] = day_tok; subset its logits to day vocab.
            day_logits = logits[:, 4, day_lo:day_hi]
            yd_logits = logits[:, 5, yd_lo:yd_hi]
            loss = (
                F.cross_entropy(day_logits, batch["day"])
                + F.cross_entropy(yd_logits, batch["year_digit"])
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            n_batches += 1
            step += 1
            if step % log_every == 0:
                ckpt.log(TrainStats(epoch=epoch, step=step, train_loss=loss.item()))
        sched.step()

        # Validation: predict.py-style sampling using model.sample_logits over prompts.
        def _sample(*, dow, month, leap, decade):
            B = dow.shape[0]
            prompts = torch.stack([
                torch.full((B,), tok.ids.bos, device=dow.device, dtype=torch.long),
                dow + tok.ids.dow_start,
                month + tok.ids.month_start,
                leap + tok.ids.leap_start,
                decade + tok.ids.decade_start,
            ], dim=1)
            return model.sample_logits(prompts)

        val = validate(model, val_loader, sample_logits_fn=_sample, n_samples=val_n_samples)
        improved = ckpt.maybe_save(model, val["val_joint_compliance"])
        ckpt.log(TrainStats(
            epoch=epoch, step=step,
            train_loss=running / max(1, n_batches),
            val_joint_compliance=val["val_joint_compliance"],
            val_validity=val["val_validity"],
            val_acc_dow=val["val_acc_dow"],
            val_acc_leap=val["val_acc_leap"],
            secs=time.time() - start,
            extra={"improved": improved},
        ))
        print(
            f"[transformer] ep {epoch:02d} loss={running/max(1,n_batches):.4f} "
            f"val_joint={val['val_joint_compliance']:.4f} val_dow={val['val_acc_dow']:.4f}"
            + (" *" if improved else "")
        )
    return ckpt.best_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.txt")
    p.add_argument("--out", default="runs/transformer")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train(
        data_path=args.data, out_dir=args.out,
        epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, seed=args.seed,
    )


if __name__ == "__main__":
    main()
