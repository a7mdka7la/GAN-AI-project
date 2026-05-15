"""Conditional Diffusion training loop (classifier-free guidance)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..common.seed import set_seed
from ..common.training import (
    CheckpointWriter,
    TrainStats,
    make_loaders,
    validate,
)
from .model import CondDiffusion


def train(
    data_path: str,
    out_dir: str,
    *,
    epochs: int = 60,
    batch_size: int = 1024,
    lr: float = 2e-3,
    num_timesteps: int = 200,
    cond_dim: int = 96,
    hidden: int = 384,
    cond_dropout_p: float = 0.1,
    guidance_w: float = 2.0,
    ddim_steps_val: int = 20,
    grad_clip: float = 1.0,
    val_n_samples: int = 4000,
    device: str = "cuda",
    seed: int = 42,
    log_every: int = 50,
) -> Path:
    set_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    train_loader, val_loader, _, _ = make_loaders(data_path, batch_size=batch_size, seed=seed)

    model = CondDiffusion(
        num_timesteps=num_timesteps, cond_dim=cond_dim, hidden=hidden,
        cond_dropout_p=cond_dropout_p, guidance_w=guidance_w,
    ).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    ckpt = CheckpointWriter(out_dir, model_name="diffusion")
    step = 0
    start = time.time()

    def _sample(*, dow, month, leap, decade):
        return model.sample_logits(dow, month, leap, decade, ddim_steps=ddim_steps_val)

    for epoch in range(epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device_t) for k, v in batch.items()}
            loss = model.loss(
                day_idx=batch["day"], yd_idx=batch["year_digit"],
                dow=batch["dow"], month=batch["month"],
                leap=batch["leap"], decade=batch["decade"],
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
            f"[diffusion] ep {epoch:02d} mse={running/max(1,n_batches):.4f} "
            f"val_joint={val['val_joint_compliance']:.4f} val_dow={val['val_acc_dow']:.4f}"
            + (" *" if improved else "")
        )
    return ckpt.best_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.txt")
    p.add_argument("--out", default="runs/diffusion")
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
