"""Conditional VAE training loop."""
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
from .model import CVAE


def beta_schedule(epoch: int, warmup_epochs: int, beta_final: float) -> float:
    """Linear KL warmup. Avoids posterior-collapse on tiny categorical outputs."""
    if warmup_epochs <= 0:
        return beta_final
    return min(1.0, epoch / warmup_epochs) * beta_final


def train(
    data_path: str,
    out_dir: str,
    *,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 2e-3,
    z_dim: int = 16,
    cond_dim: int = 64,
    hidden: int = 256,
    beta_final: float = 0.1,
    warmup_epochs: int = 5,
    val_n_samples: int = 4000,
    device: str = "cuda",
    seed: int = 42,
    log_every: int = 50,
) -> Path:
    set_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    train_loader, val_loader, _, _ = make_loaders(data_path, batch_size=batch_size, seed=seed)

    model = CVAE(z_dim=z_dim, cond_dim=cond_dim, hidden=hidden).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    ckpt = CheckpointWriter(out_dir, model_name="cvae")
    step = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        beta = beta_schedule(epoch, warmup_epochs, beta_final)
        running_nll, running_kl, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device_t) for k, v in batch.items()}
            out = model(
                day_idx=batch["day"], yd_idx=batch["year_digit"],
                dow=batch["dow"], month=batch["month"],
                leap=batch["leap"], decade=batch["decade"],
            )
            nll = (
                F.cross_entropy(out["logits_day"], batch["day"])
                + F.cross_entropy(out["logits_yd"], batch["year_digit"])
            )
            # KL(N(mu,sigma) || N(0,I))
            kl = -0.5 * (1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()).sum(dim=-1).mean()
            loss = nll + beta * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running_nll += nll.item()
            running_kl += kl.item()
            n_batches += 1
            step += 1
            if step % log_every == 0:
                ckpt.log(TrainStats(
                    epoch=epoch, step=step,
                    train_loss=loss.item(),
                    extra={"nll": nll.item(), "kl": kl.item(), "beta": beta},
                ))
        sched.step()

        val = validate(model, val_loader, sample_logits_fn=model.sample_logits, n_samples=val_n_samples)
        improved = ckpt.maybe_save(model, val["val_joint_compliance"])
        ckpt.log(TrainStats(
            epoch=epoch, step=step,
            train_loss=running_nll / max(1, n_batches),
            val_joint_compliance=val["val_joint_compliance"],
            val_validity=val["val_validity"],
            val_acc_dow=val["val_acc_dow"],
            val_acc_leap=val["val_acc_leap"],
            secs=time.time() - start,
            extra={"kl": running_kl / max(1, n_batches), "beta": beta, "improved": improved},
        ))
        print(
            f"[cvae] ep {epoch:02d} nll={running_nll/max(1,n_batches):.4f} "
            f"kl={running_kl/max(1,n_batches):.4f} beta={beta:.3f} "
            f"val_joint={val['val_joint_compliance']:.4f}"
            + (" *" if improved else "")
        )
    return ckpt.best_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.txt")
    p.add_argument("--out", default="runs/cvae")
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
