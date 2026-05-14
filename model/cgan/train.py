"""cGAN training loop (AC-GAN style).

Use either as a script:

    python -m model.cgan.train --data data/data.txt --epochs 30 --out runs/cgan

or by importing :py:func:`train` from a notebook.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..common.seed import set_seed
from ..common.training import (
    CheckpointWriter,
    TrainStats,
    make_loaders,
    validate,
)
from ..common.tokenizer import N_DAY, N_YEAR_DIGIT
from .model import CGAN, gumbel_softmax


def gumbel_anneal(epoch: int, total: int, tau_start: float, tau_end: float) -> float:
    """Linear schedule from ``tau_start`` down to ``tau_end``."""
    if total <= 1:
        return tau_end
    frac = epoch / (total - 1)
    return tau_start + (tau_end - tau_start) * frac


def train(
    data_path: str,
    out_dir: str,
    *,
    epochs: int = 30,
    batch_size: int = 1024,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    aux_weight: float = 1.0,
    z_dim: int = 64,
    cond_dim: int = 64,
    hidden: int = 256,
    tau_start: float = 1.0,
    tau_end: float = 0.3,
    val_n_samples: int = 4000,
    device: str = "cuda",
    seed: int = 42,
    log_every: int = 50,
) -> Path:
    set_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    train_loader, val_loader, _, _ = make_loaders(data_path, batch_size=batch_size, seed=seed)

    model = CGAN(z_dim=z_dim, cond_dim=cond_dim, hidden=hidden).to(device_t)
    opt_g = torch.optim.AdamW(model.G.parameters(), lr=lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.AdamW(model.D.parameters(), lr=lr_d, betas=(0.5, 0.9))

    ckpt = CheckpointWriter(out_dir, model_name="cgan")
    step = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        tau = gumbel_anneal(epoch, epochs, tau_start, tau_end)
        running_d, running_g = 0.0, 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device_t) for k, v in batch.items()}
            B = batch["dow"].shape[0]
            real_day = F.one_hot(batch["day"], N_DAY).float()
            real_yd = F.one_hot(batch["year_digit"], N_YEAR_DIGIT).float()

            # ---- discriminator step ------------------------------------
            with torch.no_grad():
                z = model.G.sample_z(B, device_t)
                fake_day_logits, fake_yd_logits = model.G(
                    z, batch["dow"], batch["month"], batch["leap"], batch["decade"]
                )
                fake_day = gumbel_softmax(fake_day_logits, tau)
                fake_yd = gumbel_softmax(fake_yd_logits, tau)

            rf_real, aux_dow_r, aux_mon_r, aux_leap_r, aux_dec_r = model.D(
                real_day, real_yd, batch["dow"], batch["month"], batch["leap"], batch["decade"]
            )
            rf_fake, aux_dow_f, aux_mon_f, aux_leap_f, aux_dec_f = model.D(
                fake_day, fake_yd, batch["dow"], batch["month"], batch["leap"], batch["decade"]
            )
            d_adv = (
                F.binary_cross_entropy_with_logits(rf_real, torch.ones_like(rf_real))
                + F.binary_cross_entropy_with_logits(rf_fake, torch.zeros_like(rf_fake))
            )
            # AC-GAN auxiliary loss on real samples only — pushes D to be a real classifier.
            d_aux = (
                F.cross_entropy(aux_dow_r, batch["dow"])
                + F.cross_entropy(aux_mon_r, batch["month"])
                + F.cross_entropy(aux_leap_r, batch["leap"])
                + F.cross_entropy(aux_dec_r, batch["decade"])
            )
            d_loss = d_adv + aux_weight * d_aux
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # ---- generator step ----------------------------------------
            z = model.G.sample_z(B, device_t)
            fake_day_logits, fake_yd_logits = model.G(
                z, batch["dow"], batch["month"], batch["leap"], batch["decade"]
            )
            fake_day = gumbel_softmax(fake_day_logits, tau)
            fake_yd = gumbel_softmax(fake_yd_logits, tau)
            rf_fake, aux_dow_f, aux_mon_f, aux_leap_f, aux_dec_f = model.D(
                fake_day, fake_yd, batch["dow"], batch["month"], batch["leap"], batch["decade"]
            )
            g_adv = F.binary_cross_entropy_with_logits(rf_fake, torch.ones_like(rf_fake))
            # AC-GAN: the *generator* is also rewarded for fooling the auxiliary classifiers
            # into agreeing the inputs are correct — this is the condition-enforcement signal.
            g_aux = (
                F.cross_entropy(aux_dow_f, batch["dow"])
                + F.cross_entropy(aux_mon_f, batch["month"])
                + F.cross_entropy(aux_leap_f, batch["leap"])
                + F.cross_entropy(aux_dec_f, batch["decade"])
            )
            g_loss = g_adv + aux_weight * g_aux
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            running_d += d_loss.item()
            running_g += g_loss.item()
            n_batches += 1
            step += 1

            if step % log_every == 0:
                ckpt.log(TrainStats(
                    epoch=epoch, step=step,
                    train_loss=g_loss.item(),
                    extra={"d_loss": d_loss.item(), "tau": tau},
                ))

        val = validate(model, val_loader, sample_logits_fn=model.sample_logits, n_samples=val_n_samples)
        improved = ckpt.maybe_save(model, val["val_joint_compliance"])
        ckpt.log(TrainStats(
            epoch=epoch, step=step,
            train_loss=running_g / max(1, n_batches),
            val_joint_compliance=val["val_joint_compliance"],
            val_validity=val["val_validity"],
            val_acc_dow=val["val_acc_dow"],
            val_acc_leap=val["val_acc_leap"],
            secs=time.time() - start,
            extra={"d_loss": running_d / max(1, n_batches), "tau": tau, "improved": improved},
        ))
        print(
            f"[cgan] ep {epoch:02d} g={running_g/max(1,n_batches):.3f} d={running_d/max(1,n_batches):.3f} "
            f"val_joint={val['val_joint_compliance']:.4f} val_dow={val['val_acc_dow']:.4f}"
            + (" *" if improved else "")
        )
    return ckpt.best_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.txt")
    p.add_argument("--out", default="runs/cgan")
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
