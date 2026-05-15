"""cGAN training loop — projection discriminator + day-of-week compliance.

Use either as a script:

    python -m model.cgan.train --data data/data.txt --epochs 60 --out runs/cgan

or by importing :py:func:`train` from a notebook.

The generator is trained with two signals:

* a **projection-discriminator hinge loss** (the adversarial / diversity term),
  and
* an **auxiliary day-of-week compliance loss**. The adversarial signal alone
  does not reliably transmit the modular day-of-week constraint, so — in the
  spirit of AC-GAN's auxiliary classifier, but using the calendar as an exact
  oracle instead of a learned classifier (which provably cannot acquire the
  weekday function) — we add a differentiable term that rewards the
  generator's soft output for landing on fully-compliant (day, year-digit)
  cells.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..common.format import joint_condition_id
from ..common.seed import set_seed
from ..common.training import (
    CheckpointWriter,
    TrainStats,
    make_loaders,
    validate,
)
from ..common.tokenizer import N_DAY, N_YEAR_DIGIT
from ..common.valid_mask import build_compliance_table
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
    epochs: int = 60,
    batch_size: int = 1024,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    comp_weight: float = 1.0,
    z_dim: int = 64,
    cond_dim: int = 96,
    hidden: int = 384,
    tau_start: float = 0.7,
    tau_end: float = 0.3,
    grad_clip: float = 5.0,
    val_n_samples: int = 4000,
    device: str = "cuda",
    seed: int = 42,
    log_every: int = 50,
) -> Path:
    set_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    train_loader, val_loader, _, _ = make_loaders(data_path, batch_size=batch_size, seed=seed)

    model = CGAN(z_dim=z_dim, cond_dim=cond_dim, hidden=hidden).to(device_t)
    opt_g = torch.optim.Adam(model.G.parameters(), lr=lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(model.D.parameters(), lr=lr_d, betas=(0.0, 0.9))

    # Exact day-of-week compliance oracle, indexed by joint-condition id.
    comp_table = torch.tensor(build_compliance_table(), dtype=torch.float32, device=device_t)

    skipped_batches = 0
    ckpt = CheckpointWriter(out_dir, model_name="cgan")
    step = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        tau = gumbel_anneal(epoch, epochs, tau_start, tau_end)
        running_d, running_g, running_comp = 0.0, 0.0, 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device_t) for k, v in batch.items()}
            B = batch["dow"].shape[0]
            real_day = F.one_hot(batch["day"], N_DAY).float()
            real_yd = F.one_hot(batch["year_digit"], N_YEAR_DIGIT).float()
            cond = (batch["dow"], batch["month"], batch["leap"], batch["decade"])
            joint = joint_condition_id(*cond)
            comp_m = comp_table[joint]                       # [B, 31, 10]

            # ---- discriminator step (hinge loss) -----------------------
            with torch.no_grad():
                z = model.G.sample_z(B, device_t)
                fl_d, fl_y = model.G(z, *cond)
                fake_day = gumbel_softmax(fl_d, tau)
                fake_yd = gumbel_softmax(fl_y, tau)
            d_real = model.D(real_day, real_yd, *cond)
            d_fake = model.D(fake_day, fake_yd, *cond)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            if not torch.isfinite(d_loss):
                skipped_batches += 1
                step += 1
                continue
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.D.parameters(), grad_clip)
            opt_d.step()

            # ---- generator step ----------------------------------------
            z = model.G.sample_z(B, device_t)
            fl_d, fl_y = model.G(z, *cond)
            fake_day = gumbel_softmax(fl_d, tau)
            fake_yd = gumbel_softmax(fl_y, tau)
            g_adv = -model.D(fake_day, fake_yd, *cond).mean()
            # Expected day-of-week compliance of the generator's soft output.
            soft_day = F.softmax(fl_d, dim=-1)
            soft_yd = F.softmax(fl_y, dim=-1)
            exp_comp = torch.einsum("bd,by,bdy->b", soft_day, soft_yd, comp_m)
            comp_loss = -(exp_comp + 1e-8).log().mean()
            g_loss = g_adv + comp_weight * comp_loss
            if not torch.isfinite(g_loss):
                skipped_batches += 1
                step += 1
                continue
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.G.parameters(), grad_clip)
            opt_g.step()

            running_d += d_loss.item()
            running_g += g_loss.item()
            running_comp += comp_loss.item()
            n_batches += 1
            step += 1
            if step % log_every == 0:
                ckpt.log(TrainStats(
                    epoch=epoch, step=step, train_loss=g_loss.item(),
                    extra={"d_loss": d_loss.item(), "comp": comp_loss.item(), "tau": tau},
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
            extra={"d_loss": running_d / max(1, n_batches),
                   "comp": running_comp / max(1, n_batches),
                   "tau": tau, "improved": improved},
        ))
        print(
            f"[cgan] ep {epoch:02d} g={running_g/max(1,n_batches):.3f} "
            f"d={running_d/max(1,n_batches):.3f} comp={running_comp/max(1,n_batches):.3f} "
            f"val_joint={val['val_joint_compliance']:.4f} val_dow={val['val_acc_dow']:.4f} "
            f"skipped={skipped_batches}"
            + (" *" if improved else "")
        )
    return ckpt.best_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.txt")
    p.add_argument("--out", default="runs/cgan")
    p.add_argument("--epochs", type=int, default=60)
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
