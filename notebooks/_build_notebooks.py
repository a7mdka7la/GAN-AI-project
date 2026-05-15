"""Generate the 5 Colab-friendly notebooks for this repo.

Run once with ``python notebooks/_build_notebooks.py``. Each notebook is
~10 cells, all parametrised, so we keep them in sync via this generator
rather than hand-editing JSON. The cells use a Colab-aware bootstrap that
clones or finds the repo, installs nothing (PyTorch is preinstalled on
Colab), and trains the chosen model.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = Path(__file__).resolve().parent

GITHUB_PLACEHOLDER = "https://github.com/a7mdka7la/GAN-AI-project.git"


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def bootstrap_cell() -> dict:
    return code(
        f"""# Colab bootstrap: clone the repo if we're on Colab, set the cwd.
import os, sys, subprocess, pathlib
REPO_URL = {GITHUB_PLACEHOLDER!r}
REPO_DIR = pathlib.Path("/content/Assignment2")
if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
    if not REPO_DIR.exists():
        # If the user has already uploaded the zip, prefer that.
        zip_path = pathlib.Path("/content/Assignment2.zip")
        if zip_path.exists():
            subprocess.run(["unzip", "-q", str(zip_path), "-d", "/content"], check=True)
        else:
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    os.chdir(str(REPO_DIR))
else:
    # Local: cd to the repo (this notebooks/ dir's parent).
    os.chdir(str(pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()))
sys.path.insert(0, os.getcwd())
print("cwd =", os.getcwd())
print("files:", os.listdir(".")[:10])
"""
    )


def training_cell(model_module: str) -> dict:
    return code(
        f"""# Train. epochs/batch_size can be raised if you have Colab time budget.
from {model_module}.train import train

best_path = train(
    data_path="data/data.txt",
    out_dir="runs/{model_module.split('.')[-1]}",
    epochs=60,
    batch_size=1024,
    device="cuda",
    seed=42,
)
print("Best checkpoint:", best_path)
"""
    )


def promote_cell(model_name: str) -> dict:
    return code(
        f"""# Promote the best checkpoint to the location predict.py expects.
import shutil, pathlib
src = pathlib.Path("runs/{model_name}/{model_name}_best.pt")
dst = pathlib.Path("model/weights/{model_name}.pt")
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)
print(f"Copied {{src}} -> {{dst}}")
"""
    )


def eval_cell(model_name: str) -> dict:
    return code(
        f"""# Quick smoke-eval on example_input.txt
import subprocess, sys
out = subprocess.run(
    [sys.executable, "-m", "model.evaluate",
     "--input", "data/example_input.txt", "--model", "{model_name}"],
    capture_output=True, text=True,
)
print(out.stdout)
if out.returncode != 0:
    print("STDERR:", out.stderr, file=sys.stderr)
"""
    )


def plot_cell(run_dir: str) -> dict:
    return code(
        f"""# Plot training loss + val joint compliance.
import json, matplotlib.pyplot as plt, pathlib
rows = [json.loads(l) for l in pathlib.Path("runs/{run_dir}/log.jsonl").read_text().splitlines() if l.strip()]
ep_rows = [r for r in rows if r.get("val_joint_compliance") is not None]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot([r["step"] for r in rows], [r["train_loss"] for r in rows], label="train_loss")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].set_title("Training loss"); ax[0].legend()
ax[1].plot([r["epoch"] for r in ep_rows], [r["val_joint_compliance"] for r in ep_rows], label="val_joint")
ax[1].plot([r["epoch"] for r in ep_rows], [r["val_acc_dow"] for r in ep_rows], label="val_acc_dow")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("rate"); ax[1].set_title("Validation compliance"); ax[1].legend()
plt.tight_layout(); plt.show()
"""
    )


def build_train_notebook(model_name: str, friendly: str) -> dict:
    module = f"model.{model_name}"
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            md(f"# Train {friendly}\n\nDataset: `data/data.txt`. Saves the best checkpoint by validation joint compliance.\n\n**Reproducibility:** `seed=42` set in every entry point.\n"),
            bootstrap_cell(),
            md("## Train"),
            training_cell(module),
            md("## Promote checkpoint"),
            promote_cell(model_name),
            md("## Quick smoke-evaluation on example_input.txt"),
            eval_cell(model_name),
            md("## Training/validation curves"),
            plot_cell(model_name),
        ],
    }


def build_eval_notebook() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "cells": [
            md("# Evaluate all 4 models\n\nLoads each `model/weights/<name>.pt` and computes the metric table on `data/example_input.txt`.\n"),
            bootstrap_cell(),
            code(
                """import subprocess, sys
for name in ["cgan", "cvae", "transformer", "diffusion"]:
    print(f"\\n=== {name.upper()} ===")
    out = subprocess.run(
        [sys.executable, "-m", "model.evaluate",
         "--input", "data/example_input.txt", "--model", name],
        capture_output=True, text=True,
    )
    print(out.stdout)
    if out.returncode != 0:
        print("STDERR:", out.stderr[-2000:], file=sys.stderr)
"""
            ),
            md("## Pick the best model as the default for predict.py"),
            code(
                """# After looking at the table above, pick the best model and write it to
# model/weights/active_model.txt, plus copy its checkpoint to model/weights/<name>.pt.
import pathlib
BEST = "transformer"   # <-- edit after seeing results
pathlib.Path("model/weights/active_model.txt").write_text(BEST)
print(f"Active model set to {BEST}")
"""
            ),
        ],
    }


def main() -> None:
    notebooks = {
        "01_train_cgan.ipynb":        build_train_notebook("cgan", "Conditional GAN (AC-GAN)"),
        "02_train_cvae.ipynb":        build_train_notebook("cvae", "Conditional VAE"),
        "03_train_transformer.ipynb": build_train_notebook("transformer", "Conditional Transformer"),
        "04_train_diffusion.ipynb":   build_train_notebook("diffusion", "Conditional Diffusion (DDPM + CFG)"),
        "05_eval_all.ipynb":          build_eval_notebook(),
    }
    for fname, nb in notebooks.items():
        path = NB_DIR / fname
        path.write_text(json.dumps(nb, indent=1))
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
