# Dates Generator — DSAI 490 Assignment 2

Generate a date `D-M-YYYY` that satisfies four conditions
`[DOW] [MON] [True|False] [DEC]` using four conditional generative models:

| Family               | Origin         | Module                |
|----------------------|----------------|-----------------------|
| Conditional GAN      | course (req.)  | `model/cgan/`         |
| Conditional VAE      | course         | `model/cvae/`         |
| Conditional Transformer (small GPT) | outside course | `model/transformer/`  |
| Conditional Diffusion (DDPM + CFG)  | outside course | `model/diffusion/`    |

All four share the same condition encoder and produce a pair of categorical
logits ``(day ∈ 1..31, year_last_digit ∈ 0..9)``. The month and the first three
year digits come straight from the input conditions. A learned model only has
to figure out the day-of-week and leap-year conditions; everything else is
masked at inference time so every emitted date is parseable and in
`[1-1-1800, 31-12-2200]`.

## Quick start (Google Colab)

1. Open `notebooks/01_train_cgan.ipynb` (or any of the four training
   notebooks) in Colab. The first cell clones this repo automatically.
2. Run all cells with a GPU runtime — each model trains in ~5–15 min on a
   T4 and saves its best checkpoint to `runs/<model>/<model>_best.pt`,
   which the notebook then copies to `model/weights/<model>.pt`.
3. After running all four training notebooks, open
   `notebooks/05_eval_all.ipynb`, run it, and choose the best model — that
   becomes the default for `predict.py`.

## Inference (the assignment's mandated command)

```
python predict.py -i $path_to_input_file -o $path_to_output_file
```

Run from `repo/model/`. Default model is the one named in
`model/weights/active_model.txt` (set after training). Pass
`--model {cgan,cvae,transformer,diffusion}` to override.

## Repository layout

```
repo/
├── data/{data.txt, example_input.txt}
├── model/
│   ├── common/        # tokenizer, dataset, condition encoder, valid_mask,
│   │                  # metrics, sampling, training helpers, seed
│   ├── cgan/          # AC-GAN with Gumbel-softmax outputs
│   ├── cvae/          # Categorical-decoder VAE with KL warmup
│   ├── transformer/   # 4-layer GPT, 7-token sequence
│   ├── diffusion/     # DDPM on one-hots with classifier-free guidance
│   ├── weights/       # *.pt files + active_model.txt
│   ├── predict.py     # CLI entry point per assignment
│   └── evaluate.py    # Compliance + diversity table
├── notebooks/         # one per model + 05_eval_all.ipynb
├── tests/             # pure-Python unit tests (no torch needed)
├── environment.yml    # conda spec
├── PLAN.md            # one-page design summary
└── README.md
```

## Evaluation metric

Since multiple dates satisfy each condition, we don't use exact-match
accuracy. Instead we report:

* **per-condition compliance** for each of the four conditions,
* **joint compliance** — fraction of generated dates that satisfy all four,
* **validity** — fraction that are parseable, in-range, and calendar-legal
  (should be ~1.0 thanks to the validity mask),
* **diversity entropy** — Shannon entropy of dates per distinct condition
  tuple across N samples; used to catch cGAN mode-collapse.

## Reproducibility

* `model.common.seed.set_seed(42)` in every entry point seeds Python,
  NumPy, and Torch (CPU + CUDA).
* `from __future__ import annotations` + type hints in every module.
* DataLoaders are shuffled with a seeded `torch.Generator`.

## Tests

Pure-Python tests (no torch) cover the tokenizer, validity masks, metrics,
and a roundtrip of every line in `data/data.txt`.

```bash
python -m unittest discover tests -v
```

## Output format

Exact match to `data/data.txt` line format (unpadded day/month):

```
[WED] [JAN] [False] [180] 1-1-1800
[SAT] [DEC] [True] [219] 31-12-2196
```
