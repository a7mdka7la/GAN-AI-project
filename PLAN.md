# Conditional Date Generator — Implementation Plan

This file is a one-shot reference; the design as approved in the brainstorming
conversation governs.

## Goal
Train four conditional generative models that, given
`[day_of_week] [month] [leap_year] [decade]`, produce a date `D-M-YYYY` in
`[1-1-1800, 31-12-2200]` satisfying all four conditions.

## Architecture set
- **cGAN** (course / required) — AC-GAN style with Gumbel-softmax outputs.
- **cVAE** (course) — categorical decoder + β-warmup KL.
- **Conditional Transformer** (outside course) — tiny GPT, two-token autoregressive output.
- **Conditional Diffusion** (outside course) — DDPM on one-hot concat + classifier-free guidance.

## Output representation
Two categorical heads: `day ∈ {1..31}`, `year_last_digit ∈ {0..9}`. Month and the
first three year digits come directly from the input conditions. A `valid_mask`
zeroes-out impossible (day, digit) combinations per condition before sampling so
every generated date is parseable and in-range.

## Repository layout
```
repo/
├── data/{data.txt, example_input.txt}
├── model/
│   ├── common/{tokenizer.py, dataset.py, condition_encoder.py,
│   │           valid_mask.py, metrics.py, seed.py, format.py}
│   ├── cgan/{model.py, train.py}
│   ├── cvae/{model.py, train.py}
│   ├── transformer/{model.py, train.py}
│   ├── diffusion/{model.py, train.py}
│   ├── weights/*.pt
│   ├── predict.py
│   └── evaluate.py
├── notebooks/  # one per model + eval_all
├── tests/      # pure-Python unit tests (no torch)
├── environment.yml
└── README.md
```

## Training defaults
- 90/5/5 random split, seed 42.
- AdamW, batch 1024, 30 epochs, cosine LR.
- Save best checkpoint by validation joint compliance.

## Evaluation
- Per-condition compliance + joint compliance + date validity + diversity entropy.
- Test set + `example_input.txt` (the assignment’s intended grading input).

## predict.py
- `python predict.py -i $in -o $out` (assignment-mandated).
- Default model selected after training (best val joint compliance).
- Optional `--model {cgan,cvae,transformer,diffusion}` and `--seed`.
- Output format: `[DOW] [MON] [True/False] [DEC] D-M-YYYY`, unpadded day/month.

## Reproducibility
- `set_seed(42)` everywhere; `from __future__ import annotations` + type hints.
- `environment.yml` pinned for replicable Colab runs.
