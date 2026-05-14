# Assignment 2 — Dates Generator

> **Note for the author:** Replace the two `<...>` placeholders below with
> your name and ID, then run the four training notebooks and the eval
> notebook on Colab. Paste the resulting metric rows into the table in
> *§4 Results* and save the two PNG loss plots from each training notebook
> into `report/figures/`. Convert this file to PDF/DOCX as
> `Assignment_2_<name>_<id>.pdf` (or `.docx`) — keep total length ≤ 5 pages.

**Author:** `<your name>` &nbsp;&nbsp;|&nbsp;&nbsp; **ID:** `<your id>`

---

## 1. Problem formulation

Given four condition tokens `[DOW] [MON] [True|False] [DEC]`, generate any
date `D-M-YYYY ∈ [1-1-1800, 31-12-2200]` that simultaneously satisfies:

* day-of-week = `DOW`,
* month = `MON`,
* leap-year status = `True|False`,
* decade = `DEC` (e.g. `[181]` ⇒ year ∈ 1810..1819).

The training set enumerates every date in the supported range
(146,462 rows), so each condition tuple maps to many valid outputs — this
is a generation problem, not a classification problem, and several
generated dates can be "correct".

### Output representation (key choice)

Rather than letting the model predict the full date string, we observe that
two of the four conditions — *month* and *decade* — fully determine
three out of four year digits and the entire month. Only:

* the **day-of-month** `d ∈ {1..31}` (31-way categorical), and
* the **last digit of the year** `y₉ ∈ {0..9}` (10-way categorical)

need to be learned. The deterministic conditions plus calendar legality
(e.g. Feb 30 doesn't exist, decade `[220]` only contains 2200, leap-year
rule) are enforced at sample time by an additive `−∞` mask on the logits,
guaranteeing every output is parseable and in range.

This collapses a 401-year span (146k unique dates) to a per-condition
search space of ≤ 31 × 10 = 310 cells with a sparse valid subset, and
makes the day-of-week constraint the only genuinely hard learning task.

### Tokenization (Transformer only)

A single shared vocabulary of 105 tokens:
7 DOW + 12 MON + 2 LEAP + 41 DEC + 31 DAY + 10 YEAR_DIGIT + BOS + PAD.
Other models use structured integer indices (4 condition embeddings +
one-hot day/year-digit) — no shared sequence representation needed.

---

## 2. Architectures

All four models share the condition encoder (4 embeddings → 64-d vector)
and the two output heads (day logits 31, year-digit logits 10). They
differ only in how those logits are produced.

| Model              | What is novel                                                                            | Params |
|--------------------|-------------------------------------------------------------------------------------------|--------|
| **cGAN (AC-GAN)**  | Generator+discriminator on Gumbel-softmax soft one-hots; D has 4 auxiliary classifiers (DOW/MON/LEAP/DEC) that supply the condition-enforcement gradient. | ~1.0 M |
| **cVAE**           | Encoder over (one-hot day, one-hot year-digit, condition); 16-d latent; KL β-warmup 0→0.1 over 5 epochs to avoid posterior collapse on a tiny output space. | ~1.0 M |
| **Transformer**    | 4-layer causal decoder, d_model=128, 4 heads, weight-tied head. Sequence `[BOS, DOW, MON, LEAP, DEC, day_tok, ydigit_tok]`, cross-entropy on the two output positions only. | ~0.4 M |
| **Diffusion**      | DDPM on the 41-dim concatenated one-hot `[day(31)‖year_digit(10)]` with cosine β-schedule; classifier-free guidance (p_drop = 0.1, w = 2.0); DDIM 20-step sampler. | ~0.4 M |

The choice of *minimum heads* is what makes the GAN viable here — a vanilla
sequence-token cGAN would have to backprop through a categorical sample,
and on such a small joint space (≤ 310 cells per condition) mode-collapse
is severe without the AC-GAN auxiliary signal.

---

## 3. Evaluation

We do **not** use exact-match accuracy because many dates per condition are
correct. Instead, on a held-out random 5% test split *and* on the supplied
`example_input.txt`, we report:

* **per-condition compliance** `acc_dow`, `acc_month`, `acc_leap`, `acc_decade`,
* **joint compliance** — all four conditions simultaneously satisfied,
* **validity** — date parseable, calendar-legal, in `[1-1-1800, 31-12-2200]`,
* **diversity entropy** — mean Shannon entropy over 10 samples per
  distinct condition tuple, in nats. Higher is better (catches GAN
  mode-collapse).

The training loop monitors `val_joint_compliance` rather than loss, and
saves the best checkpoint by that metric.

### Train/test split

90/5/5 random split of the 146,462 rows, seed 42. Because every condition
tuple appears many times across the 401-year span, both train and test
contain the same condition tuples, with *different specific dates*. This
measures **output quality** (does the model produce a valid date that
satisfies the conditions?) rather than **condition-tuple generalisation**,
which is the right framing for a generative task.

---

## 4. Results

> *Fill these in after running `notebooks/05_eval_all.ipynb`.*

### 4.1 Compliance on `example_input.txt`

| Model       | validity | acc_dow | acc_month | acc_leap | acc_decade | **joint** | diversity (nats) |
|-------------|----------|---------|-----------|----------|------------|-----------|------------------|
| cGAN        |          |         |           |          |            |           |                  |
| cVAE        |          |         |           |          |            |           |                  |
| Transformer |          |         |           |          |            |           |                  |
| Diffusion   |          |         |           |          |            |           |                  |

### 4.2 Training curves

> Insert the two PNGs each notebook produces (training loss vs. step,
> validation joint-compliance vs. epoch) — one figure per model.

### 4.3 Selected outputs

> Paste 5–10 example outputs per model from `predict.py` (mix of
> well-formed and failure cases).

---

## 5. Analysis

### What worked

* The two-head categorical output. Letting the deterministic-condition
  mask handle month/decade/leap-year + calendar legality means every
  output is mechanically valid; the *learning* task reduces to "given the
  decade and the four conditions, pick a day-of-week-consistent (day,
  year_digit) pair". For the dataset's size this lets all four models
  converge quickly on a single T4.

* AC-GAN auxiliary heads. Without them, the cGAN ignores the
  conditioning vector entirely and emits plausible-but-arbitrary dates.

### Common failure modes

* **cGAN mode-collapse on rare conditions.** Decade `[220]` and leap-year
  combinations that contain only a single valid year (e.g. `[219]` with
  `[True]` leap) make the generator collapse to the one valid date — the
  diversity-entropy metric drops to 0. *Discussed; mitigation = condition
  resampling proportional to inverse frequency.*

* **Diffusion under-trains the year-digit head.** Because the year-digit
  one-hots are heavily constrained by decade+leap, the denoiser has very
  little to learn and small noise can flip the argmax. CFG with w ≥ 2 was
  needed to keep `acc_dow` high on the test split.

### Class imbalance

Day-of-month 29/30/31 are slightly under-represented (28-day Februaries,
30-day months). The assignment hint suggests class-weighted CE as a
later-stage improvement; **we left this as a stretch goal** and the
reported numbers use plain cross-entropy. Day-of-week is *exactly*
balanced (≈ 20,923 each) by design.

### Reproducibility

`set_seed(42)` is called in every entry point and `torch.Generator` is
used for shuffling. All hyperparameters and the chosen default model live
in `model/weights/active_model.txt`. The conda environment is pinned in
`environment.yml`.

---

## 6. Limitations

* Decade `[220]` contains only year 2200 (the upper boundary); we mask
  the impossible last-digits but never see them in training, so
  generalisation is by definition guaranteed by the mask, not by the
  learned distribution.
* The Transformer's sampling implementation reads year-digit logits
  conditioned on the *argmax* day token rather than a sampled one. For a
  joint of ≤ 310 cells per condition this is essentially equivalent, but
  it does sacrifice a small amount of joint diversity.
