# Assignment 2 — Dates Generator

**Author:** Ahmed Kahla &nbsp;&nbsp;|&nbsp;&nbsp; **ID:** 202202231

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

All four models share the condition encoder — four factored embeddings
(DOW/MON/LEAP/DEC) plus one joint-condition embedding fused into a 96-d
vector — and the two output heads (day logits 31, year-digit logits 10).
The joint embedding gives each `(dow,month,leap,decade)` tuple a dedicated
learnable vector, which is what lets the models memorise the day-of-week
function instead of trying to factor it across independent embeddings.
The models differ only in how the logits are produced.

| Model              | What is novel                                                                            | Params |
|--------------------|-------------------------------------------------------------------------------------------|--------|
| **cGAN**           | Generator + projection-discriminator critic (Miyato & Koyama 2018) on Gumbel-softmax soft one-hots, spectral-normalised, hinge loss. The adversarial term alone cannot transmit the modular day-of-week constraint, so an auxiliary differentiable calendar-compliance loss rewards the generator's soft output for landing on fully-compliant `(day, year-digit)` cells. | ~4.6 M |
| **cVAE**           | Encoder over (one-hot day, one-hot year-digit, condition); 16-d latent; free-bits KL (0.3-nat/dim floor, β annealed up to 0.5) to avoid posterior collapse on a tiny output space. | ~1.8 M |
| **Transformer**    | 5-layer causal decoder, d_model=160, 4 heads, d_ff=384, weight-tied head. Joint-condition embedding added at every position; cross-entropy on the two output positions (year-digit then day) only. | ~2.3 M |
| **Diffusion**      | Categorical diffusion on the 41-dim concatenated one-hot `[day(31)||year_digit(10)]`; x0-prediction denoiser trained with cross-entropy; classifier-free guidance (p_drop = 0.1, w = 3.0); DDIM 20-step sampler. | ~1.6 M |

The choice of *minimum heads* is what makes the GAN viable here — a vanilla
sequence-token cGAN would have to backprop through a categorical sample,
and on such a small joint space (≤ 310 cells per condition) mode-collapse
is severe without the auxiliary calendar-compliance signal.

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

### 4.1 Compliance on `example_input.txt`

| Model       | validity | acc_dow | acc_month | acc_leap | acc_decade | **joint** | diversity (nats) |
|-------------|----------|---------|-----------|----------|------------|-----------|------------------|
| cGAN        | 1.000    | 0.7188  | 1.000     | 1.000    | 1.000      | **0.7188** | 0.88            |
| cVAE        | 1.000    | 0.7420  | 1.000     | 1.000    | 1.000      | **0.7420** | 2.12            |
| Transformer | 1.000    | 0.3263  | 1.000     | 1.000    | 1.000      | **0.3263** | 2.03            |
| Diffusion   | 1.000    | 0.9959  | 1.000     | 1.000    | 1.000      | **0.9959** | 2.02            |

`acc_month`, `acc_leap` and `acc_decade` are exactly 1.000 because those three
conditions are enforced deterministically by the sample-time validity mask, so
joint compliance equals `acc_dow` — the day-of-week is the only learned term.
**Conditional Diffusion is the strongest model** and is set as the default in
`model/weights/active_model.txt`. The cGAN's low diversity entropy (0.88 vs
~2.0 for the others) is the expected signature of GAN mode-collapse on a small
per-condition output space.

---

## 5. Analysis

### What worked

* The two-head categorical output. Letting the deterministic-condition
  mask handle month/decade/leap-year + calendar legality means every
  output is mechanically valid; the *learning* task reduces to "given the
  decade and the four conditions, pick a day-of-week-consistent (day,
  year_digit) pair". For the dataset's size this lets all four models
  converge quickly on a single T4.

* Projection discriminator + calendar-compliance loss. The projection
  critic supplies a per-condition validity gradient, and the differentiable
  compliance loss supplies the exact day-of-week signal the adversarial game
  alone cannot transmit. An earlier AC-GAN-style auxiliary day-of-week
  classifier was tried first and failed — a learned classifier provably
  cannot acquire the modular weekday function — so it was replaced by the
  calendar as an exact oracle.

### Common failure modes

* **cGAN mode-collapse on rare conditions.** Decade `[220]` and leap-year
  combinations that contain only a single valid year (e.g. `[219]` with
  `[True]` leap) make the generator collapse to the one valid date — the
  diversity-entropy metric drops to 0. *Discussed; mitigation = condition
  resampling proportional to inverse frequency.*

* **Diffusion under-trains the year-digit head.** Because the year-digit
  one-hots are heavily constrained by decade+leap, the denoiser has very
  little to learn and small noise can flip the argmax. CFG with w = 3 was
  needed to keep `acc_dow` high on the test split — and even so, Diffusion
  is by a wide margin the best model (joint 0.996).

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
* The Transformer generates year-digit first and then the day token
  conditioned on the *argmax* year-digit rather than a sampled one. For a
  joint of ≤ 310 cells per condition this is essentially equivalent, but
  it does sacrifice a small amount of joint diversity, and the Transformer
  remains the weakest model (joint 0.33) — its loss plateaus before it
  fully memorises the weekday function.
