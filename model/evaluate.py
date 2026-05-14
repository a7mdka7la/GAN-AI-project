"""End-to-end evaluation: run a generated output file against the assignment's
metric and print a summary table.

Two modes:

* ``--predictions <file>`` — evaluate any file that follows the data.txt
  format. Useful for grading our own ``predict.py`` output.
* ``--input <input_file>`` + ``--model <name>`` — run predict.py internally
  and evaluate the resulting outputs.

Also computes a diversity-entropy score by sampling N dates per distinct
input condition tuple (helps spot mode-collapse, especially for the cGAN).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from model.common.format import parse_input_line, Condition                     # noqa: E402
from model.common.metrics import diversity_entropy, evaluate                    # noqa: E402
from model.common.sampling import sample_day_and_digit                          # noqa: E402
from model.common.seed import set_seed                                          # noqa: E402
from model.common.tokenizer import ConditionTokenizer                           # noqa: E402
from model.predict import _build_model, _conds_to_tensors, _sample_fn_for, predict_file  # noqa: E402


def _read_pred_file(path: Path) -> list[tuple[Condition, _dt.date | None]]:
    out: list[tuple[Condition, _dt.date | None]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cond, date = parse_input_line(line)
            out.append((cond, date))
    return out


def _diversity_for_model(
    name: str,
    input_path: Path,
    samples_per_cond: int = 10,
    device: str = "auto",
    seed: int = 42,
    batch_size: int = 512,
) -> float:
    """Sample N dates per distinct input condition and compute mean entropy."""
    set_seed(seed)
    device_t = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                            ("cpu" if device == "auto" else device))
    tok = ConditionTokenizer()
    model = _build_model(name, device_t, tok)
    fn = _sample_fn_for(name, model, tok)

    distinct: list[Condition] = []
    seen: set[Condition] = set()
    for cond, _ in _read_pred_file(input_path):
        if cond not in seen:
            seen.add(cond)
            distinct.append(cond)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    bucket: dict[Condition, list[_dt.date]] = defaultdict(list)
    # For each of N rounds, sample one date per distinct condition.
    for _ in range(samples_per_cond):
        for i in range(0, len(distinct), batch_size):
            chunk = distinct[i : i + batch_size]
            tensors = _conds_to_tensors(chunk, device_t)
            logits_day, logits_yd = fn(**tensors)
            samples = sample_day_and_digit(
                logits_day.cpu(), logits_yd.cpu(), chunk, generator=generator
            )
            for cond, (day, yd) in zip(chunk, samples):
                year = cond.decade_int * 10 + yd
                try:
                    bucket[cond].append(_dt.date(year, cond.month + 1, day))
                except ValueError:
                    pass
    return diversity_entropy(list(bucket.values()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", help="Evaluate an existing predictions file")
    p.add_argument("--input", help="Run a model on this input file first, then evaluate")
    p.add_argument(
        "--model", choices=["cgan", "cvae", "transformer", "diffusion"], default=None,
        help="(with --input) which model to run",
    )
    p.add_argument("--diversity_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.predictions:
        pairs = _read_pred_file(Path(args.predictions))
        rep = evaluate(pairs)
        print(rep.as_table())
        return

    if not args.input:
        raise SystemExit("Must pass either --predictions or --input")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    predict_file(args.input, tmp_path, model_name=args.model,
                 device=args.device, seed=args.seed)
    pairs = _read_pred_file(tmp_path)
    rep = evaluate(pairs)
    if args.model is not None:
        rep.diversity_entropy_mean = _diversity_for_model(
            args.model, Path(args.input),
            samples_per_cond=args.diversity_samples,
            device=args.device, seed=args.seed,
        )
    print(rep.as_table())


if __name__ == "__main__":
    main()
