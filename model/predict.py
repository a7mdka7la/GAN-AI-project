"""Inference CLI required by the assignment.

Usage (the form mandated by the assignment):

    python predict.py -i $path_to_input_file -o $path_to_output_file

The input file has lines like ``[WED] [JAN] [False] [180]`` (one per row).
The output file has matching lines with a date appended:
``[WED] [JAN] [False] [180] 5-1-1809``, preserving input order.

By default we use the model checkpoint that achieved the best validation
joint-compliance during training. The active choice is recorded in
``model/weights/active_model.txt`` (a single token: ``cgan``, ``cvae``,
``transformer``, or ``diffusion``) and can be overridden with ``--model``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Sequence

# Make this file runnable both as ``python predict.py`` (script, cwd=repo/model)
# and ``python -m model.predict`` (module, cwd=repo). Either way we want
# ``from model.common...`` to resolve.
_HERE = Path(__file__).resolve().parent              # repo/model
_REPO_ROOT = _HERE.parent                            # repo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from model.common.format import Condition, format_output_line, parse_input_line  # noqa: E402
from model.common.sampling import sample_day_and_digit                            # noqa: E402
from model.common.seed import set_seed                                            # noqa: E402
from model.common.tokenizer import ConditionTokenizer                             # noqa: E402


_MODULE_DIR = Path(__file__).resolve().parent
_WEIGHTS_DIR = _MODULE_DIR / "weights"
_ACTIVE_FILE = _WEIGHTS_DIR / "active_model.txt"
_DEFAULT_FALLBACK = "transformer"


def _active_model_name() -> str:
    if _ACTIVE_FILE.exists():
        name = _ACTIVE_FILE.read_text().strip().lower()
        if name in {"cgan", "cvae", "transformer", "diffusion"}:
            return name
    return _DEFAULT_FALLBACK


def _build_model(name: str, device: torch.device, tok: ConditionTokenizer):
    """Instantiate the requested model and load weights from ``model/weights/<name>.pt``."""
    if name == "cgan":
        from model.cgan.model import CGAN
        m = CGAN()
    elif name == "cvae":
        from model.cvae.model import CVAE
        m = CVAE()
    elif name == "transformer":
        from model.transformer.model import ConditionalTransformer
        m = ConditionalTransformer(tokenizer=tok)
    elif name == "diffusion":
        from model.diffusion.model import CondDiffusion
        m = CondDiffusion()
    else:
        raise ValueError(f"unknown model {name!r}")

    weight_path = _WEIGHTS_DIR / f"{name}.pt"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"missing weights at {weight_path}. "
            f"Train the model first (see notebooks/) and copy the checkpoint to model/weights/{name}.pt."
        )
    state = torch.load(weight_path, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    return m


def _sample_fn_for(name: str, model, tok: ConditionTokenizer):
    """Return a callable ``(dow, month, leap, decade) -> (logits_day, logits_yd)``."""
    if name == "transformer":
        def fn(*, dow, month, leap, decade):
            B = dow.shape[0]
            prompts = torch.stack([
                torch.full((B,), tok.ids.bos, device=dow.device, dtype=torch.long),
                dow + tok.ids.dow_start,
                month + tok.ids.month_start,
                leap + tok.ids.leap_start,
                decade + tok.ids.decade_start,
            ], dim=1)
            return model.sample_logits(prompts)
        return fn
    # All other models expose ``sample_logits(dow, month, leap, decade)`` directly.
    return lambda *, dow, month, leap, decade: model.sample_logits(
        dow=dow, month=month, leap=leap, decade=decade
    )


def _read_conditions(path: Path) -> list[Condition]:
    conds: list[Condition] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            cond, _ = parse_input_line(stripped)
            conds.append(cond)
    return conds


def _conds_to_tensors(
    conds: Sequence[Condition], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "dow":    torch.tensor([c.dow for c in conds], device=device, dtype=torch.long),
        "month":  torch.tensor([c.month for c in conds], device=device, dtype=torch.long),
        "leap":   torch.tensor([c.leap for c in conds], device=device, dtype=torch.long),
        "decade": torch.tensor([c.decade for c in conds], device=device, dtype=torch.long),
    }


def predict_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str | None = None,
    batch_size: int = 512,
    device: str = "auto",
    seed: int = 42,
) -> None:
    set_seed(seed)
    name = (model_name or _active_model_name()).lower()
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)
    tok = ConditionTokenizer()
    model = _build_model(name, device_t, tok)
    fn = _sample_fn_for(name, model, tok)

    conds = _read_conditions(Path(input_path))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out_lines: list[str] = []
    for i in range(0, len(conds), batch_size):
        chunk = conds[i : i + batch_size]
        tensors = _conds_to_tensors(chunk, device_t)
        logits_day, logits_yd = fn(**tensors)
        # Bring logits to CPU for the masking+sampling step (small batches; cheap).
        logits_day = logits_day.cpu()
        logits_yd = logits_yd.cpu()
        samples = sample_day_and_digit(logits_day, logits_yd, chunk, generator=generator)
        for cond, (day, yd) in zip(chunk, samples):
            year = cond.decade_int * 10 + yd
            date = _dt.date(year, cond.month + 1, day)
            out_lines.append(format_output_line(cond, date))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate dates that satisfy the given condition lines."
    )
    p.add_argument("-i", "--input", required=True, help="Path to input file (conditions only)")
    p.add_argument("-o", "--output", required=True, help="Path to output file")
    p.add_argument(
        "--model",
        choices=["cgan", "cvae", "transformer", "diffusion"],
        default=None,
        help="Model to use. Defaults to weights/active_model.txt (else 'transformer').",
    )
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    predict_file(
        input_path=args.input, output_path=args.output,
        model_name=args.model, batch_size=args.batch_size,
        device=args.device, seed=args.seed,
    )


if __name__ == "__main__":
    main()
