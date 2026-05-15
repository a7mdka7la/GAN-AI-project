"""Pure-Python check: does uniform-random-within-mask sampling give
joint_compliance == dow_compliance? It MUST, because the validity mask
enforces month/leap/decade deterministically. If joint < dow here, there is
a logic bug in valid_mask or metrics (independent of torch / the models)."""
from __future__ import annotations

import datetime as _dt
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.common.format import parse_input_line
from model.common.metrics import condition_compliance, date_validity
from model.common.valid_mask import valid_day_mask, valid_year_digit_mask

random.seed(42)

rows = []
with open(ROOT / "data" / "data.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            cond, _ = parse_input_line(line)
            rows.append(cond)

# Sample 8000 conditions and do uniform-random-within-mask generation.
sample = random.sample(rows, 8000)
n = 0
valid_n = 0
counts = {"dow": 0, "month": 0, "leap": 0, "decade": 0, "joint": 0}
all_false_yd = 0
all_false_day = 0

for cond in sample:
    n += 1
    yd_mask = valid_year_digit_mask(cond)
    valid_yds = [i for i, ok in enumerate(yd_mask) if ok]
    if not valid_yds:
        all_false_yd += 1
        # mimic gumbel-max on all -inf: arbitrary pick
        yd = random.randrange(10)
    else:
        yd = random.choice(valid_yds)

    day_mask = valid_day_mask(cond, yd)
    valid_days = [i for i, ok in enumerate(day_mask) if ok]
    if not valid_days:
        all_false_day += 1
        day_idx = random.randrange(31)
    else:
        day_idx = random.choice(valid_days)

    year = cond.decade_int * 10 + yd
    try:
        date = _dt.date(year, cond.month + 1, day_idx + 1)
    except ValueError:
        date = None

    if not date_validity(date):
        continue
    assert date is not None
    valid_n += 1
    checks = condition_compliance(cond, date)
    for k in ("dow", "month", "leap", "decade"):
        if checks[k]:
            counts[k] += 1
    if all(checks.values()):
        counts["joint"] += 1

print(f"n                  = {n}")
print(f"validity           = {valid_n / n:.4f}")
print(f"acc_dow            = {counts['dow'] / n:.4f}")
print(f"acc_month          = {counts['month'] / n:.4f}")
print(f"acc_leap           = {counts['leap'] / n:.4f}")
print(f"acc_decade         = {counts['decade'] / n:.4f}")
print(f"acc_joint          = {counts['joint'] / n:.4f}")
print(f"all-False yd masks = {all_false_yd}")
print(f"all-False day masks= {all_false_day}")
print()
if abs(counts["dow"] - counts["joint"]) <= 5:
    print("OK: joint ~= dow  -> mask/metric logic is consistent.")
else:
    print("BUG: joint != dow -> a deterministic condition is not being enforced.")
