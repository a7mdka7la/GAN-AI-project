"""Pure-Python tests for the common utilities. Run with ``python -m pytest``
or ``python tests/test_common.py``. No torch required for these tests."""
from __future__ import annotations

import datetime as _dt
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.common.format import (  # noqa: E402
    DAYS_OF_WEEK,
    MONTHS,
    Condition,
    format_output_line,
    is_leap,
    parse_input_line,
)
from model.common.metrics import (  # noqa: E402
    condition_compliance,
    date_validity,
    diversity_entropy,
    evaluate,
    joint_compliance,
)
from model.common.tokenizer import ConditionTokenizer  # noqa: E402
from model.common.valid_mask import (  # noqa: E402
    full_valid_mask,
    valid_day_mask,
    valid_year_digit_mask,
)


class FormatTests(unittest.TestCase):
    def test_roundtrip_dataset_line(self) -> None:
        line = "[WED] [JAN] [False] [180] 1-1-1800"
        cond, date = parse_input_line(line)
        self.assertEqual(cond, Condition(dow=2, month=0, leap=0, decade=0))
        self.assertEqual(date, _dt.date(1800, 1, 1))
        self.assertEqual(format_output_line(cond, date), line)

    def test_unpadded_output(self) -> None:
        cond, date = parse_input_line("[SAT] [DEC] [True] [219] 31-12-2196")
        self.assertIsNotNone(date)
        self.assertEqual(format_output_line(cond, date), "[SAT] [DEC] [True] [219] 31-12-2196")
        # Edge: single-digit day/month should be unpadded
        cond2 = Condition(dow=0, month=2, leap=0, decade=2)  # MAR 1820s (decade index 2 -> 182)
        self.assertEqual(format_output_line(cond2, _dt.date(1825, 3, 5)),
                         "[MON] [MAR] [False] [182] 5-3-1825")

    def test_input_only_line(self) -> None:
        cond, date = parse_input_line("[WED] [JAN] [False] [180]")
        self.assertEqual(cond, Condition(dow=2, month=0, leap=0, decade=0))
        self.assertIsNone(date)

    def test_decade_range_check(self) -> None:
        # decade 221 would extend past 2200 — must be rejected
        with self.assertRaises(ValueError):
            parse_input_line("[WED] [JAN] [False] [221] 1-1-2210")

    def test_leap_year_rule(self) -> None:
        self.assertTrue(is_leap(2000))
        self.assertFalse(is_leap(1900))
        self.assertTrue(is_leap(2004))
        self.assertFalse(is_leap(2001))
        # Edge: 1800 is not leap (centurial but not divisible by 400)
        self.assertFalse(is_leap(1800))


class ValidMaskTests(unittest.TestCase):
    def test_decade_220_only_year_2200(self) -> None:
        # decade 220 -> only year 2200 in range [1800, 2200]
        cond = Condition(dow=0, month=0, leap=0, decade=40)
        mask = valid_year_digit_mask(cond)
        # Only digit 0 (year 2200) is valid; 2200 is not a leap year
        self.assertEqual(mask, [True] + [False] * 9)

    def test_decade_220_leap_true_impossible(self) -> None:
        # No leap year exists in decade 220 (only year 2200, not leap)
        cond = Condition(dow=0, month=0, leap=1, decade=40)
        self.assertEqual(valid_year_digit_mask(cond), [False] * 10)

    def test_feb_non_leap_28_days(self) -> None:
        # decade 180 + digit 1 = 1801, not leap, FEB
        cond = Condition(dow=0, month=1, leap=0, decade=0)
        day_mask = valid_day_mask(cond, year_digit=1)  # 1801
        self.assertEqual(sum(day_mask), 28)
        self.assertFalse(day_mask[28])  # day 29 invalid
        self.assertTrue(day_mask[27])    # day 28 valid

    def test_feb_leap_29_days(self) -> None:
        # decade 180 + digit 4 = 1804, leap year, FEB
        cond = Condition(dow=0, month=1, leap=1, decade=0)
        day_mask = valid_day_mask(cond, year_digit=4)  # 1804
        self.assertEqual(sum(day_mask), 29)
        self.assertTrue(day_mask[28])     # day 29 valid
        self.assertFalse(day_mask[29])    # day 30 invalid

    def test_full_mask_consistency(self) -> None:
        cond = Condition(dow=2, month=0, leap=0, decade=0)  # WED JAN non-leap 180s
        m = full_valid_mask(cond)
        # Total valid (day, ydigit) pairs ≈ 31 days × 8 non-leap years in decade 180
        # (1804 and 1808 are leap, so 8 non-leap years out of 10)
        non_leap_years = sum(valid_year_digit_mask(cond))
        self.assertEqual(non_leap_years, 8)
        total = sum(sum(row) for row in m)
        self.assertEqual(total, 31 * non_leap_years)


class TokenizerTests(unittest.TestCase):
    def test_vocab_layout(self) -> None:
        tok = ConditionTokenizer()
        ids = tok.ids
        self.assertEqual(ids.vocab_size, 105)
        # Decoders should invert encoders exactly
        prompt = tok.encode_prompt(Condition(dow=2, month=0, leap=0, decade=0))
        self.assertEqual(prompt, [ids.bos, 2, ids.month_start + 0, ids.leap_start + 0, ids.decade_start + 0])

    def test_target_roundtrip(self) -> None:
        tok = ConditionTokenizer()
        target = tok.encode_target(day=15, year_digit=7)
        self.assertEqual(tok.decode_day_token(target[0]), 15)
        self.assertEqual(tok.decode_year_digit_token(target[1]), 7)

    def test_target_out_of_range(self) -> None:
        tok = ConditionTokenizer()
        with self.assertRaises(ValueError):
            tok.encode_target(day=32, year_digit=0)
        with self.assertRaises(ValueError):
            tok.encode_target(day=1, year_digit=10)


class MetricsTests(unittest.TestCase):
    def test_compliance_all_correct(self) -> None:
        cond, date = parse_input_line("[WED] [JAN] [False] [180] 1-1-1800")
        self.assertIsNotNone(date)
        assert date is not None
        c = condition_compliance(cond, date)
        self.assertTrue(all(c.values()))
        self.assertTrue(joint_compliance(cond, date))

    def test_compliance_wrong_dow(self) -> None:
        cond = Condition(dow=DAYS_OF_WEEK.index("MON"), month=0, leap=0, decade=0)
        # 1-1-1800 was a Wednesday, not Monday
        date = _dt.date(1800, 1, 1)
        c = condition_compliance(cond, date)
        self.assertFalse(c["dow"])
        self.assertFalse(joint_compliance(cond, date))

    def test_out_of_range_invalid(self) -> None:
        self.assertFalse(date_validity(_dt.date(1799, 12, 31)))
        self.assertFalse(date_validity(_dt.date(2201, 1, 1)))
        self.assertTrue(date_validity(_dt.date(1800, 1, 1)))
        self.assertTrue(date_validity(_dt.date(2200, 12, 31)))
        self.assertFalse(date_validity(None))

    def test_evaluate_aggregates(self) -> None:
        cond, date = parse_input_line("[WED] [JAN] [False] [180] 1-1-1800")
        report = evaluate([(cond, date), (cond, _dt.date(1800, 1, 8))])  # both WED JAN 180 non-leap
        self.assertAlmostEqual(report.acc_joint, 1.0)
        self.assertAlmostEqual(report.validity, 1.0)

    def test_diversity_entropy(self) -> None:
        d1 = _dt.date(1800, 1, 1)
        d2 = _dt.date(1800, 1, 8)
        d3 = _dt.date(1800, 1, 15)
        # All-same -> 0 entropy
        self.assertAlmostEqual(diversity_entropy([[d1, d1, d1]]), 0.0)
        # All-different -> log(N)
        import math
        self.assertAlmostEqual(diversity_entropy([[d1, d2, d3]]), math.log(3))


class DataFileTests(unittest.TestCase):
    """Sanity-check the actual data files we are about to train on."""

    def test_every_data_line_parses(self) -> None:
        path = ROOT / "data" / "data.txt"
        if not path.exists():
            self.skipTest("data/data.txt not present")
        n_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cond, date = parse_input_line(line)
                self.assertIsNotNone(date)
                # Every line in data.txt must satisfy its own conditions
                assert date is not None
                self.assertTrue(joint_compliance(cond, date),
                                f"non-compliant data line: {line!r}")
                n_lines += 1
        # The dataset has every date 1-1-1800..31-12-2200 inclusive = 146462 days
        # (400 years 1800-2199 with 97 leap years = 146097 days, plus 365 days in 2200)
        self.assertEqual(n_lines, 146462)

    def test_example_input_lines_parse(self) -> None:
        path = ROOT / "data" / "example_input.txt"
        if not path.exists():
            self.skipTest("data/example_input.txt not present")
        n_lines = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cond, date = parse_input_line(line)
                self.assertIsNone(date, f"example line had a date: {line!r}")
                n_lines += 1
        self.assertGreater(n_lines, 1000)


if __name__ == "__main__":
    unittest.main()
