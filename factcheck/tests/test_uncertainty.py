"""Unit tests for the Wilson interval and the decision rule (no model calls)."""
from __future__ import annotations

import unittest

from yjcheck.uncertainty import (
    MIN_TRIALS,
    Interval,
    cost_floor,
    decide,
    minimum_trials_for,
    unanimous_interval,
    wilson_interval,
)


class WilsonIntervalTest(unittest.TestCase):
    def test_known_values(self) -> None:
        self.assertAlmostEqual(wilson_interval(0, 10).lo, 0.0, places=6)
        self.assertAlmostEqual(wilson_interval(0, 10).hi, 0.2775, places=4)
        self.assertAlmostEqual(wilson_interval(10, 10).lo, 0.7225, places=4)
        self.assertAlmostEqual(wilson_interval(10, 10).hi, 1.0, places=6)
        self.assertAlmostEqual(wilson_interval(5, 10).lo, 0.2366, places=4)
        self.assertAlmostEqual(wilson_interval(5, 10).hi, 0.7634, places=4)

    def test_bounds_stay_inside_unit_interval(self) -> None:
        for votes, trials in ((0, 1), (1, 1), (0, 3), (3, 3), (7, 9)):
            interval = wilson_interval(votes, trials)
            self.assertGreaterEqual(interval.lo, 0.0)
            self.assertLessEqual(interval.hi, 1.0)
            self.assertLessEqual(interval.lo, interval.p)
            self.assertGreaterEqual(interval.hi, interval.p)

    def test_more_trials_narrow_the_interval(self) -> None:
        narrow = unanimous_interval(20)
        wide = unanimous_interval(3)
        self.assertGreater(narrow.lo, wide.lo)
        self.assertLessEqual(narrow.hi, 1.0)

    def test_unanimity_is_not_certainty(self) -> None:
        self.assertLess(unanimous_interval(3).lo, 0.5)
        self.assertAlmostEqual(unanimous_interval(20).lo, 0.8389, places=3)

    def test_invalid_inputs_are_rejected(self) -> None:
        for votes, trials in ((1, 0), (-1, 5), (6, 5), (True, 5)):
            with self.assertRaises(ValueError):
                wilson_interval(votes, trials)

    def test_pretty_dict_rounds_without_changing_semantics(self) -> None:
        payload = wilson_interval(9, 10).as_dict()
        self.assertEqual(payload["method"], "wilson")
        self.assertEqual(payload["votes"], 9)
        self.assertEqual(payload["trials"], 10)
        self.assertLess(payload["lo"], payload["p"])


class DecisionTest(unittest.TestCase):
    def test_two_trials_are_never_enough(self) -> None:
        self.assertEqual(decide(Interval(2, 2, 1.0, 0.342, 1.0)), "needs_review")

    def test_needs_many_unanimous_runs_to_auto_confirm(self) -> None:
        self.assertEqual(decide(unanimous_interval(3)), "needs_review")
        self.assertEqual(decide(unanimous_interval(10)), "needs_review")
        self.assertEqual(decide(unanimous_interval(20)), "auto_confirm")

    def test_upper_bound_below_drop_threshold_drops(self) -> None:
        self.assertEqual(decide(wilson_interval(0, 20)), "drop")
        self.assertEqual(decide(wilson_interval(1, 30)), "drop")

    def test_mixed_agreement_stays_for_review(self) -> None:
        self.assertEqual(decide(wilson_interval(6, 10)), "needs_review")

    def test_invalid_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            decide(unanimous_interval(5), floor=0.4, drop_below=0.5)

    def test_minimum_trials_matches_the_rule(self) -> None:
        k = minimum_trials_for(0.80)
        self.assertGreaterEqual(unanimous_interval(k).lo, 0.80)
        self.assertLess(unanimous_interval(k - 1).lo, 0.80)
        self.assertGreaterEqual(k, MIN_TRIALS)

    def test_cost_floor_reuses_existing_importance(self) -> None:
        self.assertGreater(cost_floor("计算错误"), cost_floor("冗余语句"))


if __name__ == "__main__":
    unittest.main()
