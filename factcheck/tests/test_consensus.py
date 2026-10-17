"""Unit tests for merging repeated runs (no model calls, no gold)."""
from __future__ import annotations

import unittest

from yjcheck.consensus import candidate_key, merge_reports, merge_runs, summary


def candidate(error_type: str, start: int, end: int, reason: str = "r") -> dict:
    return {
        "error_type": error_type,
        "spans": [{"start": start, "end": end, "text": "x" * (end - start)}],
        "reason": reason,
        "status": "needs_review",
    }


class MergeRunsTest(unittest.TestCase):
    def test_counts_across_runs_and_keeps_first_text(self) -> None:
        run1 = [candidate("冗余语句", 10, 20, "first")]
        run2 = [candidate("冗余语句", 10, 20, "second")]
        run3 = [candidate("冗余语句", 10, 20, "third")]
        merged = merge_runs([run1, run2, run3])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["reason"], "first")
        self.assertEqual(merged[0]["confidence"]["votes"], 3)
        self.assertEqual(merged[0]["confidence"]["trials"], 3)
        self.assertLess(merged[0]["confidence"]["lo"], 0.5)

    def test_duplicate_inside_one_run_counts_once(self) -> None:
        run = [candidate("冗余语句", 10, 20), candidate("冗余语句", 10, 20)]
        merged = merge_runs([run, [candidate("冗余语句", 10, 20)]])
        self.assertEqual(merged[0]["confidence"]["votes"], 2)

    def test_single_run_candidate_gets_wide_interval(self) -> None:
        merged = merge_runs([[candidate("冗余语句", 10, 20)], []])
        self.assertEqual(merged[0]["confidence"]["votes"], 1)
        self.assertEqual(merged[0]["confidence"]["trials"], 2)
        self.assertGreater(merged[0]["confidence"]["hi"], 0.9)

    def test_type_and_spans_both_identify_a_candidate(self) -> None:
        a = candidate("冗余语句", 10, 20)
        b = candidate("格式错误", 10, 20)
        c = candidate("冗余语句", 10, 21)
        self.assertNotEqual(candidate_key(a), candidate_key(b))
        self.assertNotEqual(candidate_key(a), candidate_key(c))
        self.assertEqual(len(merge_runs([[a, b, c]])), 3)

    def test_status_is_preserved_and_not_upgraded(self) -> None:
        merged = merge_runs([[candidate("冗余语句", 1, 5)] for _ in range(20)])
        self.assertEqual(merged[0]["status"], "needs_review")

    def test_empty_runs_return_empty(self) -> None:
        self.assertEqual(merge_runs([]), [])
        self.assertEqual(merge_runs([[], []]), [])

    def test_report_merge_records_run_count(self) -> None:
        base = {"document_id": "d1", "coverage": {"complete": True}, "errors": [candidate("冗余语句", 1, 5)]}
        other = {"document_id": "d1", "coverage": {"complete": True}, "errors": [candidate("冗余语句", 1, 5)]}
        merged = merge_reports([base, other])
        self.assertEqual(merged["consensus"]["runs"], 2)
        self.assertEqual(merged["errors"][0]["confidence"]["votes"], 2)
        self.assertTrue(merged["coverage"]["complete"])

    def test_summary_bands(self) -> None:
        many = merge_runs([[candidate("冗余语句", 1, 5)] for _ in range(20)])
        never = merge_runs([[] for _ in range(20)] + [[candidate("冗余语句", 9, 10)]])
        self.assertEqual(summary(many)["auto_confirm"], 1)
        self.assertEqual(summary(never)["drop"], 1)


if __name__ == "__main__":
    unittest.main()
