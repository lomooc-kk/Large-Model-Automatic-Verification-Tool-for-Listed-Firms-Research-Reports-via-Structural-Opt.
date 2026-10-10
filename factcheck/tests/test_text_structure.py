"""Source-derived relation cases and controls through the production entrypoint."""
from __future__ import annotations

import json
import unittest

from yjcheck.text_context import TextSlice
from yjcheck.text_review import detect_text, _messages


def findings(text: str, detector: str) -> list[dict]:
    result = detect_text(text, document_id="structure-contract", chat=None)
    assert result["coverage"]["model_ran"] is False
    return [item for item in result["errors"] if item["detector_id"] == detector]


class SourceStructureTests(unittest.TestCase):
    def test_paired_series_count_is_numeric_consistency_not_arithmetic(self):
        text = "甲公司两年收入依次为8.4万元/9.2万元，YOY依次为2%/3%/4%。"
        rows = findings(text, "verified.aligned_series_cardinality")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["error_type"], "数值不一致错误")
        self.assertEqual(rows[0]["spans"][0]["text"], text)
        proof = next(e for e in rows[0]["evidence"] if e["kind"] == "deterministic")
        self.assertEqual((proof["value_count"], proof["rate_count"]), (2, 3))
        self.assertEqual(proof["value_tokens"], ["8.4万元", "9.2万元"])
        self.assertEqual(proof["rate_tokens"], ["2%", "3%", "4%"])

    def test_equal_or_not_explicitly_paired_lists_are_not_errors(self):
        for text in (
            "收入8.4万元/9.2万元，YOY依次为2%/3%。",
            "收入8.4万元/9.2万元，另列年度增长率2%/3%/4%。",
            "收入8.4万元/9.2亿元，YOY依次为2%/3%/4%。",
            "收入8.4万元/9.2万元，YOY为2%/3%/4%。",
            "16件产品由6家公司和7家公司生产。",
        ):
            with self.subTest(text=text):
                self.assertEqual(findings(text, "verified.aligned_series_cardinality"), [])

    def test_partial_compound_lists_are_not_counted(self):
        for text in (
            "收入1万元/2万元（调整后）/3万元，YOY依次为2%/3%。",
            "收入1万元/2万元，YOY依次为2%/3%/4%-5%。",
            "收入1万元/2万元，YOY依次为2%/3%/4%/5（预测）。",
            "收入1万元/2万元，YOY依次为2%/3%/4%至5%。",
            "收入1-2万元/3万元/4万元，YOY依次为2%/3%/4%。",
        ):
            with self.subTest(text=text):
                self.assertEqual(findings(text, "verified.aligned_series_cardinality"), [])

    def test_other_series_and_rounded_growth_are_not_added_to_counts(self):
        text = ("甲公司收入8.4万元/9.2万元，YOY依次为2%/3%；"
                "利润1.18万元/1.37万元，YOY依次为2.3%/16.1%。")
        self.assertEqual(findings(text, "verified.aligned_series_cardinality"), [])

    def test_quoted_or_hypothetical_count_mismatch_is_not_confirmed(self):
        self.assertEqual(findings("错误示例：收入1万元/2万元，YOY依次为1%/2%/3%。",
                                  "verified.aligned_series_cardinality"), [])

    def test_suspended_semicolon_item_keeps_common_subject_and_narrow_proof(self):
        text = "本季度采购甲型设备，数量8台；乙型设备，；丙型设备，数量6台。"
        rows = findings(text + "下一季度另议。", "verified.suspended_punctuation")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["spans"][0]["text"], text)
        self.assertEqual(rows[0]["surface_spans"][0]["text"], "，；")
        self.assertEqual(rows[0]["status"], "needs_review")
        for span in rows[0]["spans"] + rows[0]["surface_spans"]:
            self.assertEqual(text[span["start"]:span["end"]], span["text"])

    def test_adjacent_explicit_ordinal_sibling_supplies_missing_field_context(self):
        text = "第一档次企业8家，销售额12亿元。第二档次企业6家，。"
        rows = findings(text, "verified.suspended_punctuation")
        self.assertEqual(rows[0]["spans"][0]["text"], text)
        self.assertNotIn("replacement", rows[0])

    def test_context_does_not_absorb_unrelated_sentence_or_previous_paragraph(self):
        for prefix in ("行业收入稳定。", "第一档次企业8家。\n\n"):
            current = "第二档次企业6家，。"
            rows = findings(prefix + current, "verified.suspended_punctuation")
            self.assertEqual(rows[0]["spans"][0]["text"], current)

    def test_hypothetical_context_before_semicolon_prevents_confirmation(self):
        text = "错误示例：采购甲型设备，数量8台；乙型设备，；丙型设备，数量6台。"
        self.assertEqual(findings(text, "verified.suspended_punctuation"), [])

    def test_valid_but_reversed_month_and_day_endpoints_are_time_relations(self):
        for text in ("报告载明2024年8-2月收入稳定。", "活动期间为5月9日-2日。"):
            rows = findings(text, "structured.reversed_date_range")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["error_type"], "时间矛盾")
            self.assertEqual(rows[0]["status"], "needs_review")
            self.assertEqual(rows[0]["spans"][0]["text"], text)

    def test_illegal_dates_and_explicit_cross_period_ranges_are_not_relabeled(self):
        for text in ("2024年2月31日-1日。", "2023年2月29日-1日。",
                     "2024年8-12月。", "2024年12月至2025年1月。",
                     "跨年期间2024年12-1月。", "5月31日至6月1日。",
                     "按倒序显示5月9日-2日。", "错误示例：2024年8-2月。"):
            with self.subTest(text=text):
                self.assertEqual(findings(text, "structured.reversed_date_range"), [])

    def test_display_precision_overlap_only_prevents_unproven_conflict(self):
        text = "甲公司2024年营业收入1.2亿元。甲公司2024年营业收入12003万元。"
        self.assertEqual(findings(text, "hybrid.cross_section_numeric"), [])
        text = "甲公司2024年毛利率10.1%。甲公司2024年毛利率10.14%。"
        self.assertEqual(findings(text, "hybrid.cross_section_numeric"), [])

    def test_disjoint_display_intervals_still_surface_same_scope_conflict(self):
        text = "甲公司2024年营业收入100万元。甲公司2024年营业收入101万元。"
        rows = findings(text, "hybrid.cross_section_numeric")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "needs_review")
        proof = next(e for e in rows[0]["evidence"] if e["kind"] == "comparison")
        self.assertEqual(proof["display_intervals"], [["995000", "1005000"], ["1005000", "1015000"]])

    def test_old_and_new_guidance_or_actual_and_forecast_are_not_mixed(self):
        for suffix in ("（旧指引）", "（此前预测）", "（原先目标）"):
            text = f"甲公司2024年营业收入100万元{suffix}。甲公司2024年营业收入200万元（新指引）。"
            self.assertEqual(findings(text, "hybrid.cross_section_numeric"), [])

    def test_coarse_first_value_does_not_hide_later_disjoint_values(self):
        text = ("甲公司2024年营业收入1亿元。甲公司2024年营业收入0.8亿元。"
                "甲公司2024年营业收入1.2亿元。")
        rows = findings(text, "hybrid.cross_section_numeric")
        self.assertEqual(len(rows), 1)
        self.assertEqual([s["text"] for s in rows[0]["spans"]],
                         ["甲公司2024年营业收入0.8亿元", "甲公司2024年营业收入1.2亿元"])

    def test_source_arithmetic_check_reaches_messages_with_global_offsets(self):
        text = "现货价格为180元/吨，环比上周-20元/吨，跌幅10.0%。"
        context = TextSlice(17, 17 + len(text), text)
        messages = _messages([context, context], "arithmetic-input", "", [])
        checks = json.loads(messages[-1]["content"])["source_arithmetic_checks"]
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["derived_previous_price"], "200")
        self.assertTrue(checks[0]["rounding_compatible"])
        span = checks[0]["source"]
        self.assertEqual(text[span["start"] - 17:span["end"] - 17], span["text"])
        self.assertIn("不是业务结论", messages[0]["content"])

    def test_arithmetic_context_is_bounded_across_all_slices(self):
        text = "价格为180元/吨，环比-20元/吨，跌幅10.0%。"
        contexts = [TextSlice(i * len(text), (i + 1) * len(text), text) for i in range(20)]
        checks = json.loads(_messages(contexts, "bounded", "", [])[-1]["content"])["source_arithmetic_checks"]
        self.assertEqual(len(checks), 12)


if __name__ == "__main__":
    unittest.main()
