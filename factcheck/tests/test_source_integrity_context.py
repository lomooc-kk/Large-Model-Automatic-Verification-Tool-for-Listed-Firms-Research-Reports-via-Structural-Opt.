import unittest

from yjcheck.source_integrity_context import source_integrity_checks


class SourceIntegrityContextTests(unittest.TestCase):
    def assert_exact_spans(self, text, checks, offset=0):
        for check in checks:
            for span in [check["source"], *check["tokens"].values()]:
                self.assertEqual(text[span["start"] - offset:span["end"] - offset], span["text"])
                self.assertLess(span["start"], span["end"])

    def test_direct_metric_magnitude_is_surface_only(self):
        text = "本期实现营业总收入亿，同比增19%。"
        check = source_integrity_checks(text, offset=23)[0]
        self.assertEqual(check["kind"], "printed_magnitude_after_metric_without_number")
        self.assert_exact_spans(text, [check], 23)
        self.assertFalse(check["missing_field_proven"])
        self.assertFalse(check["business_error_proven"])
        self.assertFalse(check["relationship_binding_proven"])
        self.assertNotIn("error_type", check)
        self.assertNotIn("corrected_text", check)

    def test_zero_approximate_amount_and_unit_heading_are_not_gaps(self):
        for text in ("实现营业总收入0亿元，同比增19%。", "实现营业收入约1亿元。",
                     "营业收入（亿元）", "实现营业收入亿元级突破。", "营业收入亿，净利润亿。",
                     "录得净利润-0.01亿元。", "实现营业收入1亿元。", "实现营业收入数亿元。"):
            with self.subTest(text=text):
                self.assertEqual(source_integrity_checks(text), [])

    def test_quantified_line_ending_at_prefix_keeps_excerpt_ambiguity(self):
        for text in ("基本每股收益1.16元，加权\n\n股息政策", "营收占比90.2%；毛\n\n其他业务",
                     "营收170.6亿，同比增10.6%，扣非归母净利"):
            with self.subTest(text=text):
                checks = source_integrity_checks(text, offset=7)
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0]["kind"], "printed_metric_prefix_at_text_boundary")
                self.assertTrue(checks[0]["supplied_text_boundary_only"])
                self.assert_exact_spans(text, checks, 7)

    def test_normal_headings_complete_metrics_and_unrelated_paragraphs_abstain(self):
        for text in ("加权\n平均价格", "营收170亿元。\n扣非归母净利", "每股收益1元。加权",
                     "营收1亿元，毛利率20%。", "基本每股收益1元，加权平均每股收益0.9元。",
                     "营收1亿元\n毛利率", "业务覆盖物流、消费和航空", "公司以精细化管理改善经营"):
            with self.subTest(text=text):
                self.assertEqual(source_integrity_checks(text), [])

    def test_direct_dimension_joins_are_local_not_whole_document(self):
        text = "新能源工程并网0.54万/吨，下降5%。电网累计投资1408亿元，同比增长45.33GW，同比11%。其他业务正常。"
        checks = source_integrity_checks(text, offset=41)
        self.assertEqual([c["kind"] for c in checks], ["capacity_phrase_directly_followed_by_price_unit", "money_metric_growth_followed_by_power_unit"])
        self.assert_exact_spans(text, checks, 41)
        self.assertTrue(all("其他业务" not in c["source"]["text"] for c in checks))
        self.assertTrue(all(not c["relationship_binding_proven"] for c in checks))

    def test_normal_cross_business_news_and_adjacent_metrics_abstain(self):
        for text in ("电网投资1408亿元，新增装机45.33GW。", "投资1408亿元，同比增长11%。装机45GW。",
                     "项目并网0.54GW，材料价格0.54万/吨。", "装机容量为5GW，投资额30亿元。",
                     "新能源工程并网；化工报价0.54万/吨。", "投资1408亿元，同比增长45.33%。",
                     "并网电价0.54元/度。", "光伏并网后，运费30元/吨。"):
            with self.subTest(text=text):
                self.assertEqual(source_integrity_checks(text), [])

    def test_bounded_list_tail_is_recorded_without_guessing_an_item(self):
        for text in ("涨幅前五为甲公司、乙公司、丙公司、丁公司、；", "收入同比增速分别16.5%、14.15%、，后文继续。",
                     "销量分别为12万台、14万台、。", "其中甲乙丙三类收入分别为16%、14%、，后文继续。"):
            with self.subTest(text=text):
                checks = source_integrity_checks(text)
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0]["kind"], "bounded_list_separator_before_terminator")
                self.assertFalse(checks[0]["missing_field_proven"])
                self.assert_exact_spans(text, checks)

    def test_legal_omissions_open_examples_and_different_metrics_abstain(self):
        for text in ("包括甲公司、乙公司等。", "例如，涨幅前五为甲公司、乙公司、；其余略。",
                     "涨幅前五包括甲公司、乙公司等。", "涨幅前五为甲、乙、丙、丁、戊。",
                     "收入同比增速分别为16%、14%。", "营收上涨，利润下降。", "营业收入1亿元，，同比增长。",
                     "收入分别为甲业务16%、其他业务等、。", "TOP品牌持续增长。"):
            with self.subTest(text=text):
                self.assertEqual(source_integrity_checks(text), [])

    def test_blank_line_cannot_bind_unrelated_fragments(self):
        for text in ("实现\n\n营业收入亿，", "并网\n\n0.54万/吨，",
                     "投资1亿元，\n\n同比增长2GW。", "合计7GW：陆上6GW/海上1GW：\n\n陆上0.5GW。"):
            with self.subTest(text=text):
                self.assertEqual(source_integrity_checks(text), [])

    def test_repeated_role_requires_explicit_total_and_same_unit_sequence(self):
        text = "本月招标7.31GW，同比+39.78%：陆上6.81GW/海上0.85GW：陆上0.5GW，同比+53%。"
        check = source_integrity_checks(text, offset=13)[0]
        self.assertEqual(check["kind"], "repeated_role_inside_explicit_component_sequence")
        self.assertFalse(check["relationship_binding_proven"])
        self.assert_exact_spans(text, [check], 13)
        self.assertNotIn("component_sum", check)
        self.assertNotIn("arithmetic_error", check)
        generic = "合计100万吨：甲业务30万吨/乙业务40万吨：甲业务20万吨。"
        self.assertEqual(len(source_integrity_checks(generic)), 1)

    def test_period_metric_subset_switches_are_not_repeated_roles(self):
        for text in ("陆上6.81GW/海上0.85GW：陆上0.5GW。",
                     "本月招标7.31GW：陆上6.81GW/海上0.5GW。",
                     "本月招标7GW：陆上6GW/海上1GW；上月陆上0.5GW。",
                     "合计7GW：陆上6GW/海上1GW：陆上新增0.5GW。",
                     "合计7GW：陆上6GW/海上1GW：其中陆上0.5GW。",
                     "合计7GW：本月陆上6GW/海上1GW：本月陆上0.5GW。",
                     "合计7GW：陆上6GW/海上1MW：陆上0.5GW。",
                     "合计7GW：陆上6GW/海上1GW。产量：陆上0.5万吨。"):
            with self.subTest(text=text):
                self.assertEqual(source_integrity_checks(text), [])

    def test_limit_stable_order_offsets_and_input_validation(self):
        text = "实现营业收入亿。并网0.5万/吨。投资1亿元，同比增长2GW。"
        all_checks = source_integrity_checks(text)
        self.assertEqual(len(all_checks), 3)
        self.assertEqual(source_integrity_checks(text, limit=2), all_checks[:2])
        self.assertEqual(source_integrity_checks(text, limit=0), [])
        self.assertEqual(source_integrity_checks(""), [])
        self.assertEqual(source_integrity_checks(text), all_checks)
        self.assert_exact_spans(text, all_checks)
        for bad_text, offset, limit in ((None, 0, 12), (text, True, 12), (text, -1, 12), (text, 0, True), (text, 0, -1)):
            with self.subTest(values=(bad_text, offset, limit)), self.assertRaises(ValueError):
                source_integrity_checks(bad_text, offset=offset, limit=limit)


if __name__ == "__main__":
    unittest.main()
