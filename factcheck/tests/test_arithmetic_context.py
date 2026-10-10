from decimal import Decimal
import unittest

from yjcheck.arithmetic_context import (price_change_checks, inventory_change_checks,
    total_component_checks, valuation_rounding_checks, rounded_share_checks, source_arithmetic_checks)


class PriceArithmeticContextTests(unittest.TestCase):
    def test_decline_uses_previous_price_not_current_price(self):
        text = "现货价格为180元/吨，环比上周-20元/吨，跌幅10.0%。"
        result = price_change_checks(text, offset=17)
        self.assertEqual(len(result), 1)
        check = result[0]
        self.assertEqual(Decimal(check["derived_previous_price"]), Decimal(200))
        self.assertEqual(Decimal(check["point_change_percent"]), Decimal(-10))
        self.assertTrue(check["rounding_compatible"])
        span = check["source"]
        self.assertEqual(text[span["start"] - 17:span["end"] - 17], span["text"])

    def test_increase_and_real_disagreement(self):
        good = price_change_checks("报价为220元/股，较昨日上涨20元/股，涨幅10.0%。")[0]
        bad = price_change_checks("报价为220元/股，较昨日上涨20元/股，涨幅20.0%。")[0]
        self.assertTrue(good["rounding_compatible"])
        self.assertFalse(bad["rounding_compatible"])
        self.assertEqual(bad["scope"], "calculation_only_not_business_verdict")

    def test_rounding_intervals_preserve_valid_displayed_rates(self):
        check = price_change_checks("价格为101元/吨，环比+1元/吨，涨幅1.4%。")[0]
        self.assertNotEqual(Decimal(check["point_change_percent"]), Decimal("1.4"))
        self.assertTrue(check["rounding_compatible"])

    def test_negative_decline_not_double_negated(self):
        check = price_change_checks("价格为180元/吨，环比-20元/吨，跌幅-10.0%。")[0]
        self.assertEqual(Decimal(check["reported_signed_change_percent"]), Decimal(-10))
        self.assertTrue(check["rounding_compatible"])

    def test_units_and_missing_relation_abstain(self):
        for text in (
            "价格为180元/吨，环比-20美元/吨，跌幅10%。",
            "价格为180元/吨。另一商品环比-20元/吨，跌幅10%。",
            "价格为180元/吨，环比20元/吨，跌幅10%。",
            "价格为20元/吨，环比+20元/吨，涨幅10%。",
            "价格约为180元/吨，环比-20元/吨，跌幅10%。",
        ):
            with self.subTest(text=text):
                self.assertEqual(price_change_checks(text), [])

    def test_multiple_local_prices_are_not_cross_compared(self):
        text = "甲价格为180元/吨，环比-20元/吨，跌幅10%。乙价格为220元/吨，环比+20元/吨，涨幅10%。"
        checks = price_change_checks(text)
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(c["rounding_compatible"] for c in checks))
        self.assertEqual(len(price_change_checks(text, limit=1)), 1)
        self.assertEqual(price_change_checks(text, limit=0), [])

    def test_thousands_separators_and_fullwidth_unit(self):
        check = price_change_checks("价格为1,800元／吨，环比5月20日−200元／吨，跌幅10.0%。")[0]
        self.assertEqual(check["unit"], "元/吨")
        self.assertTrue(check["rounding_compatible"])


class StructuredArithmeticContextTests(unittest.TestCase):
    def test_inventory_change_base_and_exact_global_source(self):
        text = "甲港库存53.5万吨，周环比增加3.5万吨（7.0%）。"
        check = inventory_change_checks(text, offset=9)[0]
        self.assertEqual(Decimal(check['derived_previous_quantity']),Decimal('50'))
        self.assertTrue(check['rounding_compatible'])
        self.assertEqual(text[check['source']['start']-9:check['source']['end']-9],check['source']['text'])
        decline=inventory_change_checks("库存47.5万吨，月环比减少2.5万吨（5.0%）。")[0]
        self.assertEqual(Decimal(decline['point_change_percent']),Decimal('-5'))
        self.assertTrue(decline['rounding_compatible'])

    def test_inventory_mismatched_unit_unknown_base_and_object_abstain(self):
        for text in ('库存53.5万吨，周环比增加3.5吨（7.0%）。',
                     '库存53.5万吨。另一港周环比增加3.5万吨（7.0%）。',
                     '库存3.5万吨，周环比增加3.5万吨（7.0%）。',
                     '库存超53.5万吨，周环比增加3.5万吨（7.0%）。'):
            with self.subTest(text=text): self.assertEqual(inventory_change_checks(text),[])

    def test_sum_rounding_and_incomplete_partition_not_a_verdict(self):
        check=total_component_checks('总新增装机100万千瓦，其中火电33万千瓦，水电33万千瓦，风电35万千瓦。')[0]
        self.assertEqual(check['component_sum'],'101')
        self.assertTrue(check['rounding_compatible'])
        self.assertFalse(check['exhaustive_partition_proven'])
        self.assertFalse(check['relation_proven'])
        bad=total_component_checks('合计100.0万元，其中甲业务80.0万元、乙业务40.0万元。')[0]
        self.assertFalse(bad['rounding_compatible'])
        self.assertEqual(bad['scope'],'calculation_only_not_business_verdict')
        self.assertFalse(bad['relation_proven'])

    def test_sum_rejects_partial_parse_units_overlap_labels_and_qualifiers(self):
        for text in ('总计100万元，其中甲业务80万元，乙业务20亿元。',
                     '合计100万元，其中甲业务80万元，甲业务20万元。',
                     '总计100万元，其中甲业务约80万元，乙业务20万元。',
                     '总计100万元，其中甲业务超过80万元，乙业务20万元。',
                     '总计100万元，其中甲业务亏损80万元，乙业务180万元。',
                     '总计100万元，其中甲业务80万元，乙业务20万元，同比增20%。',
                     '总计100万元。其中甲业务80万元，乙业务20万元。'):
            with self.subTest(text=text): self.assertEqual(total_component_checks(text),[])

    def test_eps_pe_rounding_compatibility_and_real_disagreement(self):
        text='EPS分别为0.33元、0.50元，按某日收盘价10.00元计算，对应PE为30.10倍、20.00倍。'
        check=valuation_rounding_checks(text)[0]
        self.assertTrue(check['rounding_compatible'])
        self.assertNotEqual(Decimal(check['pairs'][0]['point_implied_price']),Decimal('10'))
        bad=valuation_rounding_checks(text.replace('30.10倍','40.10倍'))[0]
        self.assertFalse(bad['rounding_compatible'])
        self.assertIn('EPS_definition_matches_PE_denominator',bad['required_business_checks'])

    def test_valuation_abstains_on_mismatch_currency_qualifier_or_missing_price(self):
        for text in ('EPS为0.33元、0.50元，按收盘价10.00元计算，对应PE为30倍、20倍、15倍。',
                     'EPS为0.33元、0.50元，按收盘价10.00港元计算，对应PE为30倍、20倍。',
                     'EPS为约0.33元、0.50元，按收盘价10.00元计算，对应PE为30倍、20倍。',
                     'EPS为0.33元、0.50元，对应PE为30倍、20倍。',
                     'EPS为-0.33元、0.50元，按收盘价10.00元计算，对应PE为-30倍、20倍。'):
            with self.subTest(text=text): self.assertEqual(valuation_rounding_checks(text),[])

    def test_aggregation_preserves_old_price_output_without_extra_matches(self):
        text='价格为180元/吨，环比-20元/吨，跌幅10.0%。'
        self.assertEqual(source_arithmetic_checks(text),price_change_checks(text))
        self.assertEqual(source_arithmetic_checks(text,limit=0),[])
        for offset,limit in ((-1,12),(0,-1),(True,12),(0,True)):
            with self.subTest(offset=offset,limit=limit), self.assertRaises(ValueError):
                source_arithmetic_checks(text,offset=offset,limit=limit)


class RoundedShareContextTests(unittest.TestCase):
    def test_displayed_money_and_percent_precision_are_independent(self):
        template = '总营收{}亿美元，其中甲营收{}亿美元，占总营收{}%。'
        for denominator, numerator, percent, expected in (
            ('441', '391', '88', True),
            ('441.0', '391.0', '88', False),
            ('441', '391', '88.0', False),
            ('441', '391', '87', False),
        ):
            with self.subTest(values=(denominator, numerator, percent)):
                check = rounded_share_checks(template.format(denominator, numerator, percent))[0]
                self.assertEqual(check['rounding_compatible'], expected)
                self.assertEqual(check['numerator_raw'], numerator)
                self.assertEqual(check['reported_percent_raw'], percent)
                self.assertFalse(check['relation_proven'])
        check = rounded_share_checks(template.format('441', '391', '88'))[0]
        self.assertEqual(check['numerator_interval'], ['390.5', '391.5'])
        self.assertLess(Decimal(check['possible_percent'][0]), Decimal('88.5'))
        self.assertGreater(Decimal(check['possible_percent'][1]), Decimal('88.5'))

    def test_report_paragraph_uses_actual_amounts_not_forecasts_or_growth(self):
        text = ('某企业发布FY26Q1业绩，一季度整体营收为441亿美元，同比增长69%，'
                '高于市场预期的432.9亿美元，净利润187.8亿美元，同比增长26%，'
                '芯片相关费用45亿美元，低于此前预估水平。'
                '其中核心业务数据中心营收同比增长73%，达到391亿美元'
                '（市场预期393亿美元），占销售额的88%。')
        check = rounded_share_checks(text, offset=29)[0]
        self.assertEqual((check['numerator_raw'], check['denominator_raw'],
                          check['reported_percent_raw']), ('391', '441', '88'))
        self.assertTrue(check['rounding_compatible'])
        self.assertEqual({x['role'] for x in check['excluded_roles']},
                         {'period_growth_not_share', 'forecast_not_actual'})
        self.assertIn('revenue_sales_metric_equivalence', check['required_business_checks'])
        for span in [check['source'], *check['operand_sources'].values(),
                     *[x['source'] for x in check['excluded_roles']]]:
            self.assertEqual(text[span['start']-29:span['end']-29], span['text'])

    def test_general_amounts_currencies_and_thousands_not_case_ids(self):
        for text, values in (
            ('总金额1,000万元，其中甲项目金额250万元，占总金额25%。', ('250', '1,000', '25')),
            ('整体销售额80港元，其中乙销售额20港元，占销售额25％。', ('20', '80', '25')),
            ('合计金额800美元，其中乙金额200美元，占总额25%。', ('200', '800', '25')),
        ):
            with self.subTest(text=text):
                check = rounded_share_checks(text)[0]
                self.assertEqual(tuple(check[x] for x in ('numerator_raw', 'denominator_raw',
                                                         'reported_percent_raw')), values)
                self.assertTrue(check['rounding_compatible'])

    def test_scope_is_bounded_same_paragraph_and_unique_total(self):
        for text in (
            '总营收441亿元。\n\n其中甲营收391亿元，占总营收88%。',
            '总营收441亿元。\r\n  \r\n其中甲营收391亿元，占总营收88%。',
            '总营收441亿元。' + '资料。' * 300 + '其中甲营收391亿元，占总营收88%。',
            '甲总营收441亿元，乙总营收450亿元，其中业务营收391亿元，占总营收88%。',
            '其中甲营收391亿元，占总营收88%。总营收441亿元。',
        ):
            with self.subTest(text=text[:70]): self.assertEqual(rounded_share_checks(text), [])
        self.assertEqual(len(rounded_share_checks(
            '总营收100亿元。\n其中甲营收25亿元，占总营收25%。')), 1)

    def test_period_entity_and_reporting_basis_switches_abstain(self):
        for text in (
            '2024年总营收441亿元，2025年其中甲营收391亿元，占总营收88%。',
            '去年总营收441亿元，其中今年甲营收391亿元，占总营收88%。',
            '年度总营收441亿元。其中Q1甲营收391亿元，占总营收88%。',
            '甲公司总营收441亿元。乙公司其中业务营收391亿元，占总营收88%。',
            '总营收441亿元，其中另一公司营收391亿元，占总营收88%。',
            '含税总营收441亿元，其中甲营收391亿元，占总营收88%。',
            '总营收441亿元，其中甲净收入391亿元，占总营收88%。',
            '总金额441亿元，其中甲营收391亿元，占总营收88%。',
            '总GMV441亿元，其中甲净收入391亿元，占收入88%。',
        ):
            with self.subTest(text=text): self.assertEqual(rounded_share_checks(text), [])

    def test_currency_scale_counts_zero_and_negative_abstain(self):
        for text in (
            '总营收441亿元，其中甲营收391万元，占总营收88%。',
            '总营收441亿美元，其中甲营收391亿元，占总营收88%。',
            '总营收441亿元/人，其中甲营收391亿元，占总营收88%。',
            '总企业数441家，其中甲企业数391家，占总企业数88%。',
            '总营收0亿元，其中甲营收1亿元，占总营收88%。',
            '总营收-441亿元，其中甲营收391亿元，占总营收88%。',
            '总营收441亿元，其中甲营收-391亿元，占总营收88%。',
        ):
            with self.subTest(text=text): self.assertEqual(rounded_share_checks(text), [])

    def test_forecast_and_unbounded_precision_do_not_get_rounding_intervals(self):
        for text in (
            '预计总营收441亿元，其中甲营收391亿元，占总营收88%。',
            '预计，总营收441亿元，其中甲营收391亿元，占总营收88%。',
            '据预测，公司的整体营收441亿元，其中甲营收391亿元，占总营收88%。',
            '总营收441亿元，其中预计甲营收391亿元，占总营收88%。',
            '总营收约441亿元，其中甲营收391亿元，占总营收88%。',
            '总营收441亿元，其中甲营收超过391亿元，占总营收88%。',
            '总营收441亿元，其中甲营收391亿元，占总营收至少88%。',
            '金额均精确未舍入。总营收441亿元，其中甲营收391亿元，占总营收88%。',
            '总营收441亿元，其中甲营收391亿元（预计占比88%），占总营收88%。',
            '总营收441亿元，其中甲营收391亿元，占总营收88% 至90%。',
        ):
            with self.subTest(text=text): self.assertEqual(rounded_share_checks(text), [])
        check = rounded_share_checks(
            '市场预期总营收110亿元，实际总营收100亿元，其中甲营收25亿元，占总营收25%。')[0]
        self.assertEqual(check['denominator_raw'], '100')

    def test_repeated_paragraphs_anchor_each_occurrence_and_respect_limit(self):
        paragraph = '总营收100亿元，其中甲营收25亿元，占总营收25%。'
        text = paragraph + '\n\n' + paragraph
        checks = rounded_share_checks(text, offset=17)
        self.assertEqual(len(checks), 2)
        self.assertGreater(checks[1]['source']['start'], checks[0]['source']['end'])
        for check in checks:
            for span in [check['source'], *check['operand_sources'].values()]:
                self.assertEqual(text[span['start']-17:span['end']-17], span['text'])
        self.assertEqual(len(rounded_share_checks(text, limit=1)), 1)
        self.assertEqual(rounded_share_checks(text, limit=0), [])
        for args in ({'offset': -1}, {'offset': True}, {'limit': -1}, {'limit': True}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                rounded_share_checks(text, **args)

    def test_additive_helper_does_not_emit_or_remove_error_candidates(self):
        text = ('5月32日，总营收100亿元，其中甲营收25亿元，占总营收25%。'
                '价格为180元/吨，环比-20元/吨，跌幅10%。')
        checks = source_arithmetic_checks(text)
        self.assertEqual({x['kind'] for x in checks}, {'conditional_part_total_share', 'price_period_change'})
        self.assertEqual(price_change_checks(text), [x for x in checks if x['kind']=='price_period_change'])
        self.assertTrue(all(x['scope']=='calculation_only_not_business_verdict' for x in checks))
        self.assertTrue(all('error_type' not in x and 'status' not in x for x in checks))


if __name__ == "__main__":
    unittest.main()
