import json
import unittest

from yjcheck.table_unit_context import eps_unit_comparisons
from yjcheck.text_review import detect_text


REFERENCE = '项目 | 2025A | 2026E | 2027E\n摊薄每股收益（元） | 2.31 | 3.42 | 4.53\n'
TARGET = '指标 | 2024A | 2025A | 2026E | 2027E\nEPS(X) | 1.20 | 2.31 | 3.42 | 4.53\n'


def chat_errors(errors):
    def chat(messages, *, purpose):
        return {'content': json.dumps({'errors': errors}, ensure_ascii=False), 'trace': {}}
    return chat


class TableUnitContextTests(unittest.TestCase):
    def assert_source_spans(self, text, value):
        if isinstance(value, dict):
            if {'start', 'end', 'text'} <= value.keys():
                self.assertEqual(text[value['start']:value['end']], value['text'])
            for item in value.values():
                self.assert_source_spans(text, item)
        elif isinstance(value, list):
            for item in value:
                self.assert_source_spans(text, item)

    def test_matching_year_subset_is_source_evidence_and_local_label_only(self):
        text = '😀研究预测\r\n' + REFERENCE + '\n估值比率\n' + TARGET
        result = eps_unit_comparisons(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['label']['text'], 'EPS(X)')
        self.assertEqual(result[0]['references'][0]['matching_years'], ['2025A', '2026E', '2027E'])
        self.assertFalse(result[0]['business_confirmation'])
        self.assert_source_spans(text, result)
        json.dumps(result)

    def test_reordered_columns_join_by_year_not_position(self):
        target = '项目 | 2027E | 2025A\nEPS(倍) | 4.53 | 2.31\n'
        self.assertEqual(len(eps_unit_comparisons(REFERENCE + target)), 1)

    def test_no_reference_no_header_or_only_one_matching_year_is_insufficient(self):
        cases = [TARGET, REFERENCE, 'EPS(X) | 2.31 | 3.42\n',
                 REFERENCE + '指标 | 2023A | 2025A\nEPS(X) | 1.2 | 2.31\n']
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(eps_unit_comparisons(text), [])

    def test_actual_estimated_and_bare_years_never_equated(self):
        for old, new in [('2025A', '2025E'), ('2026E', '2026A'), ('2027E', '2027')]:
            target = TARGET.replace(old, new)
            # Two remaining exact columns can still support a comparison.
            result = eps_unit_comparisons(REFERENCE + target)
            self.assertEqual(len(result), 1)
            self.assertNotIn(old, result[0]['references'][0]['matching_years'])
        self.assertEqual(eps_unit_comparisons(REFERENCE + TARGET.replace('2025A', '2025E').replace('2026E', '2026A')), [])

    def test_any_shared_year_with_different_value_disqualifies_reference(self):
        self.assertEqual(eps_unit_comparisons(REFERENCE + TARGET.replace('3.42', '3.43')), [])

    def test_equivalent_decimal_spelling_and_finite_currency_units(self):
        for unit in ['元', '人民币元/股', '港元', '美元/股']:
            text = REFERENCE.replace('（元）', f'（{unit}）') + TARGET.replace('2.31', '2.310')
            self.assertEqual(len(eps_unit_comparisons(text)), 1)
        text = REFERENCE.replace('2.31', '1,200.00') + TARGET.replace('2.31', '1200')
        self.assertEqual(len(eps_unit_comparisons(text)), 1)

    def test_markdown_and_tab_tables_preserve_exact_spans(self):
        for delimiter in [' | ', '\t']:
            text = (REFERENCE + TARGET).replace(' | ', delimiter)
            if delimiter == ' | ':
                text = '\n'.join('| ' + line + ' |' for line in text.splitlines())
            result = eps_unit_comparisons(text)
            self.assertEqual(len(result), 1)
            self.assert_source_spans(text, result)

    def test_empty_corner_tab_header_is_allowed_but_interior_empty_is_not(self):
        text = '\t2025A\t2026E\nEPS(元)\t2.31\t3.42\n\t2025A\t2026E\nEPS(X)\t2.31\t3.42'
        self.assertEqual(len(eps_unit_comparisons(text)), 1)
        for broken in [TARGET.replace('2.31 | ', '| '), TARGET.replace('2025A | ', '| '),
                       TARGET.replace('2.31', '-')]:
            self.assertEqual(eps_unit_comparisons(REFERENCE + broken), [])

    def test_duplicate_year_and_malformed_numeric_columns_are_not_guessed(self):
        for target in [TARGET.replace('2026E', '2025A'), TARGET.replace('2.31', '2..31'),
                       TARGET.replace('2.31', '2.31%'), TARGET.replace('2.31', '2,31')]:
            self.assertEqual(eps_unit_comparisons(REFERENCE + target), [])

    def test_prose_or_unknown_heading_breaks_inherited_year_context(self):
        for separator in ['另一家公司', '这里只介绍指标口径。', '异常 | 少一列']:
            target = TARGET.replace('\nEPS', '\n' + separator + '\nEPS')
            self.assertEqual(eps_unit_comparisons(REFERENCE + target), [])

    def test_known_subheading_and_markdown_separator_keep_explicit_columns(self):
        for separator in ['估值比率', '| --- | --- | --- | --- | --- |']:
            target = TARGET.replace('\nEPS', '\n' + separator + '\nEPS')
            self.assertEqual(len(eps_unit_comparisons(REFERENCE + target)), 1)

    def test_asserted_monetary_eps_pe_pb_and_unknown_unit_are_not_targets(self):
        for label in ['EPS(元)', 'PE(X)', 'PB(X)', 'EPS(%)', 'EPS(未知)', '每股净资产(X)']:
            self.assertEqual(eps_unit_comparisons(REFERENCE + TARGET.replace('EPS(X)', label)), [])

    def test_example_and_negated_context_is_not_a_business_assertion(self):
        for prefix in ['示例', '请勿误写为如下表', '假设情形']:
            self.assertEqual(eps_unit_comparisons(REFERENCE + prefix + '\n' + TARGET), [])
        self.assertEqual(eps_unit_comparisons(REFERENCE + TARGET.replace('指标 |', '示例 |')), [])

    def test_case_and_fullwidth_label_unit_spelling(self):
        for label in ['eps(x)', '每股收益（倍）', 'EPS（X）']:
            self.assertEqual(len(eps_unit_comparisons(REFERENCE + TARGET.replace('EPS(X)', label))), 1)

    def test_hybrid_candidate_requires_review_and_carries_exact_evidence(self):
        text = REFERENCE + TARGET
        report = detect_text(text, document_id='synthetic:unit', chat=chat_errors([]))
        matches = [x for x in report['errors'] if x['detector_id'] == 'hybrid.table_eps_unit']
        self.assertEqual(len(matches), 1)
        finding = matches[0]
        self.assertEqual(finding['status'], 'needs_review')
        self.assertEqual(finding['error_type'], '数值单位错误')
        self.assertEqual([s['text'] for s in finding['spans']], ['EPS(X)'])
        self.assertFalse(finding['source_unit_comparison']['business_confirmation'])
        self.assertGreaterEqual(len(finding['evidence']), 5)
        self.assert_source_spans(text, finding)

    def test_model_direct_is_not_given_the_new_rule_candidate(self):
        result = detect_text(REFERENCE + TARGET, document_id='synthetic:unit',
                             detector='model_direct', chat=chat_errors([]))
        self.assertEqual(result['errors'], [])

    def test_model_same_label_fuses_but_separate_number_issue_is_preserved(self):
        text = REFERENCE + TARGET
        raw = [{'error_type': '数值单位错误', 'spans': [{'text': 'EPS(X)'}], 'reason': '单位需要核对。'},
               {'error_type': '数值不一致错误', 'spans': [{'text': '1.20'}], 'reason': '数字需要核对。'}]
        result = detect_text(text, document_id='synthetic:unit', chat=chat_errors(raw))
        self.assertEqual(len(result['raw_candidates']), 2)
        self.assertEqual(len(result['model_candidate_representation']), 2)
        self.assertEqual(len(result['errors']), 2)
        self.assertEqual(len([x for x in result['errors'] if x['error_type'] == '数值单位错误']), 1)

    def test_whole_row_unit_allegation_does_not_fuse_on_overlap(self):
        raw = [{'error_type': '数值单位错误', 'spans': [{'text': TARGET.splitlines()[1]}],
                'reason': '整行另有单位口径需要核对。'}]
        result = detect_text(REFERENCE + TARGET, document_id='synthetic:unit', chat=chat_errors(raw))
        self.assertEqual(len(result['errors']), 2)
        self.assertTrue(all(x['status'] == 'needs_review' for x in result['errors']))


if __name__ == '__main__':
    unittest.main()
