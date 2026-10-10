import copy
import json
import unittest

from yjcheck.candidate_decisions import (
    MAX_REASON_CHARS,
    SCHEMA_VERSION,
    VERDICTS,
    CandidateDecisionProtocolError,
    load_candidate_decisions,
    parse_candidate_decisions,
)
from yjcheck.text_taxonomy import FINED_ERROR_TYPES


_UNSET = object()


def decision(verdict="error_supported", *, kind="计算错误", reason="同一明确算式不相等。", spans=_UNSET):
    return {"verdict": verdict, "error_type": kind,
            "spans": [{"text": "甲项为2，乙项为3，合计为8。"}] if spans is _UNSET else spans,
            "reason": reason}


class CandidateDecisionsTests(unittest.TestCase):
    def assert_invalid(self, payload, code, index=None, path=None):
        before = copy.deepcopy(payload)
        with self.assertRaises(CandidateDecisionProtocolError) as caught:
            parse_candidate_decisions(payload)
        self.assertEqual(caught.exception.error_code, code)
        self.assertEqual(caught.exception.decision_index, index)
        if path is not None:
            self.assertEqual(caught.exception.path, path)
        self.assertEqual(payload, before)

    def test_schema_and_empty_decisions_do_not_require_normal_sentence_inventory(self):
        self.assertEqual(SCHEMA_VERSION, "candidate-decisions/1.0")
        self.assertEqual(MAX_REASON_CHARS, 400)
        self.assertEqual(load_candidate_decisions('{"decisions":[]}'), [])

    def test_all_verdicts_preserved_in_order_with_all_source_members(self):
        members = [{"text": "上年收入10。"}, {"text": "今年收入20。", "start": 9, "end": 16}]
        payload = {"decisions": [decision(v, spans=copy.deepcopy(members)) for v in VERDICTS]}
        before = copy.deepcopy(payload)
        result = parse_candidate_decisions(payload)
        self.assertEqual(result, before["decisions"])
        self.assertEqual(payload, before)
        self.assertIsNot(result, payload["decisions"])
        result[1]["spans"][0]["text"] = "changed output"
        payload["decisions"][2]["spans"][0]["text"] = "changed input"
        self.assertEqual(payload["decisions"][1], before["decisions"][1])
        self.assertEqual(result[2], before["decisions"][2])

    def test_reason_words_never_override_explicit_verdict(self):
        # This is deliberately semantically inconsistent. Structural validation
        # cannot resolve it; the original record must remain available for audit.
        for verdict in VERDICTS:
            with self.subTest(verdict=verdict):
                item = decision(verdict, reason=" 无冲突，不报告；但不能据此抹去其他独立问题。\n")
                self.assertEqual(parse_candidate_decisions({"decisions": [item]}), [item])

    def test_different_span_type_verdict_or_reason_are_independent_records(self):
        first = decision(reason="需核对。")
        cases = [decision(spans=[{"text": "另外一笔独立交易。"}], reason="需核对。"),
                 decision(kind="数值不一致错误", reason="需核对。"),
                 decision("no_error", reason="需核对。"),
                 decision(reason="需核对另一项独立关系。")]
        self.assertEqual(parse_candidate_decisions({"decisions": [first, *cases]}), [first, *cases])

    def test_duplicate_whole_object_fails_entire_batch_regardless_of_key_order(self):
        for verdict in VERDICTS:
            original = decision(verdict)
            reordered = dict(reversed(list(original.items())))
            reordered["spans"] = [dict(reversed(list(s.items()))) for s in original["spans"]]
            with self.subTest(verdict=verdict):
                self.assert_invalid({"decisions": [original, reordered]}, "candidate_decision_duplicate", 1,
                                    "/decisions/1")

    def test_duplicate_reason_hundreds_of_times_is_not_silently_deduplicated(self):
        self.assert_invalid({"decisions": [decision()] * 300}, "candidate_decision_duplicate", 1)

    def test_span_order_and_original_whitespace_are_not_normalized_for_identity(self):
        first = decision(spans=[{"text": "甲。"}, {"text": "乙。"}], reason=" 原因 ")
        second = decision(spans=[{"text": "乙。"}, {"text": "甲。"}], reason=" 原因 ")
        third = decision(spans=[{"text": "甲。"}, {"text": "乙。"}], reason="原因")
        self.assertEqual(parse_candidate_decisions({"decisions": [first, second, third]}),
                         [first, second, third])

    def test_two_identical_member_quotes_are_left_for_pair_anchoring(self):
        item = decision(kind="冗余语句", spans=[{"text": "完整重复句。"}, {"text": "完整重复句。"}])
        self.assertEqual(parse_candidate_decisions({"decisions": [item]}), [item])

    def test_reason_limit_counts_unicode_characters_without_truncation(self):
        item = decision(reason="证🙂" * (MAX_REASON_CHARS // 2))
        self.assertEqual(len(item["reason"]), MAX_REASON_CHARS)
        self.assertEqual(load_candidate_decisions(json.dumps({"decisions": [item]}, ensure_ascii=False)), [item])
        over = copy.deepcopy(item)
        over["reason"] += "。"
        self.assert_invalid({"decisions": [over]}, "candidate_decision_reason_too_long", 0,
                            "/decisions/0/reason")

    def test_long_explicit_arithmetic_reason_fits_without_an_eighty_character_cut(self):
        reason = "依照原文已明示的同口径封闭关系，" + "100+200+300+400+500+600+700+800+900+1000" * 3 + "不能得到10。"
        self.assertGreater(len(reason), 80)
        self.assertLess(len(reason), MAX_REASON_CHARS)
        self.assertEqual(parse_candidate_decisions({"decisions": [decision(reason=reason)]})[0]["reason"], reason)

    def test_empty_nonstring_or_whitespace_reason_invalid(self):
        for value in ["", " \t\n", None, 3, [], False]:
            with self.subTest(value=value):
                self.assert_invalid({"decisions": [decision(reason=value)]},
                                    "candidate_decision_reason_invalid", 0)

    def test_top_level_is_closed_and_old_errors_cannot_masquerade_as_new_protocol(self):
        for payload in [None, [], {"errors": []}, {}, {"decisions": [], "errors": []},
                        {"decisions": [], "gold": []}, {"decisions": [], "complete": True}]:
            with self.subTest(payload=payload):
                self.assert_invalid(payload, "candidate_decisions_top_level_invalid")
        for value in [None, {}, "", 0, ()]:
            with self.subTest(value=value):
                self.assert_invalid({"decisions": value}, "candidate_decisions_not_list", path="/decisions")

    def test_every_item_has_exact_four_fields_and_no_caller_authority(self):
        for name in ["verdict", "error_type", "spans", "reason"]:
            item = decision()
            del item[name]
            with self.subTest(missing=name):
                self.assert_invalid({"decisions": [item]}, "candidate_decision_fields_invalid", 0)
        for name, value in [("status", "confirmed_error"), ("document_id", "id"),
                            ("gold", {}), ("decision_id", "chosen"), ("source", "verified")]:
            item = decision()
            item[name] = value
            with self.subTest(extra=name):
                self.assert_invalid({"decisions": [item]}, "candidate_decision_fields_invalid", 0)
        for item in [None, [], "candidate", True]:
            with self.subTest(item=item):
                self.assert_invalid({"decisions": [item]}, "candidate_decision_fields_invalid", 0)

    def test_verdict_is_exact_closed_enum(self):
        for value in ["normal", "uncertain", "source_limited", "ERROR_SUPPORTED", " no_error", None, False, []]:
            with self.subTest(value=value):
                self.assert_invalid({"decisions": [decision(value)]}, "candidate_decision_verdict_invalid", 0)

    def test_all_fifteen_exact_types_work_for_all_verdicts(self):
        for kind in FINED_ERROR_TYPES:
            for verdict in VERDICTS:
                with self.subTest(kind=kind, verdict=verdict):
                    item = decision(verdict, kind=kind)
                    self.assertEqual(parse_candidate_decisions({"decisions": [item]}), [item])
        for kind in ["calc_error", "属性值缺失", " 计算错误", "不存在", None, 1]:
            with self.subTest(kind=kind):
                self.assert_invalid({"decisions": [decision(kind=kind)]}, "candidate_decision_type_invalid", 0)

    def test_bad_tail_including_nonerror_fails_batch_without_partial_results(self):
        for verdict in VERDICTS:
            bad = decision(verdict, reason="x" * (MAX_REASON_CHARS + 1))
            with self.subTest(verdict=verdict):
                self.assert_invalid({"decisions": [decision(), bad]},
                                    "candidate_decision_reason_too_long", 1, "/decisions/1/reason")

    def test_spans_required_for_every_verdict_including_nonerrors(self):
        for verdict in VERDICTS:
            for value in [[], None, "原句", {}, ()]:
                with self.subTest(verdict=verdict, value=value):
                    self.assert_invalid({"decisions": [decision(verdict, spans=value)]},
                                        "candidate_decision_spans_invalid", 0)

    def test_only_quote_or_complete_coordinate_span_shape_accepted(self):
        for value in ["原文", {}, {"start": 0, "end": 2}, {"text": "原文", "start": 0},
                      {"text": "原文", "end": 2}, {"text": "原文", "quote": "原文"},
                      {"text": "原文", "start": 0, "end": 2, "page": 1}]:
            with self.subTest(value=value):
                self.assert_invalid({"decisions": [decision(spans=[value])]},
                                    "candidate_decision_span_fields_invalid", 0, "/decisions/0/spans/0")
        for value in ["", " \n", None, False, 1]:
            with self.subTest(value=value):
                self.assert_invalid({"decisions": [decision(spans=[{"text": value}])]},
                                    "candidate_decision_span_text_invalid", 0)

    def test_boolean_float_negative_reversed_and_equal_coordinates_fail(self):
        for start, end in [(False, 3), (0, True), (0.0, 2), (0, 2.0), (-1, 2), (2, 2), (3, 2),
                           ("0", 2), (0, None)]:
            with self.subTest(start=start, end=end):
                self.assert_invalid({"decisions": [decision(spans=[{"text": "原文", "start": start, "end": end}])]},
                                    "candidate_decision_span_coordinates_invalid", 0)

    def test_source_anchor_correctness_is_explicitly_deferred_for_all_verdicts(self):
        for verdict in VERDICTS:
            with self.subTest(verdict=verdict):
                item = decision(verdict, spans=[{"text": "是否在原文中须由调用方核验。", "start": 999, "end": 1000}])
                self.assertEqual(parse_candidate_decisions({"decisions": [item]}), [item])

    def test_strict_json_accepts_surrounding_whitespace_but_not_recovery(self):
        self.assertEqual(load_candidate_decisions(' \n{"decisions":[]}\t'), [])
        for text in ['```json\n{"decisions":[]}\n```', '{"decisions":[]', '{"decisions":[,]}',
                     '{"decisions":[]} explanation', '{"decisions":[]} {"decisions":[]}']:
            with self.subTest(text=text), self.assertRaises(CandidateDecisionProtocolError) as caught:
                load_candidate_decisions(text)
            self.assertEqual(caught.exception.error_code, "candidate_decision_json_invalid")

    def test_duplicate_json_keys_rejected_at_top_item_and_span_levels(self):
        texts = ['{"decisions":[],"decisions":[]}',
                 '{"decisions":[{"verdict":"no_error","verdict":"error_supported","error_type":"计算错误","reason":"理由","spans":[{"text":"原文"}]}]}',
                 '{"decisions":[{"verdict":"no_error","error_type":"计算错误","reason":"理由","spans":[{"text":"甲","text":"乙"}]}]}']
        for text in texts:
            with self.subTest(text=text), self.assertRaises(CandidateDecisionProtocolError) as caught:
                load_candidate_decisions(text)
            self.assertEqual(caught.exception.error_code, "candidate_decision_json_duplicate_key")

    def test_nonfinite_json_and_nonstring_input_are_explicit_protocol_failures(self):
        for token in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(token=token), self.assertRaises(CandidateDecisionProtocolError) as caught:
                load_candidate_decisions('{"decisions":' + token + '}')
            self.assertEqual(caught.exception.error_code, "candidate_decision_json_nonfinite")
        for value in [None, b'{"decisions":[]}', {"decisions": []}]:
            with self.subTest(value=value), self.assertRaises(CandidateDecisionProtocolError) as caught:
                load_candidate_decisions(value)
            self.assertEqual(caught.exception.error_code, "candidate_decision_json_text_invalid")

    def test_safe_exception_does_not_echo_arbitrary_reason_or_unknown_keys(self):
        payload = {"decisions": [decision(reason="PRIVATE_LITERAL" * 50)]}
        with self.assertRaises(CandidateDecisionProtocolError) as caught:
            parse_candidate_decisions(payload)
        self.assertNotIn("PRIVATE_LITERAL", str(caught.exception))
        self.assertEqual(caught.exception.path, "/decisions/0/reason")


if __name__ == "__main__":
    unittest.main()
