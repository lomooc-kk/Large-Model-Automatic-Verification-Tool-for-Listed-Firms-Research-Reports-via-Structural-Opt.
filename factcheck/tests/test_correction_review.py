"""Offline correction-node contracts; no model or network dependency."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from yjcheck.correction_review import correct_text, MAX_INPUT_CHARS, PURPOSE


class MockChat:
    def __init__(self, corrected="原文", *, response=None, error=None):
        self.calls = []
        self.error = error
        self.response = response if response is not None else {
            "content": json.dumps({"corrected_text": corrected}, ensure_ascii=False),
            "trace": {"call_id": "offline-call", "cost_cny": "0.01", "finish_reason": "stop"},
        }

    def __call__(self, messages, *, purpose):
        self.calls.append((deepcopy(messages), purpose))
        if self.error:
            raise self.error
        return deepcopy(self.response)


class CorrectionReviewTests(unittest.TestCase):
    def test_only_source_enters_prompt_and_inputs_are_not_modified(self):
        payload = {"input_text": "利润提升。", "gold": "secret-gold", "corrected_text": "secret-answer",
                   "label": "Fact_Error", "evidence": "not-source", "document_id": "answer-exposing-id"}
        examples = [{"input_text": "甲企业非常非常稳定。", "corrected_text": "甲企业非常稳定。",
                     "gold": "secret-example-metadata"}]
        before = deepcopy((payload, examples))
        chat = MockChat("利润提升。")
        result = correct_text(payload, chat, examples=examples)
        messages, purpose = chat.calls[0]
        self.assertEqual(purpose, PURPOSE)
        self.assertEqual(json.loads(messages[-1]["content"]), {"input_text": "利润提升。"})
        self.assertEqual(json.loads(messages[1]["content"]), {"input_text": examples[0]["input_text"]})
        self.assertEqual(json.loads(messages[2]["content"]), {"corrected_text": examples[0]["corrected_text"]})
        self.assertNotIn("secret", json.dumps(messages))
        self.assertEqual((payload, examples), before)
        self.assertEqual(result["changes"], [])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["evidence_scope"], "source_only")

    def test_structured_and_direct_use_same_full_source_with_single_call(self):
        text = "甲公司2023年收入1亿元。乙公司2024年收入10000万元，预计未来增长。"
        prompts = []
        for variant in ("direct", "structured"):
            chat = MockChat(text)
            result = correct_text({"input_text": text}, chat, variant=variant)
            self.assertEqual(len(chat.calls), 1)
            messages, _ = chat.calls[0]
            self.assertEqual(json.loads(messages[-1]["content"])["input_text"], text)
            self.assertEqual(result["corrected_text"], text)
            self.assertEqual(result["changes"], [])
            self.assertTrue(result["coverage"]["complete"])
            prompts.append(messages[0]["content"])
        self.assertNotEqual(*prompts)
        for boundary in ("Fact_Error", "Word_Error", "Grammar_Error", "Punc_Error",
                         "不同主体、期间、口径不可直接比较", "等价", "预测", "不得臆填", "风格重写"):
            self.assertIn(boundary, prompts[1])

    def test_diff_has_exact_unicode_source_and_target_offsets_and_reconstructs_output(self):
        source = "甲公司🙂营业收人。\n利润利润为12亿元"
        target = "甲公司🙂营业收入。\n利润为12亿元。"
        result = correct_text({"input_text": source}, MockChat(target), variant="structured")
        rebuilt, cursor = [], 0
        self.assertEqual({edit["operation"] for edit in result["changes"]}, {"replace", "delete", "insert"})
        for edit in result["changes"]:
            self.assertEqual(source[edit["start"]:edit["end"]], edit["text"])
            self.assertEqual(target[edit["corrected_start"]:edit["corrected_end"]], edit["replacement"])
            self.assertEqual(edit["status"], "needs_review")
            self.assertNotIn("error_type", edit)
            rebuilt.extend((source[cursor:edit["start"]], edit["replacement"]))
            cursor = edit["end"]
        rebuilt.append(source[cursor:])
        self.assertEqual("".join(rebuilt), target)

    def test_numeric_change_remains_candidate_not_confirmed_fact(self):
        result = correct_text({"input_text": "净利润为1.50亿元。"}, MockChat("净利润为1.55亿元。"))
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(all(edit["status"] == "needs_review" for edit in result["changes"]))
        self.assertNotIn("confirmed", json.dumps(result))

    def test_overlapping_development_sources_or_answers_are_excluded(self):
        source = "营收为１００万元。"
        examples = [
            {"input_text": "营收为100万元。", "corrected_text": "营收为100元。"},
            {"input_text": "提示。营收为100万元。结尾。", "corrected_text": "提示。营收为100元。结尾。"},
            {"input_text": "业务扩大了。", "corrected_text": "营收为100万元。"},
            {"input_text": "无关表达达。", "corrected_text": "无关表达。"},
            {"input_text": "无关 表达达。", "corrected_text": "无关表达。"},
        ]
        chat = MockChat(source)
        result = correct_text({"input_text": source}, chat, examples=examples)
        self.assertEqual(result["examples"], {"provided": 5, "used": 1, "excluded_overlap": 3,
                                              "excluded_duplicate": 1, "excluded_limit": 0})
        self.assertEqual(len(chat.calls[0][0]), 4)

    def test_few_shot_count_and_length_are_bounded_without_truncation(self):
        examples = [{"input_text": "甲" * 3001, "corrected_text": "乙" * 3001}]
        examples += [{"input_text": f"示例{i}有有问题。", "corrected_text": f"示例{i}有问题。"} for i in range(7)]
        chat = MockChat("当前文本。")
        result = correct_text({"input_text": "当前文本。"}, chat, examples=examples)
        self.assertEqual(result["examples"]["used"], 4)
        self.assertEqual(result["examples"]["excluded_limit"], 4)
        self.assertEqual(len(chat.calls[0][0]), 10)

    def test_few_shot_total_characters_are_bounded(self):
        examples = [{"input_text": char * 2500, "corrected_text": char * 2499} for char in "甲乙丙丁"]
        result = correct_text({"input_text": "当前文本。"}, MockChat("当前文本。"), examples=examples)
        self.assertEqual(result["examples"]["used"], 2)
        self.assertEqual(result["examples"]["excluded_limit"], 2)

    def test_malformed_or_partial_outputs_remain_failed_predictions(self):
        responses = ["{}", "not json", '{"corrected_text":null}', '{"corrected_text":[]}',
                     '{"corrected_text":""}', '{"corrected_text":"  "}',
                     '{"corrected_text":"修正", "status":"confirmed_error"}',
                     '{"corrected_text":"甲","corrected_text":"乙"}',
                     '{"corrected_text":"半截"', '[{"corrected_text":"修正"}]']
        for content in responses:
            with self.subTest(content=content):
                result = correct_text({"input_text": "原文"}, MockChat(response={
                    "content": content, "trace": {"cost_cny": "0.01"}}))
                self.assertEqual(result["status"], "failed")
                self.assertIsNone(result["corrected_text"])
                self.assertFalse(result["coverage"]["complete"])
                self.assertEqual(result["coverage"]["total_chars"], 2)
                self.assertEqual(result["traces"][0]["runtime_trace"]["cost_cny"], "0.01")

    def test_truncated_or_failed_runtime_is_rejected_even_with_valid_json(self):
        for trace in ({"finish_reason": "length"}, {"response_content_incomplete": True}, {"status": "error"}):
            result = correct_text({"input_text": "原文"}, MockChat(response={
                "content": '{"corrected_text":"看似完整"}', "trace": trace}))
            self.assertEqual(result["status"], "failed")
            self.assertIsNone(result["corrected_text"])

    def test_provider_failure_retains_safe_budget_trace_without_exception_body(self):
        class ProviderError(RuntimeError):
            trace = {"call_id": "paid-failed-call", "cost_cny": "0.03"}
        result = correct_text({"input_text": "原文"}, MockChat(error=ProviderError("api-key-secret")))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure"]["code"], "model_call_failed")
        self.assertEqual(result["traces"][0]["runtime_trace"]["cost_cny"], "0.03")
        self.assertNotIn("api-key-secret", json.dumps(result))

    def test_invalid_envelope_or_trace_is_failure(self):
        for response in ({"content": None}, {"content": "{}", "trace": ["invalid"]}, {"answer": "原文"}):
            result = correct_text({"input_text": "原文"}, MockChat(response=response))
            self.assertEqual(result["status"], "failed")

    def test_fenced_json_is_accepted_and_corrected_whitespace_is_preserved(self):
        source = " 开头\n末尾  "
        content = "```json\n" + json.dumps({"corrected_text": source}) + "\n```"
        result = correct_text({"input_text": source}, MockChat(response={"content": content}))
        self.assertEqual(result["corrected_text"], source)
        self.assertEqual(result["changes"], [])

    def test_oversize_input_is_not_truncated_or_called(self):
        chat = MockChat()
        result = correct_text({"input_text": "甲" * (MAX_INPUT_CHARS + 1)}, chat)
        self.assertEqual(chat.calls, [])
        self.assertEqual(result["failure"]["code"], "input_too_long")
        self.assertIsNone(result["corrected_text"])

    def test_oversize_correction_is_rejected(self):
        result = correct_text({"input_text": "甲"}, MockChat("乙" * (MAX_INPUT_CHARS * 2 + 1)))
        self.assertEqual(result["failure"]["code"], "correction_too_large")

    def test_blank_input_preserves_text_without_model_call(self):
        for source in ("", " \n\t"):
            chat = MockChat()
            result = correct_text({"input_text": source}, chat)
            self.assertEqual(chat.calls, [])
            self.assertEqual(result["corrected_text"], source)
            self.assertTrue(result["coverage"]["complete"])
            self.assertFalse(result["coverage"]["model_ran"])

    def test_missing_model_is_explicit_failure_not_identity_prediction(self):
        result = correct_text({"input_text": "原文"}, None)
        self.assertEqual(result["failure"]["code"], "model_unavailable")
        self.assertIsNone(result["corrected_text"])
        self.assertFalse(result["coverage"]["model_ran"])

    def test_bad_api_arguments_fail_before_model_call(self):
        chat = MockChat()
        for payload in ({}, {"input_text": 1}, None):
            with self.assertRaises(ValueError):
                correct_text(payload, chat)
        with self.assertRaises(ValueError):
            correct_text({"input_text": "原文"}, chat, variant="unknown")
        for examples in (None, {"input_text": "样例"}, [None], [{"input_text": "样例"}],
                         [{"input_text": "", "corrected_text": "样例"}]):
            with self.assertRaises(ValueError):
                correct_text({"input_text": "原文"}, chat, examples=examples)
        self.assertEqual(chat.calls, [])


if __name__ == "__main__":
    unittest.main()
