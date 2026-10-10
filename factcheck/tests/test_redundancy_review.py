from copy import deepcopy
import hashlib
import json
import unittest

from yjcheck.redundancy_review import PURPOSE, review_redundancy
from yjcheck.review_hints import all_review_hints


SOURCE = "公司业绩承压。公司业绩承压，主要因为需求下降。其他独立术语问题。另一段正常信息。"


def span(text, source=SOURCE, start=0):
    begin = source.index(text, start)
    return {"start": begin, "end": begin + len(text), "text": text}


FIRST = span("公司业绩承压。")
SECOND = span("公司业绩承压，主要因为需求下降。")
WIDE = {"start": 0, "end": SECOND["end"], "text": SOURCE[:SECOND["end"]]}


def base_report():
    target = {"id": "r1", "error_type": "冗余语句", "status": "needs_review",
              "reason": "两句都称承压，可能重复。", "spans": [deepcopy(WIDE)],
              "original_spans": [{"text": WIDE["text"]}],
              "represented_model_candidates": [{"candidate_ref": "model:2:0", "candidate": {"reason": "原始理由"}}],
              "source_issue_key": "old-proof", "verification_spans": [deepcopy(FIRST)],
              "verified_by": "localization_only", "span_normalization": "source_adjacent_v1"}
    return {
        "errors": [target,
                   {"id": "term", "error_type": "术语误用", "status": "needs_review", "reason": "术语可能不当", "spans": [span("其他独立术语问题。")]},
                   {"id": "confirmed", "error_type": "冗余语句", "status": "confirmed_error", "reason": "规则确认", "spans": [deepcopy(WIDE)]}],
        "raw_candidates": [{"candidate_ref": "model:2:0", "candidate": {"error_type": "冗余语句", "reason": "raw-DONT-SEND", "spans": [{"text": WIDE["text"]}]}}],
        "rejected_candidates": [{"candidate": {"error_type": "冗余语句", "reason": "拒绝定位-DONT-SEND", "spans": [{"text": "原文没有的文本"}]}, "reason": "anchor_failure"}],
        "model_candidate_representation": [{"candidate_ref": "model:2:0", "error_id": "r1", "source_issue_key": "old-proof", "raw_candidate_sha256": "unchanged-hash"}],
        "traces": [{"purpose": "text_review.hybrid", "job_index": 2, "status": "ok", "runtime_trace": {"call_id": "base-call", "cost_cny": "0.01"}}],
        "coverage": {"complete": True, "execution_complete": True, "model_complete": True,
                     "candidate_quality_complete": True, "processed_ranges": [[0, len(SOURCE)]],
                     "processed_chars": len(SOURCE), "truncation_reasons": [],
                     "planned_model_calls": 3, "finished_model_calls": 3},
        "gold": "DONT-SEND-gold", "document_id": "DONT-SEND-doc-id",
    }


def retain(error_id="r1"):
    return {"error_id": error_id, "action": "retain", "reason": "仍存在局部重复疑点，保留复核。", "evidence": []}


def withdraw(error_id="r1"):
    return {"error_id": error_id, "action": "withdraw", "reason": "第一句是主题，第二句新增需求下降原因。",
            "normal_context": "topic_expansion", "evidence": [deepcopy(FIRST), deepcopy(SECOND)]}


def receipt(messages, **overrides):
    return {"call_id": "synthetic-review-call", "cost_cny": "0.02", "finish_reason": "stop", "status": "ok",
            "purpose": PURPOSE, "request_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
            **overrides}


class Chat:
    def __init__(self, decisions=None, *, response=None, error=None):
        self.decisions = [retain()] if decisions is None else decisions
        self.response = response
        self.error = error
        self.calls = []
        self.last_response = None

    def __call__(self, messages, *, purpose):
        self.calls.append((deepcopy(messages), purpose))
        if self.error:
            raise self.error
        if self.response is not None:
            self.last_response = self.response(messages) if callable(self.response) else deepcopy(self.response)
            return deepcopy(self.last_response)
        decisions = self.decisions(messages) if callable(self.decisions) else self.decisions
        self.last_response = {"content": json.dumps({"decisions": decisions}, ensure_ascii=False), "trace": receipt(messages)}
        return deepcopy(self.last_response)


class TestRedundancyReview(unittest.TestCase):
    def assert_unchanged_payloads(self, before, after):
        for key in ("raw_candidates", "rejected_candidates"):
            self.assertEqual(json.dumps(before[key], ensure_ascii=False), json.dumps(after[key], ensure_ascii=False))

    def assert_incomplete(self, before, after):
        self.assertEqual(after["errors"], before["errors"])
        self.assertEqual(after["model_candidate_representation"], before["model_candidate_representation"])
        self.assert_unchanged_payloads(before, after)
        self.assertFalse(after["redundancy_review"]["complete"])
        self.assertEqual(after["redundancy_review"]["status"], "incomplete")
        for key in ("complete", "execution_complete", "model_complete"):
            self.assertFalse(after["coverage"][key])
        self.assertEqual(after["coverage"]["processed_ranges"], before["coverage"]["processed_ranges"])
        self.assertEqual(after["coverage"]["finished_model_calls"], 3)
        self.assertEqual(after["coverage"]["planned_model_calls"], 4)
        self.assertIn("redundancy_review_incomplete", after["coverage"]["truncation_reasons"])
        self.assertEqual(after["redundancy_review"]["decisions"], [])

    def test_one_whole_source_call_and_strict_input_whitelist(self):
        before = base_report()
        untouched = deepcopy(before)
        chat = Chat()
        out = review_redundancy(SOURCE, before, chat)
        self.assertEqual(before, untouched)
        self.assertEqual(len(chat.calls), 1)
        messages, purpose = chat.calls[0]
        self.assertEqual(purpose, PURPOSE)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(set(payload), {"input_text", "candidates"})
        self.assertEqual(payload["input_text"], SOURCE)
        self.assertEqual(len(payload["candidates"]), 1)
        self.assertEqual(set(payload["candidates"][0]), {"error_id", "spans", "reason"})
        self.assertNotIn("DONT-SEND", json.dumps(messages))
        self.assertEqual(out["errors"], before["errors"])
        out["errors"][0]["spans"][0]["text"] = "changed only in deep copy"
        self.assertEqual(before, untouched)

    def test_success_trace_counters_and_raw_response_are_exact(self):
        chat = Chat(response=lambda messages: {
            "content": '  {"decisions":' + json.dumps([retain()], ensure_ascii=False) + '}\n',
            "trace": receipt(messages, call_id="synthetic-id", cost_cny="0.023", custom={"x": 1})})
        before = base_report()
        out = review_redundancy(SOURCE, before, chat)
        stage = out["redundancy_review"]
        self.assertEqual(stage["raw_response"], chat.last_response["content"])
        self.assertEqual(stage["response_sha256"], hashlib.sha256(chat.last_response["content"].encode()).hexdigest())
        self.assertEqual(stage["runtime_trace"], chat.last_response["trace"])
        self.assertEqual(out["traces"][:-1], before["traces"])
        trace = out["traces"][-1]
        self.assertEqual((trace["job_index"], trace["status"], trace["purpose"]), (3, "ok", PURPOSE))
        self.assertEqual(trace["runtime_trace"], stage["runtime_trace"])
        self.assertEqual(out["coverage"]["planned_model_calls"], 4)
        self.assertEqual(out["coverage"]["finished_model_calls"], 4)
        self.assertTrue(out["coverage"]["complete"])
        self.assertTrue(stage["base_execution_complete"])

    def test_withdraw_has_full_audit_and_no_dangling_raw_representation(self):
        before = base_report()
        out = review_redundancy(SOURCE, before, Chat([withdraw()]))
        self.assertEqual(out["errors"], before["errors"][1:])
        self.assert_unchanged_payloads(before, out)
        audit = out["redundancy_review"]["decisions"][0]
        self.assertEqual(audit["before"], before["errors"][0])
        self.assertIsNone(audit["after"])
        self.assertEqual(audit["evidence"], [{**s, "anchor_method": "provided_exact_offsets"} for s in (FIRST, SECOND)])
        self.assertEqual(out["redundancy_review"]["withdrawn"][0]["error"], before["errors"][0])
        rep = out["model_candidate_representation"][0]
        self.assertIsNone(rep["error_id"])
        self.assertEqual(rep["previous_error_id"], "r1")
        self.assertEqual(rep["review_record_id"], "r1")
        self.assertEqual(rep["disposition"], "withdrawn_by_redundancy_review")
        self.assertEqual(rep["raw_candidate_sha256"], "unchanged-hash")
        self.assertEqual(audit["representation_before"], before["model_candidate_representation"])
        hints = all_review_hints(deepcopy(out))["errors"]
        self.assertFalse(any(h.get("id") == "r1" for h in hints))
        self.assertEqual(len(hints), 3)  # unchanged term + confirmed + pre-existing rejection

    def test_revise_two_members_stable_id_and_originals_preserved(self):
        before = base_report()
        old_proofs = {"source_anchor_proof": {"proof": "old"}, "paired_source_spans": [FIRST, SECOND],
                      "anchor_method": "old_proof", "source_structure": {"proof_status": "confirmed_error"},
                      "source_rule_proofs": [{"proof": "old"}]}
        before["errors"][0].update(deepcopy(old_proofs))
        decision = {"error_id": "r1", "action": "revise", "reason": "局部重复仍需裁定。",
                    "spans": [FIRST, SECOND], "evidence": [FIRST, SECOND]}
        out = review_redundancy(SOURCE, before, Chat([decision]), normalize_spans=False)
        revised = out["errors"][0]
        self.assertEqual(revised["id"], "r1")
        self.assertEqual(revised["spans"], [FIRST, SECOND])
        self.assertEqual(revised["status"], "needs_review")
        self.assertEqual(revised["original_spans"], before["errors"][0]["original_spans"])
        self.assertEqual(revised["represented_model_candidates"], before["errors"][0]["represented_model_candidates"])
        self.assertEqual(revised["spans_before_redundancy_review"], before["errors"][0]["spans"])
        self.assertNotIn("source_issue_key", revised)
        self.assertNotIn("verification_spans", revised)
        for key in old_proofs:
            self.assertNotIn(key, revised)
        self.assertEqual(revised["detector_id"], "model.redundancy_review")
        self.assertEqual(out["redundancy_review"]["decisions"][0]["before"], before["errors"][0])
        self.assertEqual(out["errors"][1:], before["errors"][1:])
        self.assert_unchanged_payloads(before, out)

    def test_invalid_decisions_are_atomic_noop(self):
        invalid = []
        unknown = withdraw("not-a-target")
        invalid += [[unknown], [], [retain(), retain()], [{**retain(), "status": "confirmed_error"}],
                    [{**retain(), "action": "confirm"}], [{**retain(), "reason": "  "}],
                    [{**withdraw(), "normal_context": "model_says_it_is_correct"}],
                    [{k: v for k, v in withdraw().items() if k != "reason"}],
                    [{**withdraw(), "evidence": [FIRST]}], [{**withdraw(), "evidence": [FIRST, FIRST]}],
                    [{**withdraw(), "evidence": [FIRST, {**SECOND, "text": "伪造引文"}]}],
                    [{**withdraw(), "evidence": [FIRST, {**SECOND, "start": True}]}],
                    [{**withdraw(), "evidence": [FIRST, {**SECOND, "end": 9999}]}],
                    [{**withdraw(), "evidence": [FIRST, {**SECOND, "extra": "unsupported"}]}],
                    [{**withdraw(), "evidence": [span("其他独立术语问题。"), span("另一段正常信息。")] }]]
        for decisions in invalid:
            with self.subTest(decisions=decisions):
                before = base_report()
                chat = Chat(decisions)
                out = review_redundancy(SOURCE, before, chat)
                self.assert_incomplete(before, out)
                self.assertEqual(len(chat.calls), 1)
                self.assertEqual(out["traces"][-1]["runtime_trace"]["cost_cny"], "0.02")

    def test_partial_batch_never_commits_first_valid_withdrawal(self):
        before = base_report()
        second = deepcopy(before["errors"][0])
        second.update(id="r2")
        before["errors"].append(second)
        for decisions in ([withdraw()], [withdraw(), {**retain("r2"), "reason": ""}]):
            with self.subTest(decisions=decisions):
                self.assert_incomplete(before, review_redundancy(SOURCE, before, Chat(decisions)))

    def test_revision_rejects_missing_overlapping_or_unrelated_members(self):
        valid = {"error_id": "r1", "action": "revise", "reason": "修订定位。", "evidence": [FIRST, SECOND], "spans": [FIRST, SECOND]}
        bad = [dict(valid, spans=[FIRST]), dict(valid, spans=[FIRST, FIRST]),
               dict(valid, spans=[FIRST, SECOND, span("另一段正常信息。")]),
               dict(valid, spans=[FIRST, span("另一段正常信息。")]),
               dict(valid, spans=[span("其他独立术语问题。"), span("另一段正常信息。")],
                    evidence=[FIRST, SECOND, span("其他独立术语问题。"), span("另一段正常信息。")])]
        for decision in bad:
            with self.subTest(decision=decision):
                before = base_report()
                self.assert_incomplete(before, review_redundancy(SOURCE, before, Chat([decision])))

    def test_malformed_truncated_and_provider_failure_envelopes_preserve_originals(self):
        bad_contents = ['{}', 'not json', '{"decisions":null}', '{"decisions":[]',
                        '{"decisions":[],"decisions":[]}', '[{"decisions":[]}]']
        responses = [(lambda messages, value=value: {"content": value, "trace": receipt(messages)}) for value in bad_contents]
        responses += [{"content": None}, {"content": "{}", "trace": []}]
        for trace in ({"finish_reason": "length"}, {"response_content_incomplete": True}, {"status": "failed"}):
            responses.append(lambda messages, trace=trace: {"content": json.dumps({"decisions": [retain()]}), "trace": receipt(messages, **trace)})
        for response in responses:
            with self.subTest(response=response):
                before = base_report()
                self.assert_incomplete(before, review_redundancy(SOURCE, before, Chat(response=response)))

    def test_timeout_is_not_retried_and_budget_trace_survives(self):
        class Timeout(TimeoutError):
            trace = {"call_id": "failed-paid-call", "cost_cny": "0.03"}
        before = base_report()
        chat = Chat(error=Timeout("secret-token-must-not-leak"))
        out = review_redundancy(SOURCE, before, chat)
        self.assert_incomplete(before, out)
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(out["redundancy_review"]["runtime_trace"], Timeout.trace)
        self.assertNotIn("secret-token-must-not-leak", json.dumps(out))

    def test_over_budget_is_no_call_whole_source_noop(self):
        before = base_report()
        chat = Chat()
        out = review_redundancy(SOURCE, before, chat, max_input_tokens=1)
        self.assert_incomplete(before, out)
        self.assertEqual(chat.calls, [])
        self.assertFalse(out["redundancy_review"]["model_ran"])
        self.assertEqual(out["traces"][-1]["status"], "not_run")
        self.assertIsNone(out["redundancy_review"]["raw_response"])
        self.assertEqual(out["redundancy_review"]["failure"]["code"], "input_exceeds_context_budget")

    def test_unavailable_client_is_explicit_incomplete(self):
        before = base_report()
        self.assert_incomplete(before, review_redundancy(SOURCE, before, None))

    def test_noneligible_candidates_are_not_sent_or_mutated(self):
        before = base_report()
        before["errors"][0]["spans"][0]["text"] = "not-source"
        original = deepcopy(before)
        chat = Chat()
        out = review_redundancy(SOURCE, before, chat)
        self.assertEqual(chat.calls, [])
        self.assertEqual(out["errors"], before["errors"])
        self.assertEqual(before, original)
        self.assertEqual(out["redundancy_review"]["status"], "not_applicable")
        self.assertEqual(len(out["redundancy_review"]["skipped"]), 1)
        self.assertEqual(out["coverage"]["planned_model_calls"], 3)
        self.assertEqual(out["traces"], before["traces"])

    def test_invalid_anchor_flag_and_boolean_offsets_are_excluded(self):
        for edit in ({"invalid_anchor": True}, {"spans": [{**FIRST, "start": False}]}):
            with self.subTest(edit=edit):
                before = base_report()
                before["errors"][0].update(edit)
                chat = Chat()
                out = review_redundancy(SOURCE, before, chat)
                self.assertEqual(chat.calls, [])
                self.assertEqual(out["errors"], before["errors"])

    def test_success_does_not_erase_base_incomplete_coverage(self):
        before = base_report()
        before["coverage"].update(complete=False, execution_complete=False, model_complete=False,
                                  truncation_reasons=["earlier_call_failed"], processed_ranges=[[0, 12]])
        out = review_redundancy(SOURCE, before, Chat())
        self.assertTrue(out["redundancy_review"]["complete"])
        for key in ("complete", "execution_complete", "model_complete"):
            self.assertFalse(out["coverage"][key])
        self.assertEqual(out["coverage"]["processed_ranges"], [[0, 12]])
        self.assertEqual(out["coverage"]["truncation_reasons"], ["earlier_call_failed"])

    def test_missing_id_is_generated_stably_without_mutating_retained_error(self):
        before = base_report()
        before["errors"][0].pop("id")
        before["model_candidate_representation"] = []
        def make_decisions(messages):
            target = json.loads(messages[-1]["content"])["candidates"][0]
            return [retain(target["error_id"])]
        results = [review_redundancy(SOURCE, before, Chat(make_decisions)) for _ in range(2)]
        ids = [x["redundancy_review"]["decisions"][0]["error_id"] for x in results]
        self.assertEqual(ids[0], ids[1])
        self.assertTrue(ids[0].startswith("generated:"))
        self.assertEqual(results[0]["errors"], before["errors"])

    def test_duplicate_existing_id_fails_before_model_call(self):
        before = base_report()
        before["errors"][1]["id"] = "r1"
        chat = Chat()
        out = review_redundancy(SOURCE, before, chat)
        self.assert_incomplete(before, out)
        self.assertEqual(chat.calls, [])
        self.assertEqual(out["redundancy_review"]["failure"]["code"], "nonunique_source_error_id")

    def test_unicode_offsets_are_source_characters(self):
        source = "🙂标题。🙂标题，增加原因。"
        left, right = span("🙂标题。", source), span("🙂标题，增加原因。", source)
        before = base_report()
        before["errors"][0]["spans"] = deepcopy([left, right])
        decision = dict(withdraw(), evidence=deepcopy([left, right]))
        out = review_redundancy(source, before, Chat([decision]))
        self.assertTrue(out["redundancy_review"]["complete"])
        decision["evidence"][1]["start"] += 1
        self.assert_incomplete(before, review_redundancy(source, before, Chat([decision])))

    def test_unique_text_evidence_and_revision_members_are_exactly_anchored(self):
        before = base_report()
        decision = {"error_id": "r1", "action": "revise", "reason": "仅复核原文两成员。",
                    "evidence": [{"text": FIRST["text"]}, {"text": SECOND["text"]}],
                    "spans": [{"text": FIRST["text"]}, {"text": SECOND["text"]}]}
        out = review_redundancy(SOURCE, before, Chat([decision]), normalize_spans=False)
        self.assertTrue(out["redundancy_review"]["complete"])
        self.assertEqual(out["errors"][0]["spans"], [FIRST, SECOND])
        audit = out["redundancy_review"]["decisions"][0]
        self.assertEqual(audit["evidence"], [{**s, "anchor_method": "unique_exact_quote"} for s in (FIRST, SECOND)])
        self.assertEqual(json.loads(out["redundancy_review"]["raw_response"])["decisions"][0], decision)

    def test_text_only_withdraw_handles_unicode_without_model_offsets(self):
        source = "🙂标题。🙂标题，增加原因。"
        left, right = span("🙂标题。", source), span("🙂标题，增加原因。", source)
        before = base_report()
        before["errors"][0]["spans"] = [left, right]
        decision = dict(withdraw(), evidence=[{"text": left["text"]}, {"text": right["text"]}])
        out = review_redundancy(source, before, Chat([decision]))
        self.assertTrue(out["redundancy_review"]["complete"])
        self.assertEqual(out["redundancy_review"]["decisions"][0]["evidence"][1]["start"], len(left["text"]))

    def test_ambiguous_quote_requires_explicit_valid_offsets_or_unique_numbering(self):
        source = "1）同一公司。2）同一公司。"
        first, second = span("1）同一公司。", source), span("2）同一公司。", source)
        before = base_report()
        before["errors"][0]["spans"] = [first, second]
        decision = dict(withdraw(), evidence=[{"text": "同一公司。"}, {"text": "2）同一公司。"}])
        out = review_redundancy(source, before, Chat([decision]))
        self.assert_incomplete(before, out)
        self.assertEqual(out["redundancy_review"]["failure"]["code"], "ambiguous_source_quote")
        decision["evidence"][0] = {"text": "1）同一公司。"}
        self.assertTrue(review_redundancy(source, before, Chat([decision]))["redundancy_review"]["complete"])
        decision["evidence"][0] = span("同一公司。", source)
        self.assertTrue(review_redundancy(source, before, Chat([decision]))["redundancy_review"]["complete"])

    def test_wrong_explicit_offset_is_not_repaired_by_unique_quote(self):
        before = base_report()
        decision = dict(withdraw(), evidence=[{**FIRST, "start": FIRST["start"] + 1}, {"text": SECOND["text"]}])
        out = review_redundancy(SOURCE, before, Chat([decision]))
        self.assert_incomplete(before, out)
        self.assertEqual(out["redundancy_review"]["failure"]["code"], "invalid_source_evidence")

    def test_revised_display_uses_existing_adjacent_only_normalization(self):
        for gap in ("", "\n \t", "但这句提供独立新事实。"):
            with self.subTest(gap=gap):
                source = FIRST["text"] + gap + SECOND["text"]
                first, second = span(FIRST["text"], source), span(SECOND["text"], source)
                before = base_report()
                before["errors"][0]["spans"] = [{"start": 0, "end": len(source), "text": source}]
                decision = {"error_id": "r1", "action": "revise", "reason": "仅审阅两条原文成员。",
                            "spans": [first, second], "evidence": [first, second]}
                out = review_redundancy(source, before, Chat([decision]))
                after = out["errors"][0]
                expected = [{"start": 0, "end": len(source), "text": source}] if not gap or gap.isspace() else [first, second]
                self.assertEqual(after["spans"], expected)
                audit = out["redundancy_review"]["decisions"][0]
                self.assertEqual(audit["after"], after)
                self.assertEqual(len(audit["decision"]["spans"]), 2)
                self.assertEqual(len(audit["evidence"]), 2)
                self.assertEqual(out["errors"][1:], before["errors"][1:])
                self.assert_unchanged_payloads(before, out)
                unnormalized = review_redundancy(source, before, Chat([decision]), normalize_spans=False)
                self.assertEqual(unnormalized["errors"][0]["spans"], [first, second])
                self.assertNotIn("span_normalization", unnormalized["errors"][0])

    def test_explicit_unfinished_or_unknown_runtime_status_is_not_success(self):
        for status in ("reserved", "pending", "unknown", "success", None, False):
            with self.subTest(status=status):
                before = base_report()
                response = {"content": json.dumps({"decisions": [withdraw()]}),
                            "trace": {"status": status, "call_id": "not-completed", "cost_cny": "0.02"}}
                out = review_redundancy(SOURCE, before, Chat(response=response))
                self.assert_incomplete(before, out)
                self.assertEqual(out["redundancy_review"]["runtime_trace"], response["trace"])

    def test_missing_cost_identity_or_request_receipt_rejects_valid_withdrawal(self):
        overrides = [None, {"call_id": ""}, {"call_id": None}, {"purpose": "text_review.detect"},
                     {"cost_cny": "NaN"}, {"cost_cny": "Infinity"}, {"cost_cny": "-0.01"},
                     {"cost_cny": None}, {"cost_cny": True}, {"request_sha256": "other-request"}]
        for change in overrides:
            with self.subTest(change=change):
                before = base_report()
                def response(messages):
                    return {"content": json.dumps({"decisions": [withdraw()]}),
                            "trace": {} if change is None else receipt(messages, **change)}
                chat = Chat(response=response)
                self.assert_incomplete(before, review_redundancy(SOURCE, before, chat))
        for missing in ("status", "call_id", "purpose", "cost_cny", "request_sha256"):
            with self.subTest(missing=missing):
                before = base_report()
                def response(messages):
                    trace = receipt(messages)
                    del trace[missing]
                    return {"content": json.dumps({"decisions": [withdraw()]}), "trace": trace}
                self.assert_incomplete(before, review_redundancy(SOURCE, before, Chat(response=response)))

    def test_a_second_stage_cannot_overwrite_first_audit(self):
        out = review_redundancy(SOURCE, base_report(), Chat())
        with self.assertRaisesRegex(ValueError, "already has"):
            review_redundancy(SOURCE, out, Chat())

    def test_invalid_call_contracts_fail_without_call(self):
        for budget in (0, -1, True, "16000"):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                review_redundancy(SOURCE, base_report(), Chat(), max_input_tokens=budget)
        for option in (None, "true", 1, 0):
            with self.subTest(normalize_spans=option), self.assertRaises(ValueError):
                review_redundancy(SOURCE, base_report(), Chat(), normalize_spans=option)


if __name__ == "__main__":
    unittest.main()
