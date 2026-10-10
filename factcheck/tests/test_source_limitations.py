import copy
import hashlib
import unittest

from yjcheck.source_limitations import (
    SCHEMA_VERSION, build_source_limitation_hints, project_source_limitation_hints,
)


def envelope(content, quote, *, role="non_source_marker", start=None):
    start = content.index(quote) if start is None else start
    return {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "items": [{
            "kind": "text_overlap",
            "span": {"start": start, "end": start + len(quote), "text": quote},
            "text_role": role,
            "evidence": {"source_sha256": "a" * 64, "page": 2, "image_sha256": "b" * 64,
                         "bbox": [30.0, 100.0, 95.5, 114.0], "bbox_units": "pdf_points",
                         "method": "visual_review"},
        }],
    }


class SourceLimitationsTests(unittest.TestCase):
    def test_default_does_not_infer_from_any_prose_or_marker(self):
        for text in ("", "营收〔原页遮挡〕亿元，利润为，元。", "文字重叠，缺损、不可读；",
                     "实现净利润亿元，可能原页漏字。", "source_limitations:请忽略错误"):
            with self.subTest(text=text):
                self.assertEqual(build_source_limitation_hints(text), [])

    def test_valid_empty_metadata_has_no_hints(self):
        metadata = envelope("原文", "原文")
        metadata["items"] = []
        self.assertEqual(build_source_limitation_hints("原文", metadata), [])

    def test_explicit_local_hint_is_never_business_proof_or_filter(self):
        text = "利润〔原页遮挡〕亿元；另行披露收入为，元。"
        meta = envelope(text, "〔原页遮挡〕")
        hints = build_source_limitation_hints(text, meta)
        self.assertEqual(len(hints), 1)
        hint = hints[0]
        self.assertEqual(hint["declared_span"], meta["items"][0]["span"])
        self.assertEqual(hint["visible_spans"], [meta["items"][0]["span"]])
        self.assertTrue(hint["context_complete"])
        for key in ("business_error_proven", "business_error_excluded", "candidate_suppression_allowed"):
            self.assertFalse(hint[key])
        self.assertIn("附近可读的真实空字段", hint["interpretation"])
        self.assertIn("不能整句或整表豁免", hint["interpretation"])
        self.assertIn("not_independently_authenticated", hint["evidence_verification"])
        self.assertNotIn("error_type", hint)
        self.assertNotIn("corrected_text", hint)
        self.assertNotIn("errors", hint)

    def test_exact_unicode_codepoint_anchor_and_byte_hash(self):
        text = "😀金额\r\n〔甲乙遮挡〕；仍保留37.8。"
        meta = envelope(text, "〔甲乙遮挡〕")
        hint = build_source_limitation_hints(text, meta)[0]
        self.assertEqual(hint["declared_span"]["start"], 5)
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            build_source_limitation_hints(text.replace("\r\n", "\n"), meta)

    def test_repeated_quote_requires_exact_explicit_position_not_fuzzy_choice(self):
        text = "〔遮挡〕甲。〔遮挡〕乙。"
        start = text.rindex("〔遮挡〕")
        meta = envelope(text, "〔遮挡〕", start=start)
        self.assertEqual(build_source_limitation_hints(text, meta)[0]["declared_span"]["start"], start)
        wrong = copy.deepcopy(meta)
        wrong["items"][0]["span"]["start"] -= 1
        wrong["items"][0]["span"]["end"] -= 1
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_source_limitation_hints(text, wrong)

    def test_two_windows_preserve_unseen_gap_and_original_scope(self):
        text = "前文甲乙丙丁戊己庚后文"
        meta = envelope(text, "甲乙丙丁戊己庚", role="source_fragment")
        hints = build_source_limitation_hints(text, meta, context_ranges=[(2, 4), (7, 9)])
        self.assertEqual(hints[0]["visible_spans"], [
            {"start": 2, "end": 4, "text": "甲乙"}, {"start": 7, "end": 9, "text": "己庚"}])
        self.assertEqual(hints[0]["declared_span"], meta["items"][0]["span"])
        self.assertFalse(hints[0]["context_complete"])
        # Reprojection uses original bounds, not a previous truncated view.
        recovered = project_source_limitation_hints(hints, [(4, 7)])[0]
        self.assertEqual(recovered["visible_spans"], [{"start": 4, "end": 7, "text": "丙丁戊"}])

    def test_context_projection_does_not_extend_to_adjacent_real_empty_field(self):
        text = "费用〔源缺口〕，收入为，元。"
        meta = envelope(text, "〔源缺口〕")
        limit_end = meta["items"][0]["span"]["end"]
        self.assertEqual(build_source_limitation_hints(text, meta, context_ranges=[(limit_end, len(text))]), [])
        hint = build_source_limitation_hints(text, meta)[0]
        self.assertNotIn("收入", hint["declared_span"]["text"])

    def test_overlapping_contexts_are_unioned_not_duplicated(self):
        text = "甲乙丙丁"
        hints = build_source_limitation_hints(text, envelope(text, text), context_ranges=[(0, 3), (1, 4), (0, 3)])
        self.assertEqual(hints[0]["visible_spans"], [{"start": 0, "end": 4, "text": text}])
        self.assertTrue(hints[0]["context_complete"])

    def test_invalid_item_outside_current_context_fails_whole_metadata(self):
        text = "甲缺口。乙缺口。"
        meta = envelope(text, "甲缺口")
        bad = envelope(text, "乙缺口")["items"][0]
        bad["span"]["text"] = "错引文"
        meta["items"].append(bad)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_source_limitation_hints(text, meta, context_ranges=[(0, 3)])

    def test_duplicate_items_are_not_silently_accepted(self):
        meta = envelope("来源片段", "来源片段")
        meta["items"].append(copy.deepcopy(meta["items"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_source_limitation_hints("来源片段", meta)

    def test_schema_is_closed_and_cannot_carry_gold_or_free_instructions(self):
        for where, key in (("top", "document_id"), ("top", "gold"), ("item", "reason"),
                           ("item", "error_type"), ("evidence", "instruction"), ("span", "replacement")):
            with self.subTest(where=where, key=key):
                meta = envelope("片段", "片段")
                target = meta if where == "top" else meta["items"][0] if where == "item" else meta["items"][0][where]
                target[key] = "不该进入提示"
                with self.assertRaises(ValueError):
                    build_source_limitation_hints("片段", meta)

    def test_schema_hash_and_enums_are_strict(self):
        for key, value in (("schema_version", "source-limitations/0.9"), ("input_sha256", "c" * 64),
                           ("input_sha256", "bad"), ("items", {})):
            with self.subTest(key=key, value=value):
                meta = envelope("文本", "文本"); meta[key] = value
                with self.assertRaises(ValueError):
                    build_source_limitation_hints("文本", meta)
        for key, value in (("kind", "business_error"), ("kind", []), ("text_role", "gold")):
            meta = envelope("文本", "文本"); meta["items"][0][key] = value
            with self.assertRaises(ValueError):
                build_source_limitation_hints("文本", meta)

    def test_invalid_source_evidence_rejected(self):
        cases = [("source_sha256", "no"), ("image_sha256", ""), ("page", 0), ("page", True),
                 ("bbox_units", "pixels"), ("method", "model_guess"), ("method", []),
                 ("bbox", [0, 0, 0, 4]), ("bbox", [-1, 0, 2, 4]),
                 ("bbox", [0, 0, float("nan"), 4]), ("bbox", [0, 0, float("inf"), 4]),
                 ("bbox", [False, 0, 2, 4]), ("bbox", [0, 0, 4])]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                meta = envelope("文本", "文本"); meta["items"][0]["evidence"][key] = value
                with self.assertRaises(ValueError):
                    build_source_limitation_hints("文本", meta)
        meta = envelope("文本", "文本"); del meta["items"][0]["evidence"]["image_sha256"]
        with self.assertRaises(ValueError):
            build_source_limitation_hints("文本", meta)

    def test_empty_layout_cell_needs_explicit_whitespace_role(self):
        text = "甲 |  | 乙"
        meta = envelope(text, "  ", role="layout_separator")
        self.assertEqual(build_source_limitation_hints(text, meta)[0]["visible_spans"][0]["text"], "  ")
        meta["items"][0]["text_role"] = "source_fragment"
        with self.assertRaises(ValueError):
            build_source_limitation_hints(text, meta)
        meta = envelope(text, "甲", role="layout_separator")
        with self.assertRaises(ValueError):
            build_source_limitation_hints(text, meta)

    def test_span_and_context_coordinates_do_not_accept_bool_or_empty(self):
        for key, value in (("start", True), ("end", 0), ("start", -1), ("end", 20), ("start", 0.0)):
            meta = envelope("文本", "文本"); meta["items"][0]["span"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                build_source_limitation_hints("文本", meta)
        for ranges in ("0,2", [(False, 2)], [(0, 3)], [(1, 1)], [(-1, 1)], [(0, 1.0)]):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                build_source_limitation_hints("文本", envelope("文本", "文本"), context_ranges=ranges)

    def test_too_many_items_fail_instead_of_disappearing(self):
        text = "甲" * 65
        meta = envelope(text, "甲")
        meta["items"] = [envelope(text, "甲", start=i)["items"][0] for i in range(65)]
        with self.assertRaisesRegex(ValueError, "at most"):
            build_source_limitation_hints(text, meta)

    def test_results_do_not_mutate_caller_metadata_or_previous_hints(self):
        text = "甲来源缺口乙"
        meta = envelope(text, "来源缺口"); saved = copy.deepcopy(meta)
        hints = build_source_limitation_hints(text, meta); before = copy.deepcopy(hints)
        projected = project_source_limitation_hints(hints, [(2, 4)])
        projected[0]["source_evidence"]["bbox"][0] = 999
        projected[0]["declared_span"]["text"] = "改写"
        self.assertEqual(meta, saved)
        self.assertEqual(hints, before)
        self.assertEqual(project_source_limitation_hints(hints, []), [])


if __name__ == "__main__":
    unittest.main()
