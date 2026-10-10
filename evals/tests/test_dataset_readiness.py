"""Dataset admission failure paths and isolated denominator contracts; offline."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evals.dataset_readiness import (DatasetReadinessError, audit_dataset,
    build_inference_payload, require_dataset_ready, score_clfec)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), encoding="utf-8")


class DatasetAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fined(self, archive=False):
        folder = self.root / "inputs" if archive else self.root
        split = "research442" if archive else "dev"
        write_rows(folder / f"inputs.{split}.jsonl", [{"doc_id": "d", "content": "甲乙"}])
        write_rows(folder / f"gold.{split}.jsonl", [{"document_id": "d", "errors": [
            {"type": "用词不当", "scorable": True, "spans": [{"start": 0, "end": 1, "text": "甲"}]},
            {"type": "用词不当", "scorable": False, "spans": []},
        ]}])
        return {"task": "fined", "root": str(self.root), "role": "regression",
                "scorer": "fined_strict", "scope": "all_hints", "split": "dev"}

    def external(self):
        task = "financebench_claim_verification"
        for split, suffix in (("dev", "A"), ("test", "B")):
            write_rows(self.root / f"data/{task}/inputs.{split}.jsonl", [
                {"sample_id": suffix, "claim": "收入为10元", "evidence_text": "收入为9元"}])
            write_rows(self.root / f"data/{task}/gold.{split}.jsonl", [
                {"sample_id": suffix, "has_error": True, "label": 1, "label_text": "inconsistent"}])
            write_rows(self.root / f"splits/{split}.jsonl", [{"sample_id": suffix, "group_id": suffix}])
        return {"task": "external:" + task, "root": str(self.root), "role": "development",
                "scorer": "external:" + task, "split": "dev"}

    def test_fined_preserves_excluded_denominator_and_explicit_scopes(self):
        config = self.fined()
        report = require_dataset_ready(config)
        self.assertEqual(report["counts"]["documents"], 1)
        self.assertEqual(report["counts"]["scorable_gold_errors"], 1)
        self.assertEqual(report["counts"]["excluded_gold_errors"], 1)
        self.assertEqual(report["protocol"]["scope"], "all_hints")
        config.pop("scope")
        self.assertEqual(audit_dataset(config)["status"], "blocked")

    def test_known_442_cannot_be_renamed_as_holdout_role(self):
        config = self.fined(archive=True)
        config["role"] = "evaluation"
        with self.assertRaises(DatasetReadinessError) as caught:
            require_dataset_ready(config)
        codes = {i["code"] for i in caught.exception.report["blocking_issues"]}
        self.assertIn("known_exposed_442_requires_regression", codes)

    def test_wrong_scorer_and_nested_answer_fields_block(self):
        config = self.fined()
        config["scorer"] = "clfec_edits"
        self.assertEqual(audit_dataset(config)["status"], "blocked")
        config["scorer"] = "fined_strict"
        write_rows(self.root / "inputs.dev.jsonl", [{"doc_id": "d", "content": "甲乙", "meta": {"answer": "甲"}}])
        report = audit_dataset(config)
        self.assertIn("unsafe_or_incomplete_input", {i["code"] for i in report["blocking_issues"]})

    def test_invalid_gold_span_and_missing_gold_block(self):
        config = self.fined()
        write_rows(self.root / "gold.dev.jsonl", [{"document_id": "x", "errors": [
            {"type": "wrong", "spans": [{"start": 0, "end": 1, "text": "错"}]}]}])
        report = audit_dataset(config)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("input_gold_id_mismatch", {i["code"] for i in report["blocking_issues"]})

    def test_claim_requires_evidence_and_never_exposes_answer_coded_id(self):
        row = {"sample_id": "fb-cv-A", "claim": "错值", "evidence_text": "正确证据"}
        payload = build_inference_payload("external:financebench_claim_verification", row)
        self.assertEqual(payload, {"claim": "错值", "evidence_text": "正确证据"})
        for fields in (["evidence_text"], ["claim", "evidence_text", "sample_id"]):
            with self.assertRaises(ValueError):
                build_inference_payload("external:financebench_claim_verification", row, fields)

    def test_external_task_stays_outside_fined_and_validates_source_groups(self):
        config = self.external()
        self.assertEqual(audit_dataset(config)["status"], "ready")
        config["scorer"] = "fined_strict"
        self.assertEqual(audit_dataset(config)["status"], "blocked")
        config["scorer"] = config["task"]
        write_rows(self.root / "splits/test.jsonl", [{"sample_id": "B", "group_id": "A"}])
        report = audit_dataset(config)
        self.assertIn("source_group_crosses_splits", {i["code"] for i in report["blocking_issues"]})

    def test_evaluation_requires_hash_bound_role_assignments(self):
        config = self.fined()
        config["role"] = "evaluation"
        self.assertEqual(audit_dataset(config)["status"], "blocked")
        assignment = self.root / "roles.json"
        write_json(assignment, {"assignments": [{"document_id": "d", "role": "evaluation",
            "source_group": "source-1", "input_sha256": hashlib.sha256("甲乙".encode()).hexdigest(), "exposure": "unseen"}]})
        config["split_manifest"] = str(assignment)
        self.assertEqual(audit_dataset(config)["status"], "ready")
        write_rows(self.root / "inputs.dev.jsonl", [{"doc_id": "d", "content": "甲乙丙"}])
        self.assertEqual(audit_dataset(config)["status"], "blocked")

    def test_clfec_category_cannot_be_submitted_as_dataset_split(self):
        config = {"task": "clfec", "root": str(self.root), "role": "regression",
                  "scorer": "clfec_edits", "split": "fec_only"}
        report = audit_dataset(config)
        self.assertIn("clfec_category_is_not_split", {i["code"] for i in report["blocking_issues"]})


class ClfecScoringTests(unittest.TestCase):
    def records(self):
        inputs = [{"sample_id": "bad", "input_text": "甲乙"}, {"sample_id": "clean", "input_text": "干净"}]
        golds = [{"sample_id": "bad", "corrected_text": "丙丁", "cors": [
            {"start": 0, "end": 1, "candidate_word": "丙"}, {"start": 1, "end": 2, "candidate_word": "丁"}]},
            {"sample_id": "clean", "corrected_text": "干净", "cors": []}]
        return inputs, golds

    def test_perfect_text_has_perfect_edit_credit_despite_adjacent_annotation_segmentation(self):
        inputs, golds = self.records()
        predictions = [{"sample_id": g["sample_id"], "corrected_text": g["corrected_text"]} for g in golds]
        score = score_clfec(inputs, golds, predictions)
        self.assertEqual(score["canonical_edit_correction"]["f1"], 1)
        self.assertEqual(score["exact_match_rate"], 1)
        self.assertEqual(score["native_annotated_edits"], 2)

    def test_missing_predictions_remain_failures_and_do_not_shrink_denominator(self):
        inputs, golds = self.records()
        score = score_clfec(inputs, golds, [])
        self.assertEqual(score["documents"], 2)
        self.assertEqual(score["missing_predictions"], ["bad", "clean"])
        self.assertEqual(score["paragraph_detection"]["false_negative"], 1)
        self.assertEqual(score["canonical_edit_correction"]["false_negative"], 1)
        self.assertEqual(score["exact_match_rate"], 0)
        self.assertEqual(score["missing_clean_predictions"], 1)
        self.assertIsNone(score["false_positive_rate"])

    def test_false_positive_rate_uses_only_clean_denominator(self):
        inputs, golds = self.records()
        score = score_clfec(inputs, golds, [{"sample_id": "bad", "corrected_text": "丙丁"},
                                          {"sample_id": "clean", "corrected_text": "不干净"}])
        self.assertEqual(score["false_positive_rate"], 1)
        self.assertEqual(score["clean_denominator"], 1)
        score = score_clfec(inputs[:1], golds[:1], [])
        self.assertIsNone(score["false_positive_rate"])

    def test_duplicate_and_unknown_predictions_are_rejected(self):
        inputs, golds = self.records()
        for predictions in ([{"sample_id": "unknown", "corrected_text": "x"}],
                            [{"sample_id": "bad", "corrected_text": "x"}] * 2):
            with self.assertRaises(ValueError):
                score_clfec(inputs, golds, predictions)


if __name__ == "__main__":
    unittest.main()
