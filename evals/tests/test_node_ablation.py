"""Admission must precede replay; replay must use the production node."""
from unittest import TestCase, mock

from evals import run_node_ablation as ablation
from evals.dataset_readiness import DatasetReadinessError


class NodeAblationContractTests(TestCase):
    def test_blocked_admission_prevents_any_parent_data_read(self):
        failure = {"status": "blocked", "blocking_issues": [{"code": "blocked_test"}]}
        with mock.patch.object(ablation, "_admit", side_effect=DatasetReadinessError(failure)), \
             mock.patch.object(ablation, "_rows") as load_rows, \
             mock.patch.object(ablation, "_read") as load_json:
            with self.assertRaises(DatasetReadinessError):
                ablation.run_ablation("source-package", "separate-output")
            load_rows.assert_not_called()
            load_json.assert_not_called()

    def test_replay_calls_production_normalizer_and_preserves_source_report(self):
        candidate = {"id": "p", "error_type": "redundant", "status": "needs_review", "spans": [
            {"start": 0, "end": 1, "text": "甲"}, {"start": 1, "end": 2, "text": "乙"}]}
        report = {"document_id": "d", "errors": [candidate]}
        original = ablation.span_normalization.normalize_candidate_spans
        with mock.patch.object(ablation.span_normalization, "normalize_candidate_spans", wraps=original) as node:
            normalized = ablation._normalize_report(report, "甲乙")
            node.assert_called_once_with(candidate, "甲乙")
        self.assertEqual(normalized["errors"][0]["spans"], [{"start": 0, "end": 2, "text": "甲乙"}])
        self.assertEqual(len(report["errors"][0]["spans"]), 2)

    def test_child_admission_uses_required_gate(self):
        with mock.patch.object(ablation, "require_dataset_ready", return_value={"status": "ready"}) as gate:
            self.assertEqual(ablation._admission_child({"task": "fined"}), {"status": "ready"})
            gate.assert_called_once_with({"task": "fined"})
