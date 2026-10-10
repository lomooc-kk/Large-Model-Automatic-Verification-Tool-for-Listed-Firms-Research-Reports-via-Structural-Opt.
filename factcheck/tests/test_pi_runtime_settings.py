"""Explicit Pi settings must govern every nested detector call and ledger row."""
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "factcheck/src"), str(ROOT / "pdfparse/src")]

from yjcheck.model import ModelConfig, extract_with_model
from yjcheck.model_runtime import BudgetLedger, RuntimeSettings
from yjcheck.models import Block, Document
from yjcheck.pipeline import check_documents


class PiRuntimeSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = RuntimeSettings(
            input_cny_per_mtok="3", output_cny_per_mtok="7", context_tokens=16384,
            max_output_tokens=128, budget_cny="1", ledger=self.root / "shared.sqlite3")
        self.environment = replace(self.settings, input_cny_per_mtok="99", output_cny_per_mtok="99",
                                   ledger=self.root / "environment.sqlite3")
        self.config = ModelConfig("http://127.0.0.1:1/v1", "local-fixture", review_text=True)
        self.report = Document("report", "a" * 64, "run", "report.txt", "report", "测试公司", "2024FY",
                               [Block("paragraph", "公司2024年度营业收入为10亿元。", paragraph=1)])
        self.requests = []

    def opener(self, request, **kwargs):
        self.requests.append(json.loads(request.data))
        payload = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"facts": [], "errors": []})}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10}}
        return io.BytesIO(json.dumps(payload).encode())

    def test_paired_extraction_and_text_review_use_one_explicit_ledger_and_prices(self):
        with patch.object(RuntimeSettings, "from_env", return_value=self.environment) as env, \
                patch("urllib.request.urlopen", side_effect=self.opener):
            result = check_documents(self.report, [], self.config, runtime_settings=self.settings)
        env.assert_not_called()
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(request["max_tokens"] == 128 for request in self.requests))
        self.assertEqual(result["model_traces"][0]["status"], "ok")
        self.assertTrue(result["text_review"]["coverage"]["model_ran"])
        ledger = BudgetLedger(self.settings.ledger, self.settings.budget_cny)
        with ledger.connect() as db:
            rows = db.execute("SELECT charge_micro, status, trace FROM calls").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(charge == 130 and status == "ok" for charge, status, _ in rows))
        traces = [json.loads(row[2]) for row in rows]
        self.assertEqual({trace["purpose"] for trace in traces}, {"facts-v1", "text_review.detect"})
        self.assertTrue(all(trace["input_cny_per_mtok"] == "3" and
                            trace["output_cny_per_mtok"] == "7" for trace in traces))
        self.assertFalse(self.environment.ledger.exists())

    def test_explicit_zero_budget_blocks_both_nested_detectors(self):
        settings = replace(self.settings, budget_cny="0")
        with patch.object(RuntimeSettings, "from_env", return_value=self.environment) as env, \
                patch("urllib.request.urlopen", side_effect=self.opener):
            result = check_documents(self.report, [], self.config, runtime_settings=settings)
        env.assert_not_called()
        self.assertEqual(self.requests, [])
        self.assertEqual(result["model_traces"][0]["status"], "error")
        self.assertFalse(result["text_review"]["coverage"]["model_ran"])
        self.assertFalse(result["summary"]["complete"])
        self.assertEqual(BudgetLedger(settings.ledger, "0").summary()["calls"], 0)
        self.assertFalse(self.environment.ledger.exists())

    def test_direct_extraction_uses_explicit_settings(self):
        with patch.object(RuntimeSettings, "from_env", return_value=self.environment) as env, \
                patch("urllib.request.urlopen", side_effect=self.opener):
            _, traces = extract_with_model(self.report, self.config, runtime_settings=self.settings)
        env.assert_not_called()
        self.assertEqual(traces[0]["runtime"]["cost_cny"], "0.00013")
        self.assertEqual(BudgetLedger(self.settings.ledger, "1").summary()["calls"], 1)
        self.assertFalse(self.environment.ledger.exists())

    def test_omitted_settings_preserve_existing_environment_default(self):
        with patch.object(RuntimeSettings, "from_env", return_value=self.environment) as env, \
                patch("urllib.request.urlopen", side_effect=self.opener):
            _, traces = extract_with_model(self.report, self.config)
        env.assert_called_once_with(local=True)
        self.assertEqual(traces[0]["runtime"]["cost_cny"], "0.00297")
        self.assertEqual(BudgetLedger(self.environment.ledger, "1").summary()["calls"], 1)
        self.assertFalse(self.settings.ledger.exists())


if __name__ == "__main__":
    unittest.main()
