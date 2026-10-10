import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from yjcheck.execution_scope import check_active
from yjcheck.model import ModelConfig
from yjcheck.model_runtime import BudgetLedger, RuntimeSettings
from yjcheck.pi_bridge import AgentLimits, run_pi


class PiBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.settings = RuntimeSettings(input_cny_per_mtok="1", output_cny_per_mtok="2",
                                        context_tokens=8000, max_output_tokens=64,
                                        ledger=self.path / "ledger.sqlite3")
        self.config = ModelConfig("http://127.0.0.1:1/v1", "fixture", "test-secret")

    def script(self, body):
        path = self.path / "protocol.py"
        path.write_text("import json,sys,time\nstart=json.loads(input())\n"
                        "def emit(x): print(json.dumps(x),flush=True)\n" + body, encoding="utf-8")
        return [sys.executable, str(path)]

    def test_unknown_usage_keeps_reservation_and_cannot_claim_completion(self):
        command = self.script("""
emit({'type':'model_request','id':'r','estimatedInputTokens':100,'maxOutputTokens':64})
p=json.loads(input())
emit({'type':'model_usage','id':'r','reservationId':p['reservationId'],'usageKnown':False,'status':'error'})
emit({'type':'final','protocolVersion':1,'requestId':start['requestId'],'status':'completed'})
""")
        result = run_pi({}, lambda n,a: {}, self.config, settings=self.settings, command=command)
        self.assertEqual(result["status"], "stopped")
        self.assertAlmostEqual(result["budget"]["accounted_cny"], 0.000228)
        self.assertNotIn("test-secret", json.dumps(result))

    def test_tool_timeout_returns_promptly_and_cancel_scope_blocks_later_work(self):
        command = self.script("""
emit({'type':'tool_call','id':'t','name':'detect_document','args':{}})
input()
""")
        import threading
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()
        attempted_after_cancel = []
        def tool(name, args):
            entered.set()
            try:
                # Hold the tool until the caller has observed the timeout.
                # Process startup time is unrelated to the tool's deadline.
                release.wait(5)
                check_active()
                attempted_after_cancel.append(True)
            finally:
                completed.set()
            return {}
        try:
            result = run_pi({}, tool, self.config, settings=self.settings, command=command,
                            limits=AgentLimits(timeout_seconds=2, tool_timeout_seconds=0.05))
            self.assertTrue(entered.is_set())
            self.assertEqual(result["stop_reason"], "pi_tool_timeout")
            self.assertFalse(completed.is_set(), "Timeout must return before the held tool completes")
        finally:
            release.set()
        self.assertTrue(completed.wait(1))
        self.assertEqual(attempted_after_cancel, [])

    def test_budget_denial_does_not_create_call(self):
        command = self.script("""
emit({'type':'model_request','id':'r','estimatedInputTokens':100,'maxOutputTokens':64})
p=json.loads(input())
assert p['allowed'] is False
emit({'type':'final','protocolVersion':1,'requestId':start['requestId'],'status':'stopped','stopReason':'budget'})
""")
        result = run_pi({}, lambda n,a: {}, self.config, settings=self.settings, command=command,
                        limits=AgentLimits(max_cost_cny="0"))
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["budget"]["calls"], 0)

    def test_explicit_thinking_enabled_survives_bridge(self):
        command = self.script("""
assert start['model']['thinking']=='low'
emit({'type':'final','protocolVersion':1,'requestId':start['requestId'],'status':'needs_review'})
""")
        self.config.thinking = "enabled"
        result = run_pi({}, lambda n,a: {}, self.config, settings=self.settings, command=command)
        self.assertEqual(result["effective_thinking"], "low")

    def test_detector_large_output_does_not_block_pi_first_round(self):
        from dataclasses import replace
        command = self.script("""
assert start['model']['maxOutputTokens']==4096
assert start['limits']['maxInputTokens']+4096<=start['limits']['maxTotalTokens']
emit({'type':'final','protocolVersion':1,'requestId':start['requestId'],'status':'needs_review'})
""")
        result = run_pi({}, lambda n,a: {}, self.config,
                        settings=replace(self.settings, context_tokens=131072, max_output_tokens=65536), command=command)
        self.assertEqual(result["max_output_tokens"], 4096)

    def test_invalid_cost_is_a_clear_configuration_error(self):
        for cost in ("abc", "NaN", "Infinity", "-1", "1001"):
            with self.subTest(cost=cost), self.assertRaisesRegex(ValueError, "invalid_agent_cost_limit"):
                AgentLimits(max_cost_cny=cost).validate()


if __name__ == "__main__":
    unittest.main()
