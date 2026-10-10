"""Frontend entry-point guards: old examples must not enter new text inference.

All text, models, UI state and files are synthetic and local. These tests do not
open the real legacy corpus, initialize a real budgeted client, or call services.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "frontend"), str(ROOT / "factcheck/src"), str(ROOT / "pdfparse/src")]

import text_review_view as page
import yjcheck.agent_workflow as agent_workflow


class FakeStreamlit:
    def __init__(self, use_pi):
        self.use_pi = use_pi
        self.session_state = {}
        self.errors = []

    def caption(self, *args, **kwargs):
        pass

    def file_uploader(self, *args, **kwargs):
        return None

    def text_area(self, *args, **kwargs):
        return "测试公司本期收入为100万元。"

    def checkbox(self, label, *, key, **kwargs):
        if key == "text_review_use_pi":
            return self.use_pi
        assert key == "text_review_use_model"
        return True

    def text_input(self, *args, **kwargs):
        return ""

    def button(self, label, **kwargs):
        assert label == "检查文本"
        assert not kwargs.get("disabled")
        return True

    def error(self, message):
        self.errors.append(message)


@pytest.mark.parametrize("use_pi", [False, True], ids=["direct-model-entry", "pi-entry"])
def test_text_page_never_reads_or_passes_legacy_examples(tmp_path, monkeypatch, use_pi):
    # Relocate the page's filesystem root so any accidental legacy access or
    # report write stays inside this test's private directory.
    fake_frontend = tmp_path / "frontend"
    fake_frontend.mkdir()
    monkeypatch.setattr(page, "__file__", str(fake_frontend / "text_review_view.py"))
    old_paths = [tmp_path / "data/v2/dataset/examples.dev.json",
                 fake_frontend / "data/v2/dataset/examples.dev.json"]
    for path in old_paths:
        path.parent.mkdir(parents=True)
        path.write_text('[{"content":"SYNTHETIC_LEGACY_SENTINEL"}]', encoding="utf-8")
        assert path.is_file()

    legacy_reads = []
    original_read_text, original_open = Path.read_text, Path.open

    def reject_legacy(path):
        if path.name == "examples.dev.json":
            legacy_reads.append(str(path))
            raise AssertionError("Text review attempted to read diagnostic-only legacy examples")

    def guarded_read_text(path, *args, **kwargs):
        reject_legacy(path)
        return original_read_text(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        reject_legacy(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "open", guarded_open)
    fake_st = FakeStreamlit(use_pi)
    monkeypatch.setattr(page, "st", fake_st)

    report = {"document_id": "synthetic-report", "errors": [], "coverage": {"model_ran": True}}
    fake_client = SimpleNamespace(settings=SimpleNamespace(context_tokens=32000, max_output_tokens=4096))
    config = Mock(return_value=object())
    client_factory = Mock(return_value=fake_client)
    direct = Mock(return_value=report)
    pi = Mock(return_value=(report, {}))
    rendered = Mock()
    monkeypatch.setattr(page.ModelConfig, "from_env", config)
    monkeypatch.setattr(page, "BudgetedChatClient", client_factory)
    monkeypatch.setattr(page, "detect_text", direct)
    monkeypatch.setattr(agent_workflow, "run_agent_text", pi)
    monkeypatch.setattr(page, "show_text_report", rendered)

    page.text_review_page()

    invoked, unused = (pi, direct) if use_pi else (direct, pi)
    invoked.assert_called_once()
    unused.assert_not_called()
    assert invoked.call_args.kwargs["examples"] == []
    assert invoked.call_args.args == ("测试公司本期收入为100万元。",)
    assert legacy_reads == []
    assert not fake_st.errors
    assert fake_st.session_state["text_review_result"] == report
    config.assert_called_once()
    if use_pi:
        client_factory.assert_not_called()
        assert invoked.call_args.kwargs["kb_dir"] is None
    else:
        client_factory.assert_called_once()
        assert invoked.call_args.kwargs["chat"] is fake_client
    output_dir = Path(fake_st.session_state["text_review_dir"])
    assert output_dir.is_relative_to(fake_frontend)
    assert (output_dir / "text_review.json").is_file()
    rendered.assert_called_once_with(report, str(output_dir))
