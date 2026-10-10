"""Proposed v2.1 synthetic transport tests; no model or labels."""
from copy import deepcopy
import json

import pytest

from yjcheck.redundancy_review import review_redundancy
from test_redundancy_review import Chat, SOURCE, FIRST, SECOND, base_report
from test_redundancy_review_context import normal, invoke, assert_atomic_failure, mechanical_fixture


def test_v21_bare_unique_quotes_are_equivalent_but_raw_strings_remain_audited():
    raw = normal()
    for unit in raw["context_units"]:
        unit["quote"] = unit["quote"]["text"]
    out, _ = invoke([raw])
    stage = out["redundancy_review"]
    assert stage["schema_version"] == "redundancy-review/2.1"
    assert stage["complete"]
    assert json.loads(stage["raw_response"])["decisions"][0] == raw
    assert all(u["quote"]["anchor_method"] == "unique_exact_quote"
               for u in stage["decisions"][0]["decision"]["context_units"])


@pytest.mark.parametrize("contexts", [["topic_expansion"], ["topic_expansion", "summary_or_conclusion"]])
def test_v21_all_normal_context_categories_are_preserved_without_selection(contexts):
    raw = normal()
    raw["normal_context"] = contexts
    original = deepcopy(raw)
    out, _ = invoke([raw])
    assert out["redundancy_review"]["complete"]
    audit = out["redundancy_review"]["decisions"][0]
    assert audit["decision"]["normal_context"] == contexts
    assert isinstance(audit["decision"]["normal_context"], list)
    assert audit["action"] == "withdraw"
    assert json.loads(out["redundancy_review"]["raw_response"])["decisions"][0] == original
    assert raw == original


@pytest.mark.parametrize("contexts", [[], ["topic_expansion", "topic_expansion"], ["unknown"],
                                    ["topic_expansion", 1], [None], [True], [["topic_expansion"]],
                                    {}, "", "unknown"])
def test_v21_empty_duplicate_unknown_or_nonstring_contexts_fail_transaction(contexts):
    before = base_report()
    raw = normal()
    raw["normal_context"] = contexts
    out, _ = invoke([raw], before=before)
    assert_atomic_failure(out, before)
    assert out["redundancy_review"]["failure"]["code"] == "invalid_normal_context"


@pytest.mark.parametrize("quote", ["", " ", "原文中并不存在的引文。", " " + FIRST["text"],
                                  FIRST["text"].replace("。", "."), 1, None, []])
def test_v21_bare_quotes_are_not_trimmed_rewritten_or_invented(quote):
    before = base_report()
    raw = normal()
    raw["context_units"][0]["quote"] = quote
    out, _ = invoke([raw], before=before)
    assert_atomic_failure(out, before)


def test_v21_ambiguous_string_does_not_pick_an_occurrence_or_repair_wrong_offset():
    source, before, raw = mechanical_fixture()
    raw["context_units"][0]["quote"] = "行业"
    out, _ = invoke([raw], source=source, before=before)
    assert_atomic_failure(out, before)
    assert out["redundancy_review"]["failure"]["code"] == "ambiguous_source_quote"
    raw["context_units"][0]["quote"] = {"start": 1, "end": len(source), "text": source}
    out, _ = invoke([raw], source=source, before=before)
    assert_atomic_failure(out, before)


def test_v21_bare_member_strings_still_require_unique_source_and_two_members():
    source = "①本期经营持续承压。②本期经营持续承压。"
    before = base_report()
    before["errors"][0]["spans"] = [{"start": 0, "end": len(source), "text": source}]
    raw = {"error_id": "r1", "context_units": [{"role": "parallel_item", "quote": source}],
           "new_information": [], "verdict": "redundant", "reason": "相邻条目正文复写。",
           "members": ["①本期经营持续承压。", "②本期经营持续承压。"]}
    out, _ = invoke([raw], source=source, before=before)
    assert out["redundancy_review"]["complete"]
    assert out["errors"][0]["status"] == "needs_review"
    raw["members"] = ["本期经营持续承压。", "本期经营持续承压。"]
    out, _ = invoke([raw], source=source, before=before)
    assert_atomic_failure(out, before)


def test_v21_unicode_strings_use_exact_character_offsets():
    source = "🙂公司本期经营承压。需求下降解释承压原因。"
    before = base_report()
    before["errors"][0]["spans"] = [{"start": 0, "end": len(source), "text": source}]
    raw = normal()
    raw["context_units"][0]["quote"] = "🙂公司本期经营承压。"
    raw["context_units"][1]["quote"] = "需求下降解释承压原因。"
    out, _ = invoke([raw], source=source, before=before)
    assert out["redundancy_review"]["complete"]
    units = out["redundancy_review"]["decisions"][0]["decision"]["context_units"]
    assert units[1]["quote"]["start"] == len(raw["context_units"][0]["quote"])


@pytest.mark.parametrize("bare_quotes,contexts", [(True, "topic_expansion"),
                                                  (False, ["topic_expansion"])])
def test_actions_v1_does_not_gain_v21_transport_coercions(bare_quotes, contexts):
    before = base_report()
    evidence = [FIRST["text"], SECOND["text"]] if bare_quotes else [FIRST, SECOND]
    raw = {"error_id": "r1", "action": "withdraw", "reason": "正常原因展开。",
           "normal_context": contexts, "evidence": evidence}
    out = review_redundancy(SOURCE, before, Chat([raw]))
    assert_atomic_failure(out, before)
    assert out["redundancy_review"]["schema_version"] == "redundancy-review/1.0"
