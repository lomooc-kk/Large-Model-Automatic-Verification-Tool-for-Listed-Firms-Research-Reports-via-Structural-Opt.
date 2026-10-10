import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("prepare_active_suite", Path(__file__).resolve().parents[1] / "prepare_active_suite.py")
suite = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(suite)


@pytest.fixture
def source_roots(tmp_path):
    external, clfec = tmp_path / "external", tmp_path / "clfec"
    suite.write_json(external / "MANIFEST.json", {"upstream_pins": {"FinanceBench": {"revision": "fixture"}}})
    groups = []
    for split, numbers in (("dev", (1, 2, 3, 4)), ("test", (5, 6))):
        incoming, gold = [], []
        for number in numbers:
            group = f"financebench::report-{number}"
            groups.append({"group_id": group, "splits": [split], "n_samples": 2})
            for suffix, has_error in (("A", True), ("B", False)):
                sid = f"upstream-{number}{suffix}"
                incoming.append({"sample_id": sid, "instruction": "What is the revenue?", "claim": "11" if has_error else "10", "evidence_text": f"Report {number} revenue is 10"})
                gold.append({"sample_id": sid, "has_error": has_error, "label": int(has_error), "label_text": "inconsistent" if has_error else "consistent", "correct_statement": "10", "error_category": "MR" if has_error else "none", "group_id": group})
        suite.write_rows(external / f"data/financebench_claim_verification/inputs.{split}.jsonl", incoming)
        suite.write_rows(external / f"data/financebench_claim_verification/gold.{split}.jsonl", gold)
    suite.write_rows(external / "splits/groups.jsonl", groups)
    suite.write_json(clfec / "raw/SOURCE.json", {"source": "CLFEC"})
    suite.write_json(clfec / "raw/CLFEC.official.json", [])
    for index, category in enumerate(("mix", "fec_only", "lec_only", "no_error")):
        sid = f"cl-{index}"
        suite.write_rows(clfec / f"data/inputs.{category}.jsonl", [{"sample_id": sid, "input_text": f"Input {index}"}])
        suite.write_rows(clfec / f"data/gold.{category}.jsonl", [{"sample_id": sid, "source_id": f"paragraph-{index}", "corrected_text": f"Corrected {index}", "cors": [], "num_edits": 0 if category == "no_error" else 1, "error_type_counts": {}}])
    return external, clfec, tmp_path / "active"


def refresh_artifact_hash(output, relative_path):
    manifest_path = output / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        if item["path"] == relative_path:
            item["sha256"] = suite.digest(output / relative_path)
    suite.write_json(manifest_path, manifest)


def test_new_only_split_is_reproducible_and_retains_pairs(source_roots):
    external, clfec, output = source_roots
    manifest = suite.prepare_suite(external, clfec, output)
    assert manifest["final_test_certified_unseen"] is False
    before = (output / "MANIFEST.json").read_bytes()
    suite.prepare_suite(external, clfec, output)
    assert before == (output / "MANIFEST.json").read_bytes()
    assert suite.validate_suite(output)["rows"] == 16
    assignments = suite.load_rows(output / "assignments.jsonl")
    for row in assignments:
        assert row["source"] in {"FinanceBench", "CLFEC"}
        assert row["document_id"].startswith("sample-")
        if row["upstream_split"] == "test":
            assert row["role"] == "final_candidate_reserved"
            assert row["exposure"] == "unknown_not_certified_unseen"
    groups = {}
    for row in assignments:
        groups.setdefault(row["source_group"], set()).add(row["role"])
    assert all(len(roles) == 1 for roles in groups.values())
    assert suite.load_rows(output / "legacy_exclusions.jsonl")[0]["active_rows"] == 0


def test_source_hash_drift_blocks_validation(source_roots):
    external, clfec, output = source_roots
    suite.prepare_suite(external, clfec, output)
    with (clfec / "raw/SOURCE.json").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Source identity/hash drift"):
        suite.validate_suite(output)


def test_legacy_source_cannot_replace_new_source(source_roots):
    external, clfec, output = source_roots
    suite.write_json(external / "MANIFEST.json", {"upstream_pins": {"FinED": {}}})
    with pytest.raises(ValueError, match="legacy source rejected"):
        suite.prepare_suite(external, clfec, output)
    with pytest.raises(ValueError, match="legacy data is diagnostic-only"):
        suite.prompt_payload("fined", {"content": "old"})


def test_gold_field_in_prompt_is_rejected_even_with_updated_hash(source_roots):
    external, clfec, output = source_roots
    suite.prepare_suite(external, clfec, output)
    relative = "financebench_claim_verification/inputs.development.jsonl"
    rows = suite.load_rows(output / relative)
    rows[0]["has_error"] = True
    suite.write_rows(output / relative, rows)
    refresh_artifact_hash(output, relative)
    with pytest.raises(ValueError, match="Prompt field whitelist"):
        suite.validate_suite(output)


def test_gold_id_mismatch_is_rejected(source_roots):
    external, clfec, output = source_roots
    suite.prepare_suite(external, clfec, output)
    relative = "clfec/gold.development.jsonl"
    rows = suite.load_rows(output / relative)
    rows[0]["document_id"] = "wrong-id"
    suite.write_rows(output / relative, rows)
    refresh_artifact_hash(output, relative)
    with pytest.raises(ValueError, match="input/gold IDs do not match"):
        suite.validate_suite(output)


def test_group_overlap_is_rejected(source_roots):
    external, clfec, output = source_roots
    suite.prepare_suite(external, clfec, output)
    rows = suite.load_rows(output / "assignments.jsonl")
    dev = next(row for row in rows if row["role"] == "development")
    val = next(row for row in rows if row["role"] == "validation")
    val["source_group"] = dev["source_group"]
    suite.write_rows(output / "assignments.jsonl", rows)
    refresh_artifact_hash(output, "assignments.jsonl")
    with pytest.raises(ValueError, match="Source group crosses"):
        suite.validate_suite(output)


def test_clfec_duplicate_variants_stay_together(source_roots):
    external, clfec, output = source_roots
    for category in ("mix", "fec_only"):
        path = clfec / f"data/gold.{category}.jsonl"
        rows = suite.load_rows(path)
        rows[0]["corrected_text"] = "Shared corrected original"
        suite.write_rows(path, rows)
    suite.prepare_suite(external, clfec, output)
    selected = [row for row in suite.load_rows(output / "assignments.jsonl") if row["track"] == "clfec" and row["upstream_split"] in {"mix", "fec_only"}]
    assert len({row["source_group"] for row in selected}) == 1
    assert len({row["role"] for row in selected}) == 1


def test_claim_prompt_retains_question_without_leaking_identifier():
    row = {"document_id": "sample-id", "instruction": "Which period?", "claim": "10", "evidence_text": "Evidence", "sample_id": "0001A", "has_error": True}
    assert suite.prompt_payload("financebench_claim_verification", row) == {"instruction": "Which period?", "claim": "10", "evidence_text": "Evidence"}


def test_legacy_overlap_quarantines_entire_source_pair(source_roots, tmp_path):
    external, clfec, output = source_roots
    old = tmp_path / "inputs.old.jsonl"
    suite.write_rows(old, [{"content": "Report 1 revenue is 10"}])
    manifest = suite.prepare_suite(external, clfec, output, legacy_paths=[old])
    audit = manifest["legacy_overlap_audit"]
    assert audit["legacy_gold_read"] is False
    assert audit["legacy_unique_texts"] == 1
    assert audit["excluded_rows"] == 2
    assert audit["excluded_source_components"] == 1
    assert all(row["source_group"] != "financebench::report-1" for row in suite.load_rows(output / "assignments.jsonl"))
    assert suite.validate_suite(output)["rows"] == 14


def test_legacy_gold_is_rejected(source_roots, tmp_path):
    external, clfec, output = source_roots
    path = tmp_path / "gold.old.jsonl"
    suite.write_rows(path, [])
    with pytest.raises(ValueError, match="input files only, never gold"):
        suite.prepare_suite(external, clfec, output, legacy_paths=[path])


@pytest.mark.parametrize("replacement", ["../outside.jsonl", "/absolute/outside.jsonl"])
def test_artifact_path_traversal_rejected(source_roots, replacement):
    external, clfec, output = source_roots
    suite.prepare_suite(external, clfec, output)
    path = output / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = replacement
    suite.write_json(path, manifest)
    with pytest.raises(ValueError, match="outside whitelist"):
        suite.validate_suite(output)


def test_omitted_artifact_cannot_bypass_hash_checks(source_roots):
    external, clfec, output = source_roots
    suite.prepare_suite(external, clfec, output)
    path = output / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["artifacts"].pop()
    suite.write_json(path, manifest)
    with pytest.raises(ValueError, match="Artifact inventory missing"):
        suite.validate_suite(output)


def test_cross_role_equal_evidence_is_detected(source_roots):
    external, clfec, output = source_roots
    suite.prepare_suite(external, clfec, output)
    path = output / "financebench_claim_verification/inputs.development.jsonl"
    evidence = suite.load_rows(path)[0]["evidence_text"]
    relative = "financebench_claim_verification/inputs.validation.jsonl"
    rows = suite.load_rows(output / relative)
    rows[0]["evidence_text"] = evidence
    suite.write_rows(output / relative, rows)
    refresh_artifact_hash(output, relative)
    with pytest.raises(ValueError, match="Normalized content crosses"):
        suite.validate_suite(output)
