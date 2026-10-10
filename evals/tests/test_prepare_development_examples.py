import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evals import prepare_development_examples as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    suite = tmp_path / "suite"
    incoming, golds, assignments = [], [], []
    for number, kind in enumerate(("Word_Error", "Grammar_Error", "Punc_Error", None, "Word_Error", "Fact_Error")):
        source = f"source {number} X"
        corrected = source if kind is None else source[:-1] + "Y"
        sid = f"sample-{number}"
        incoming.append({"document_id": sid, "input_text": source})
        edits = [] if kind is None else [{"start": len(source)-1, "end": len(source), "error_word": "X", "candidate_word": "Y", "error_type": kind}]
        golds.append({"document_id": sid, "corrected_text": corrected, "cors": edits, "num_edits": len(edits), "error_type_counts": {} if kind is None else {kind: 1}})
        assignments.append({"document_id": sid, "track": "clfec", "source": "CLFEC", "role": "development",
                            "upstream_id": f"up-{number}", "source_group": f"clfec::{number}", "upstream_split": "lec_only"})
    artifacts = []
    for relative, records in (("clfec/inputs.development.jsonl", incoming), ("clfec/gold.development.jsonl", golds), ("assignments.jsonl", assignments)):
        write_rows(suite / relative, records)
        artifacts.append({"path": relative, "sha256": module.digest(suite / relative), "rows": len(records)})
    # Intentionally invalid protected files: opening these would fail parsing.
    for role in ("validation", "final_candidate_reserved"):
        for prefix in ("inputs", "gold"):
            (suite / f"clfec/{prefix}.{role}.jsonl").write_text("PROTECTED DO NOT READ", encoding="utf-8")
    source_path = tmp_path / "raw/SOURCE.json"
    source = {"dataset": "CLFEC", "upstream_repo": "https://github.com/jiu2021/CLFEC-Dataset", "upstream_commit": "a"*40,
              "file_url": "https://raw.githubusercontent.com/jiu2021/CLFEC-Dataset/" + "a"*40 + "/data/CLFEC.json",
              "file_sha256": "b"*64, "license": "MIT"}
    write(source_path, source)
    # The full raw file does not even exist: its admitted hash is inherited.
    manifest = {"schema_version": "active-suite/1.0", "source_allowlist": ["CLFEC", "FinanceBench"], "artifacts": artifacts,
                "tracks": {"clfec": {"development": {"rows": 6}}}, "source_files": [
                    {"source": "CLFEC", "path": str(source_path), "sha256": module.digest(source_path)},
                    {"source": "CLFEC", "path": str(tmp_path / "raw/CLFEC.official.json"), "sha256": "b"*64}]}
    write(suite / "MANIFEST.json", manifest)
    plan_path = tmp_path / "plan.json"
    plan = {"schema_version": "active-comparison/1.0", "track": "clfec", "role": "development", "admission": {"status": "passed"},
            "suite": str(suite), "suite_sha256": module.digest(suite / "MANIFEST.json"), "input_sha256": module.digest(suite / "clfec/inputs.development.jsonl"),
            "document_ids": ["sample-4"], "planned_documents": 1}
    write(plan_path, plan)
    return suite, plan_path


def refresh(fixture, relative):
    suite, plan_path = fixture
    manifest = json.loads((suite / "MANIFEST.json").read_text())
    for entry in manifest["artifacts"]:
        if entry["path"] == relative:
            entry["sha256"] = module.digest(suite / relative)
    write(suite / "MANIFEST.json", manifest)
    plan = json.loads(plan_path.read_text())
    plan["suite_sha256"] = module.digest(suite / "MANIFEST.json")
    plan["input_sha256"] = module.digest(suite / "clfec/inputs.development.jsonl")
    write(plan_path, plan)


def test_bank_is_development_only_covers_categories_and_excludes_run(fixture):
    artifact, sft = module.prepare(*fixture)
    assert artifact["selected_document_ids"] == ["sample-0", "sample-1", "sample-2", "sample-3"]
    assert artifact["weight_training"] is False
    assert artifact["stage"] == "training-data-ready"
    assert artifact["schema_audit"]["validation_content_read"] is False
    assert len(sft) == 6
    assert all(set(row) == {"input_text", "corrected_text"} for row in artifact["examples"])
    assert all(set(row) == {"messages"} for row in sft)
    assert all(set(json.loads(row["messages"][1]["content"])) == {"input_text"} for row in sft)


def test_protected_contents_are_never_opened(fixture):
    original = Path.open
    opened = []
    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        assert not any(term in path.name for term in ("validation", "reserved", "CLFEC.official"))
        return original(path, *args, **kwargs)
    with patch.object(Path, "open", guarded):
        module.prepare(*fixture)
    assert any("gold.development" in path for path in opened)


def test_drift_blocks_before_selecting_examples(fixture):
    suite, plan = fixture
    with (suite / "clfec/gold.development.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash drift"):
        module.prepare(suite, plan)


def test_gold_must_reconstruct_exactly_even_if_hashes_are_refreshed(fixture):
    suite, _ = fixture
    golds = module.rows(suite / "clfec/gold.development.jsonl")
    golds[0]["corrected_text"] = "fabricated answer"
    write_rows(suite / "clfec/gold.development.jsonl", golds)
    refresh(fixture, "clfec/gold.development.jsonl")
    with pytest.raises(ValueError, match="reconstruct"):
        module.prepare(*fixture)


def test_non_development_role_is_rejected(fixture):
    suite, _ = fixture
    assignments = module.rows(suite / "assignments.jsonl")
    assignments[0]["role"] = "validation"
    write_rows(suite / "assignments.jsonl", assignments)
    refresh(fixture, "assignments.jsonl")
    with pytest.raises(ValueError, match="non-development"):
        module.prepare(*fixture)


def test_same_source_group_as_excluded_run_cannot_be_demonstration(fixture):
    suite, _ = fixture
    assignments = module.rows(suite / "assignments.jsonl")
    assignments[0]["source_group"] = assignments[4]["source_group"]
    write_rows(suite / "assignments.jsonl", assignments)
    refresh(fixture, "assignments.jsonl")
    with pytest.raises(ValueError, match="no short non-overlapping.*Word_Error"):
        module.prepare(*fixture)


def test_cli_emits_bank_and_candidate_sft_without_training(fixture, tmp_path, capsys):
    suite, plan = fixture
    output, sft = tmp_path / "bank.json", tmp_path / "future-sft.jsonl"
    assert module.main(["--suite", str(suite), "--exclude-plan", str(plan), "--out", str(output), "--sft-out", str(sft)]) == 0
    artifact = json.loads(output.read_text())
    assert artifact["sft"]["sha256"] == module.digest(sft)
    assert len(module.rows(sft)) == 6
    assert json.loads(capsys.readouterr().out)["weight_training"] is False


def test_cli_cannot_overwrite_suite_inputs(fixture):
    suite, plan = fixture
    with pytest.raises(SystemExit):
        module.main(["--suite", str(suite), "--exclude-plan", str(plan), "--out", str(suite / "clfec/inputs.development.jsonl")])


def test_verify_accepts_generated_bank_without_following_sft_path(fixture, tmp_path, capsys):
    suite, plan = fixture
    bank, _ = module.prepare(suite, plan)
    bank["sft"].update(file=str(tmp_path / "DO-NOT-OPEN.validation.gold.jsonl"), sha256="c" * 64)
    path = tmp_path / "bank.json"
    write(path, bank)
    before = path.read_bytes()
    assert module.main(["--suite", str(suite), "--exclude-plan", str(plan), "--verify-bank", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["sha256"] == module.digest(path)
    assert result["example_count"] == 4
    assert result["sft_file_read"] is False
    assert path.read_bytes() == before
    assert not (tmp_path / "DO-NOT-OPEN.validation.gold.jsonl").exists()


@pytest.mark.parametrize("mutation", ["validation_role", "selected_ids", "examples", "source_hash"])
def test_verify_rejects_replaced_role_ids_examples_or_hash(fixture, tmp_path, capsys, mutation):
    suite, plan = fixture
    bank, _ = module.prepare(suite, plan)
    if mutation == "validation_role":
        bank["role"] = "validation"
    elif mutation == "selected_ids":
        bank["selected_document_ids"][0] = "validation-only-id"
    elif mutation == "examples":
        bank["examples"][0]["corrected_text"] = "FORGED GOLD MUST NOT BE PRINTED"
    else:
        bank["source_hashes"]["gold_sha256"] = "0" * 64
    path = tmp_path / "tampered-bank.json"
    write(path, bank)
    assert module.main(["--suite", str(suite), "--exclude-plan", str(plan), "--verify-bank", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["valid"] is False
    assert "FORGED GOLD" not in captured.err


def test_verify_does_not_trust_example_path_or_evaluate_its_contents(fixture, tmp_path):
    suite, plan = fixture
    bank, _ = module.prepare(suite, plan)
    bank["examples"] = str(tmp_path / "validation-gold.jsonl")
    path = tmp_path / "bank.json"
    write(path, bank)
    with pytest.raises(ValueError, match="examples"):
        module.verify_bank(path, suite, plan)
