"""Read-only dataset admission and isolated CLFEC scoring. No model/network calls.

Admission certifies the requested protocol and local structure, not factual
correctness, model quality, or absence of pretraining contamination.
"""
from __future__ import annotations

import argparse
from collections import Counter
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ANSWER_FIELDS = frozenset({"errors", "gold", "gold_output", "answer", "gold_answer",
    "has_error", "label", "label_text", "corrected_text", "correct_statement",
    "cors", "candidate_word", "error_word", "error_type", "error_location",
    "modified_value", "original_value", "error_description", "num_edits"})
ID_FIELDS = frozenset({"sample_id", "doc_id", "document_id", "source_id", "group_id", "split"})
CLFEC_GROUPS = ("mix", "fec_only", "lec_only", "no_error")
# Separate native taxonomy; these are not aliases for FinED's 15 labels.
CLFEC_TYPE_MAP = {"Fact_Error": "factual", "Word_Error": "lexical",
                  "Grammar_Error": "grammatical", "Punc_Error": "punctuation"}
EXTERNAL_REQUIRED = {
    "finverbench_detection": ("instruction", "context"),
    "finverbench_correction": ("instruction", "corrupted_text"),
    "financebench_qa": ("instruction", "evidence_text"),
    "financebench_claim_verification": ("claim", "evidence_text"),
    "financebench_correction": ("instruction", "wrong_statement"),
}


class DatasetReadinessError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__("Dataset blocked: " + "; ".join(i["code"] for i in report["blocking_issues"]))


def _resolve(path: Path) -> Path:
    return path if path.exists() else Path(str(path) + ".gz")


def _rows(path: Path) -> list[dict]:
    path = _resolve(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        values = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(v, dict) for v in values):
        raise ValueError(f"Rows must be objects: {path}")
    return values


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _id(row: dict) -> str | None:
    return row.get("sample_id", row.get("doc_id", row.get("document_id")))


def _index(rows: list[dict], label: str) -> dict[str, dict]:
    values = {}
    for row in rows:
        key = _id(row)
        if not isinstance(key, str) or not key or key in values:
            raise ValueError(f"{label}: missing/duplicate identifier {key!r}")
        values[key] = row
    return values


def _answer_paths(value: Any, prefix: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if key in ANSWER_FIELDS:
                found.append(path)
            found.extend(_answer_paths(child, path))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(_answer_paths(child, f"{prefix}[{i}]"))
    return found


def _fields(task: str, selected=None) -> tuple[str, ...]:
    if task == "fined":
        required = ("content",)
    elif task == "clfec":
        required = ("input_text",)
    elif task.startswith("external:") and task[9:] in EXTERNAL_REQUIRED:
        required = EXTERNAL_REQUIRED[task[9:]]
    else:
        raise ValueError(f"Unsupported task: {task}")
    fields = tuple(selected) if selected is not None else required
    if not fields or not set(required).issubset(fields):
        raise ValueError(f"Prompt requires {required}; got {fields}")
    if any(key in ANSWER_FIELDS | ID_FIELDS for key in fields):
        raise ValueError("Identifiers, split metadata and answers cannot enter prompts")
    allowed = set(required) | {"instruction", "language", "context_full_page", "evidence_page_num"}
    if not set(fields).issubset(allowed):
        raise ValueError(f"Unsupported prompt fields: {set(fields) - allowed}")
    return fields


def build_inference_payload(task: str, row: dict, prompt_fields=None) -> dict:
    """Pure input projection. Never accepts gold or exposes IDs/partition metadata."""
    if _answer_paths(row):
        raise ValueError("Input contains answer fields")
    fields = _fields(task, prompt_fields)
    payload = {key: row[key] for key in fields}
    for key in _fields(task):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"Required text field is empty: {key}")
    return payload


def _issue(report, code, detail, *, warning=False):
    report["warnings" if warning else "blocking_issues"].append({"code": code, "detail": detail})


def _source_hash(report, path: Path):
    path = _resolve(path)
    report["source_files"].append({"path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


def _check_manifest(report, root: Path, path: Path):
    if not path.exists():
        _issue(report, "manifest_absent", str(path), warning=True)
        return
    manifest = _json(path)
    entries = manifest if isinstance(manifest, list) else manifest.get("artifacts", [])
    checked = 0
    for entry in entries:
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes dataset root")
        if target.exists():
            content = target.read_bytes()
        else:
            content = gzip.decompress(Path(str(target) + ".gz").read_bytes())
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            _issue(report, "source_hash_mismatch", entry["path"])
        checked += 1
    report["counts"]["manifest_files_checked"] = checked
    _source_hash(report, path)


def _check_split_manifest(report, config, inputs):
    """Explicit external role assignments. No assertion that a public set is blind."""
    path = config.get("split_manifest")
    if not path:
        if config["role"] == "evaluation":
            _issue(report, "evaluation_assignment_required",
                   "Provide a source-group split manifest; category names or filenames do not certify holdout status")
        return
    path = Path(path)
    data = _json(path)
    assignments = data.get("assignments", [])
    assigned, groups = {}, {}
    for item in assignments:
        key, role, group = item.get("document_id"), item.get("role"), item.get("source_group")
        if not key or key in assigned or role not in {"development", "evaluation", "regression"} or not group:
            raise ValueError("Invalid split assignment")
        assigned[key] = item
        if group in groups and groups[group] != role:
            _issue(report, "source_group_crosses_roles", group)
        groups[group] = role
    for row in inputs:
        item = assigned.get(_id(row), {})
        text = row.get("content", row.get("input_text"))
        if text is None:
            text = json.dumps(build_inference_payload(config["task"], row), ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if item.get("role") != config["role"] or item.get("input_sha256") != digest:
            _issue(report, "split_assignment_mismatch", str(_id(row)))
        if config["role"] == "evaluation" and item.get("exposure") != "unseen":
            _issue(report, "evaluation_exposure_not_unseen", str(_id(row)))
    report["protocol"]["split_manifest_is_user_attestation"] = True
    _source_hash(report, path)


def _span_ok(span: dict, text: str) -> bool:
    return (isinstance(span, dict) and type(span.get("start")) is int and
        type(span.get("end")) is int and 0 <= span["start"] < span["end"] <= len(text)
        and span.get("text") == text[span["start"]:span["end"]])


def _check_fined(report, config, root):
    archive = (root / "inputs/inputs.research442.jsonl").exists()
    split = config.get("split")
    ip = root / "inputs/inputs.research442.jsonl" if archive else root / f"inputs.{split}.jsonl"
    gp = root / "inputs/gold.research442.jsonl" if archive else root / f"gold.{split}.jsonl"
    if config.get("scope") not in {"all_hints", "candidate", "confirmed"}:
        _issue(report, "scope_required", "FinED needs all_hints, candidate or confirmed; never combine them")
    report["protocol"].update(scorer="fined_strict", scope=config.get("scope"),
        matching="typed_all_spans_contained_maximum_one_to_one",
        denominator="All planned documents; excluded gold remains separately counted",
        confirmation="confirmed requires deterministic proof; model assertion alone is insufficient")
    if archive:
        report["protocol"]["known_exposure"] = "research442 historical inference and diagnosis"
        if config["role"] != "regression":
            _issue(report, "known_exposed_442_requires_regression", "442 has been run and inspected; it is not an independent holdout")
        _check_manifest(report, root, root / "manifest.json")
    inputs, golds = _rows(ip), _rows(gp)
    im = _index(inputs, "inputs")
    count = Counter()
    for gold in golds:
        text = im.get(_id(gold), {}).get("content", "")
        if not isinstance(gold.get("errors"), list):
            raise ValueError("FinED gold requires errors[]; external QA/labels cannot be scored as FinED")
        for error in gold["errors"]:
            if type(error.get("scorable", True)) is not bool:
                raise ValueError("scorable must be boolean")
            if not error.get("scorable", True):
                count["excluded_gold_errors"] += 1
                continue
            count["scorable_gold_errors"] += 1
            spans = error.get("spans", [])
            if not isinstance(error.get("type"), str) or not spans or not all(_span_ok(s, text) for s in spans):
                _issue(report, "invalid_fined_gold_span_or_type", str(_id(gold)))
    report["counts"].update(count)
    return inputs, golds, [ip, gp]


def _check_clfec(report, config, root):
    groups = config.get("groups") or list(CLFEC_GROUPS)
    if config.get("split"):
        _issue(report, "clfec_category_is_not_split", "Use groups for mix/fec_only/lec_only/no_error; assign dev/evaluation separately")
    if not set(groups).issubset(CLFEC_GROUPS) or len(set(groups)) != len(groups):
        raise ValueError("Invalid CLFEC groups")
    inputs, golds, paths = [], [], []
    for group in groups:
        ip, gp = root / f"data/inputs.{group}.jsonl", root / f"data/gold.{group}.jsonl"
        inputs.extend(_rows(ip)); golds.extend(_rows(gp)); paths.extend((ip, gp))
    im = _index(inputs, "inputs")
    types, counts = Counter(), Counter()
    source_path, raw_path = root / "raw/SOURCE.json", root / "raw/CLFEC.official.json"
    source, raw = _json(source_path), _json(raw_path)
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != source["file_sha256"]:
        _issue(report, "source_hash_mismatch", str(raw_path))
    raw_by_id = {r["id"]: r for r in raw}
    if len({g.get("source_id") for g in golds}) != len(golds):
        _issue(report, "duplicate_clfec_source", "Each paragraph must map to one distinct source record")
    paths.extend((source_path, raw_path))
    for g in golds:
        text = im.get(_id(g), {}).get("input_text", "")
        if not isinstance(g.get("cors"), list) or not isinstance(g.get("corrected_text"), str):
            raise ValueError("CLFEC gold requires cors and corrected_text")
        if g.get("num_edits") != len(g["cors"]) or g.get("split") not in groups:
            _issue(report, "clfec_annotation_count_or_group_mismatch", str(_id(g)))
        src = raw_by_id.get(g.get("source_id"), {})
        if not (src.get("domain") == "Finance" and src.get("input_text") == text and
                src.get("corrected_text") == g["corrected_text"] and src.get("type") == g.get("split")):
            _issue(report, "clfec_source_mismatch", str(_id(g)))
        counts[g.get("split", "unknown")] += 1
        counts["annotated_edits"] += len(g["cors"])
        spans = []
        for edit in g["cors"]:
            start, end = edit.get("start"), edit.get("end")
            valid = type(start) is int and type(end) is int and 0 <= start <= end <= len(text)
            if not valid or text[start:end] != edit.get("error_word") or not isinstance(edit.get("candidate_word"), str):
                _issue(report, "invalid_clfec_edit_span", str(_id(g)))
                continue
            spans.append((start, end))
            kind = edit.get("error_type")
            if kind not in CLFEC_TYPE_MAP:
                _issue(report, "unknown_clfec_type", str(kind))
            types[str(kind)] += 1
        if any(a[1] > b[0] for a, b in zip(sorted(spans), sorted(spans)[1:])):
            _issue(report, "overlapping_clfec_edits", str(_id(g)))
        if len(spans) == len(g["cors"]):
            corrected = text
            for edit in sorted(g["cors"], key=lambda e: e["start"], reverse=True):
                corrected = corrected[:edit["start"]] + edit["candidate_word"] + corrected[edit["end"]:]
            if corrected != g["corrected_text"]:
                _issue(report, "clfec_reconstruction_mismatch", str(_id(g)))
        if g.get("split") == "no_error" and (g["cors"] or text != g["corrected_text"]):
            _issue(report, "clfec_clean_label_mismatch", str(_id(g)))
    report["counts"].update(counts)
    report["counts"]["native_error_types"] = dict(types)
    report["protocol"].update(scorer="clfec_edits", taxonomy=CLFEC_TYPE_MAP,
        category_groups=groups, category_groups_are_train_test=False,
        evidence_documents_available=False,
        denominator="All selected paragraphs; missing predictions fail and remain counted",
        edit_matching="Canonical SequenceMatcher edits for both gold and predictions; native annotation counts reported separately")
    _issue(report, "clfec_factual_evidence_absent", "Fact_Error labels have no supplied evidence documents; use paragraph correction diagnostics", warning=True)
    return inputs, golds, paths


def _check_external(report, config, root):
    task, split = config["task"][9:], config.get("split")
    if task not in EXTERNAL_REQUIRED or split not in {"dev", "test"}:
        raise ValueError("External task needs a supported dedicated task and dev/test split")
    ip, gp = root / f"data/{task}/inputs.{split}.jsonl", root / f"data/{task}/gold.{split}.jsonl"
    inputs, golds = _rows(ip), _rows(gp)
    _check_manifest(report, root, root / "MANIFEST.json")
    dev, test = _rows(root / "splits/dev.jsonl"), _rows(root / "splits/test.jsonl")
    overlap = {r["group_id"] for r in dev} & {r["group_id"] for r in test}
    if overlap:
        _issue(report, "source_group_crosses_splits", str(len(overlap)))
    selected_ids = {r["sample_id"] for r in (dev if split == "dev" else test)}
    if not {_id(i) for i in inputs}.issubset(selected_ids):
        _issue(report, "split_index_mismatch", "Input IDs are not contained in the selected source split")
    usable = [g for g in golds if not g.get("ambiguous_visible_text", False)]
    for gold in golds:
        if task in {"finverbench_detection", "financebench_claim_verification"}:
            label = gold.get("has_error")
            if type(label) is not bool or gold.get("label") != int(label) or gold.get("label_text") != ("inconsistent" if label else "consistent"):
                _issue(report, "external_label_contract_mismatch", str(_id(gold)))
        elif task == "financebench_qa" and not isinstance(gold.get("answer"), str):
            _issue(report, "external_qa_answer_missing", str(_id(gold)))
        elif task == "finverbench_correction" and not isinstance(gold.get("corrected_text"), str):
            _issue(report, "external_correction_answer_missing", str(_id(gold)))
        elif task == "financebench_correction" and not isinstance(gold.get("correct_statement"), str):
            _issue(report, "external_correction_answer_missing", str(_id(gold)))
    report["counts"].update(excluded_ambiguous=len(golds) - len(usable),
        usable_labels=len(usable), usable_clean=sum(g.get("has_error") is False for g in usable))
    report["protocol"].update(scorer="external:" + task, task_specific=True,
        native_source_split=split, denominator="All selected usable labels; ambiguous labels explicitly excluded and counted",
        not_fined_error_instances=True)
    if task == "finverbench_detection" and report["counts"]["usable_clean"] < 2:
        _issue(report, "fpr_not_estimable", "Fewer than two usable clean records; do not report a meaningful FPR", warning=True)
    if task == "financebench_claim_verification":
        counts = Counter((str(g["sample_id"])[-1], bool(g["has_error"])) for g in golds)
        report["counts"]["id_suffix_label_distribution"] = [
            {"suffix": k[0], "has_error": k[1], "count": v} for k, v in sorted(counts.items())]
        _issue(report, "raw_id_encodes_label", "Use build_inference_payload; raw A/B IDs encode class and must stay outside prompts", warning=True)
    return inputs, golds, [ip, gp]


def audit_dataset(config: dict) -> dict:
    """Audit local data and requested evaluation contract; returns ready/blocked."""
    config = dict(config)
    report = {"schema_version": "dataset-readiness/1.0", "status": "blocked",
        "task": config.get("task"), "role": config.get("role"), "blocking_issues": [],
        "warnings": [], "counts": {}, "protocol": {}, "source_files": [],
        "scope_limit": "Structural admission only; no claim of factual adjudication or independent model performance"}
    try:
        task, role = config["task"], config["role"]
        if role not in {"regression", "development", "evaluation"}:
            raise ValueError("role must be regression, development or evaluation")
        root = Path(config["root"]).resolve()
        expected = "fined_strict" if task == "fined" else "clfec_edits" if task == "clfec" else task
        if config.get("scorer") != expected:
            _issue(report, "task_scorer_mismatch", f"{task} requires scorer={expected}; got {config.get('scorer')}")
        if task != "fined" and config.get("scope") not in {None, "task"}:
            _issue(report, "fined_scope_on_external_task", "candidate/confirmed/all_hints are FinED scopes, not external task labels")
        fields = _fields(task, config.get("prompt_fields"))
        report["protocol"]["prompt_fields"] = list(fields)
        checker = _check_fined if task == "fined" else _check_clfec if task == "clfec" else _check_external
        inputs, golds, paths = checker(report, config, root)
        im, gm = _index(inputs, "inputs"), _index(golds, "gold")
        if not inputs:
            _issue(report, "empty_dataset", "No input documents")
        if set(im) != set(gm):
            _issue(report, "input_gold_id_mismatch", f"missing_gold={len(set(im)-set(gm))}, extra_gold={len(set(gm)-set(im))}")
        for row in inputs:
            try:
                build_inference_payload(task, row, fields)
            except (ValueError, KeyError, TypeError) as exc:
                _issue(report, "unsafe_or_incomplete_input", f"{_id(row)}: {exc}")
        report["counts"].update(documents=len(inputs), gold_documents=len(golds))
        for path in paths:
            _source_hash(report, path)
        _check_split_manifest(report, config, inputs)
        if role == "regression":
            _issue(report, "not_independent_holdout", "Regression results cannot be reported as a new holdout", warning=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        _issue(report, "invalid_dataset_or_contract", str(exc))
    report["status"] = "blocked" if report["blocking_issues"] else "ready"
    return report


def require_dataset_ready(config: dict) -> dict:
    report = audit_dataset(config)
    if report["status"] != "ready":
        raise DatasetReadinessError(report)
    return report


def _edits(src: str, dst: str) -> set[tuple]:
    matcher = difflib.SequenceMatcher(a=src, b=dst, autojunk=False)
    return {(a, b, dst[c:d]) for tag, a, b, c, d in matcher.get_opcodes() if tag != "equal"}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0}


def score_clfec(inputs: list[dict], golds: list[dict], predictions: list[dict]) -> dict:
    """Separate correction metric. Missing outputs never reduce the denominator.

    Both gold and prediction edits use the same deterministic segmentation, so a
    verbatim correct final paragraph has perfect edit credit even when upstream
    adjacent annotations used a different edit segmentation. No FinED type score.
    """
    im, gm, pm = _index(inputs, "inputs"), _index(golds, "gold"), _index(predictions, "predictions")
    if set(im) != set(gm) or set(pm) - set(im):
        raise ValueError("CLFEC input/gold alignment or unknown prediction IDs")
    dtp = dfp = dfn = etp = efp = efn = exact = clean = missing_clean = annotations = 0
    for sid, row in im.items():
        src, target = row["input_text"], gm[sid]["corrected_text"]
        gold_edits = _edits(src, target)
        annotations += len(gm[sid]["cors"])
        gold_error = bool(gold_edits)
        clean += not gold_error
        if sid not in pm:
            dfn += gold_error
            missing_clean += not gold_error
            efn += len(gold_edits)
            continue
        predicted = pm[sid].get("corrected_text")
        if not isinstance(predicted, str):
            raise ValueError("CLFEC prediction corrected_text must be a string")
        pred_edits = _edits(src, predicted)
        pred_error = bool(pred_edits)
        dtp += gold_error and pred_error
        dfp += not gold_error and pred_error
        dfn += gold_error and not pred_error
        etp += len(gold_edits & pred_edits)
        efp += len(pred_edits - gold_edits)
        efn += len(gold_edits - pred_edits)
        exact += predicted == target
    return {"scorer": "clfec_edits", "documents": len(im), "missing_predictions": sorted(set(im)-set(pm)),
        "missing_clean_predictions": missing_clean, "clean_denominator": clean,
        "false_positive_rate": dfp / clean if clean and not missing_clean else None,
        "observed_clean_false_positive_rate": dfp / (clean-missing_clean) if clean > missing_clean else None,
        "paragraph_detection": _prf(dtp, dfp, dfn), "canonical_edit_correction": _prf(etp, efp, efn),
        "native_annotated_edits": annotations, "exact_match_count": exact,
        "exact_match_rate": exact / len(im) if im else None,
        "type_accuracy": None, "note": "Canonical edit scoring; no implicit mapping to FinED taxonomy. Missing clean outputs remain recorded failures, not successful no-error decisions; full-clean FPR is unavailable until those outputs are complete."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--role", required=True, choices=("regression", "development", "evaluation"))
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--scope", choices=("all_hints", "candidate", "confirmed", "task"))
    parser.add_argument("--split")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--prompt-fields", nargs="+")
    parser.add_argument("--split-manifest")
    parser.add_argument("--out", required=True)
    args = vars(parser.parse_args(argv))
    out = Path(args.pop("out"))
    if out.resolve().is_relative_to(Path(args["root"]).resolve()):
        parser.error("--out must be outside the source dataset root; source data is read-only")
    report = audit_dataset(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
        "blocking_issues": report["blocking_issues"], "out": str(out)}, ensure_ascii=False))
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    sys.exit(main())
