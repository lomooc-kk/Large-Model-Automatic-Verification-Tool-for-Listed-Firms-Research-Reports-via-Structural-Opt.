"""Admitted, source-only span-node ablation on historical research442 outputs.

Dataset admission runs in a child process: its gold structural checks cannot
pass gold objects to the normalization node. The parent loads gold for scoring
only after constructing all variants. No inference, network, or paid calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "factcheck/src")]
from evals.dataset_readiness import DatasetReadinessError, require_dataset_ready
from evals.fined_bench_eval import score_paper_detection
from evals.run_v2 import validate_predictions
from yjcheck.review_hints import all_review_hints
from yjcheck import span_normalization


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _admission_child(config):
    return require_dataset_ready(config)


def _admit(config):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--admission-only",
                             json.dumps(config)], capture_output=True, text=True,
                            encoding="utf-8", check=False)
    try:
        report = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError(f"Admission process failed: {result.stderr[:1000]}") from exc
    if result.returncode or report.get("status") != "ready":
        raise DatasetReadinessError(report)
    return report


def _normalize_report(report, content):
    result = deepcopy(report)
    result["errors"] = [span_normalization.normalize_candidate_spans(error, content)
                        for error in report["errors"]]
    for before, after in zip(report["errors"], result["errors"]):
        if any(before.get(key) != after.get(key) for key in ("error_type", "status", "id")):
            raise AssertionError("Span node changed identity, error type or confirmation status")
    return result


def _compact(score):
    return {key: score[key] for key in ("true_positive", "false_positive", "false_negative",
        "precision", "recall", "f1", "excluded_gold_errors", "missing_prediction_documents")}


def run_ablation(package, out):
    package, out = Path(package).resolve(), Path(out).resolve()
    if out.is_relative_to(package):
        raise ValueError("Output must be outside the read-only source package")
    admission = _admit({"task": "fined", "root": str(package), "role": "regression",
                        "scope": "all_hints", "scorer": "fined_strict"})
    # No parent input/gold/prediction reads before successful admission.
    inputs = {r["doc_id"]: r for r in _rows(package / "inputs/inputs.research442.jsonl")}
    before, after, changes, counts = {}, {}, [], {}
    for arm in ("model_direct", "hybrid"):
        before[arm], after[arm] = [], []
        seen, counter = set(), Counter()
        for path in sorted((package / "run/predictions" / arm).glob("*.json")):
            report = _read(path)
            did = report["document_id"]
            if did in seen or did not in inputs:
                raise ValueError("Duplicate/unknown prediction document")
            seen.add(did)
            content = inputs[did]["content"]
            base = all_review_hints(validate_predictions(report, content))
            variant = _normalize_report(base, content)
            counter["documents"] += 1
            counter["candidates"] += len(base["errors"])
            for index, (old, new) in enumerate(zip(base["errors"], variant["errors"])):
                if old.get("spans") != new.get("spans"):
                    counter["changed_candidates"] += 1
                    counter["merged_boundaries"] += len(old["spans"]) - len(new["spans"])
                    changes.append({"arm": arm, "document_id": did, "candidate_index": index,
                                    "prediction_file": path.relative_to(package).as_posix(),
                                    "before": old, "after": new})
            before[arm].append(base)
            after[arm].append(variant)
        if seen != set(inputs):
            raise ValueError(f"Historical predictions incomplete: {arm}, missing={len(set(inputs)-seen)}")
        counts[arm] = dict(counter)

    # Parent opens gold only here, after every source-only variant is complete.
    golds = _rows(package / "inputs/gold.research442.jsonl")
    saved_score = _read(package / "run/score.json")
    source_files = [Path(__file__), ROOT / "evals/dataset_readiness.py",
                    ROOT / "evals/fined_bench_eval.py", ROOT / "evals/run_v2.py",
                    ROOT / "factcheck/src/yjcheck/review_hints.py",
                    Path(span_normalization.__file__)]
    summary = {"protocol": "node-ablation/1.0", "dataset_admission": admission,
        "input_package": str(package), "new_model_calls": 0, "network_calls": 0,
        "historical_predictions": True, "scope": "all_review_hints_detection",
        "normalizer": "yjcheck.span_normalization.normalize_candidate_spans",
        "matching": "Unchanged strict typed full-span containment and maximum one-to-one matching",
        "gold_isolation": "Child admission returns only metadata; parent builds every variant before opening gold",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "limitations": ["Historical 2026-10-05 model candidates; no current-prompt quality claim.",
                        "Exposed regression corpus; not an independent holdout.",
                        "Localization representation repair does not establish factual correctness or automation safety."],
        "arms": {}}
    for arm in before:
        baseline, normalized = score_paper_detection(before[arm], golds), score_paper_detection(after[arm], golds)
        expected = saved_score["detectors"][arm]["all_review_hints_detection"]
        if _compact(baseline) != _compact(expected):
            raise ValueError(f"Strict historical baseline mismatch: {arm}")
        original_docs = {r["document_id"]: r for r in baseline["by_document"]}
        deltas = [{"document_id": r["document_id"], "before": original_docs[r["document_id"]], "after": r}
                  for r in normalized["by_document"] if any(r[k] != original_docs[r["document_id"]][k]
                    for k in ("true_positive", "false_positive", "false_negative"))]
        summary["arms"][arm] = {"baseline": _compact(baseline), "normalized": _compact(normalized),
            "baseline_matches_archive": True, "counts": counts[arm],
            "delta": {k: normalized[k] - baseline[k] for k in ("true_positive", "false_positive", "false_negative")},
            "changed_score_documents": deltas}
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "changed-candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in changes), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--admission-only", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.admission_only:
        try:
            report = _admission_child(json.loads(args.admission_only))
        except DatasetReadinessError as exc:
            print(json.dumps(exc.report, ensure_ascii=True))
            return 2
        print(json.dumps(report, ensure_ascii=True))
        return 0
    if not args.package or not args.out:
        parser.error("--package and --out are required")
    result = run_ablation(args.package, args.out)
    print(json.dumps({"new_model_calls": 0, "out": str(args.out), "arms": {
        arm: {"baseline": data["baseline"], "normalized": data["normalized"], "delta": data["delta"]}
        for arm, data in result["arms"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
