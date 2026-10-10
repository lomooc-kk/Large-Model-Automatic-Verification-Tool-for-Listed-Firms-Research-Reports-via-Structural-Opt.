"""Prepare two task-specific new-data tracks without importing historical FinED.

This is data engineering, not model training or model evaluation. Reserved inputs
and answers are written to disk but never printed. Their unseen status is unknown.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "active-suite/1.0"
SEED = "active-suite-v1-20261009"
TRACK_FIELDS = {
    "financebench_claim_verification": ("instruction", "claim", "evidence_text"),
    "clfec": ("input_text",),
}
ROLES = ("development", "validation", "final_candidate_reserved")
ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT.parent / "output/node-audit-20261009/remote-audit"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_hash(text: str) -> str:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return hashlib.sha256(normalized.encode()).hexdigest()


def legacy_input_files() -> list[Path]:
    """Only old inputs are read for exclusion; never historical gold or scores."""
    return sorted(set(ROOT.glob("data/v2/dataset/inputs.*.jsonl")) |
                  set(ROOT.glob("data/v2/dataset/inputs.*.jsonl.gz")) |
                  set(ROOT.glob("artifacts/research442-rerun-20261005/inputs/inputs.*.jsonl")) |
                  set(ROOT.glob("artifacts/research442-rerun-20261005/inputs/inputs.*.jsonl.gz")))


def resolve_jsonl(path: Path) -> Path:
    if path.is_file():
        return path
    compressed = Path(str(path) + ".gz")
    if compressed.is_file():
        return compressed
    raise ValueError(f"Missing source file: {path}")


def load_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values), encoding="utf-8")


def opaque_id(track: str, original_id: str) -> str:
    token = hashlib.sha256(f"{SEED}|{track}|{original_id}".encode()).hexdigest()[:24]
    return "sample-" + token


def prompt_payload(track: str, row: dict) -> dict:
    """Only these strings may enter inference. IDs and provenance stay outside."""
    if track not in TRACK_FIELDS:
        raise ValueError("Unsupported active track; legacy data is diagnostic-only")
    result = {field: row[field] for field in TRACK_FIELDS[track]}
    if any(not isinstance(value, str) or not value.strip() for value in result.values()):
        raise ValueError("Empty or non-string prompt field")
    return result


def group_roles(groups: set[str]) -> dict[str, str]:
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{SEED}|{group}".encode()).hexdigest())
    # Keep at least one group in each side where possible; never split a group.
    n_validation = max(1, math.ceil(len(ordered) * .30)) if len(ordered) > 1 else 0
    validation = set(ordered[:n_validation])
    return {group: "validation" if group in validation else "development" for group in groups}


def joined(inputs: list[dict], gold: list[dict]) -> list[tuple[dict, dict]]:
    def index(rows):
        result = {}
        for row in rows:
            sid = row.get("sample_id")
            if not isinstance(sid, str) or not sid or sid in result:
                raise ValueError("Missing or duplicated upstream sample ID")
            result[sid] = row
        return result
    incoming, answers = index(inputs), index(gold)
    if incoming.keys() != answers.keys():
        raise ValueError("Input/gold upstream IDs do not match")
    return [(incoming[sid], answers[sid]) for sid in sorted(incoming)]


def prepare_suite(external_root: Path, clfec_root: Path, output: Path, legacy_paths: list[Path] | None = None) -> dict:
    external_root, clfec_root, output = external_root.resolve(), clfec_root.resolve(), output.resolve()
    # The source contracts are deliberately narrow. No generic FinED loader or
    # historical split is accepted as a replacement for either new-data source.
    manifest_path = external_root / "MANIFEST.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "FinanceBench" not in source_manifest.get("upstream_pins", {}):
        raise ValueError("FinanceBench source contract missing; legacy source rejected")
    source_marker = clfec_root / "raw/SOURCE.json"
    official = clfec_root / "raw/CLFEC.official.json"
    if not source_marker.is_file() or not official.is_file():
        raise ValueError("CLFEC source contract missing; legacy source rejected")
    sources: dict[str, dict] = {}

    def record_source(path, kind):
        if kind not in {"FinanceBench", "CLFEC"}:
            raise ValueError("Old or unapproved active source")
        path = path.resolve()
        sources[str(path)] = {"path": str(path), "source": kind, "sha256": digest(path)}
        return path

    for path, kind in ((manifest_path, "FinanceBench"), (source_marker, "CLFEC"), (official, "CLFEC")):
        record_source(path, kind)
    pairs: dict[str, list[dict]] = defaultdict(list)
    claim_track = "financebench_claim_verification"
    group_registry_path = record_source(external_root / "splits/groups.jsonl", "FinanceBench")
    group_registry = {row["group_id"]: row["splits"] for row in load_rows(group_registry_path)}
    for native_split in ("dev", "test"):
        base = external_root / "data" / claim_track
        input_path = record_source(resolve_jsonl(base / f"inputs.{native_split}.jsonl"), "FinanceBench")
        gold_path = record_source(resolve_jsonl(base / f"gold.{native_split}.jsonl"), "FinanceBench")
        for incoming, answer in joined(load_rows(input_path), load_rows(gold_path)):
            group = answer.get("group_id")
            if not isinstance(group, str) or not group.startswith("financebench::"):
                raise ValueError("Missing FinanceBench document source group")
            if group_registry.get(group) != [native_split]:
                raise ValueError("Source group crosses or contradicts upstream splits")
            if not isinstance(answer.get("has_error"), bool):
                raise ValueError("Claim gold requires boolean has_error")
            if answer.get("label") != int(answer["has_error"]):
                raise ValueError("Claim gold label polarity mismatch")
            pairs[claim_track].append({"input": incoming, "gold": answer, "group": group,
                                      "native_split": native_split, "source": "FinanceBench"})
    dev_groups = {row["group"] for row in pairs[claim_track] if row["native_split"] == "dev"}
    role_map = group_roles(dev_groups)
    for row in pairs[claim_track]:
        row["role"] = role_map[row["group"]] if row["native_split"] == "dev" else "final_candidate_reserved"
        row["exposure"] = "prior_development_audited" if row["native_split"] == "dev" else "unknown_not_certified_unseen"

    for category in ("mix", "fec_only", "lec_only", "no_error"):
        base = clfec_root / "data"
        input_path = record_source(resolve_jsonl(base / f"inputs.{category}.jsonl"), "CLFEC")
        gold_path = record_source(resolve_jsonl(base / f"gold.{category}.jsonl"), "CLFEC")
        for incoming, answer in joined(load_rows(input_path), load_rows(gold_path)):
            if not isinstance(answer.get("source_id"), str) or not answer["source_id"]:
                raise ValueError("Missing CLFEC source ID")
            if not isinstance(answer.get("corrected_text"), str) or not isinstance(answer.get("cors"), list):
                raise ValueError("CLFEC correction gold contract missing")
            pairs["clfec"].append({"input": incoming, "gold": answer, "group": "clfec::" + answer["source_id"],
                                   "native_split": category, "source": "CLFEC", "exposure": "full_scorer_sanity_audited_not_blind"})
    # Source IDs are paragraph IDs, not proven report IDs. Link identical input
    # OR corrected text as well, preventing duplicates/variants crossing roles.
    parents = {row["group"]: row["group"] for row in pairs["clfec"]}
    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    fingerprints = {}
    for row in pairs["clfec"]:
        for content in (row["input"]["input_text"], row["gold"]["corrected_text"]):
            fingerprint = normalized_hash(content)
            if fingerprint in fingerprints:
                a, b = root(row["group"]), root(fingerprints[fingerprint])
                parents[max(a, b)] = min(a, b)
            fingerprints[fingerprint] = row["group"]
    for row in pairs["clfec"]:
        row["group"] = root(row["group"])
    clfec_role_map = group_roles({row["group"] for row in pairs["clfec"]})
    for row in pairs["clfec"]:
        row["role"] = clfec_role_map[row["group"]]

    # Old inputs serve only as a denylist. Matching one member excludes its whole
    # source component, including paired correct/incorrect variants. Do not read
    # historical gold, add legacy samples, or silently rebrand them as new data.
    old_hashes, legacy_sources = set(), []
    for path in legacy_input_files() if legacy_paths is None else legacy_paths:
        path = path.resolve()
        if not path.name.startswith("inputs."):
            raise ValueError("Legacy dedup accepts input files only, never gold")
        records = load_rows(path)
        for record in records:
            for key in ("content", "input_text", "text"):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    old_hashes.add(normalized_hash(value))
        legacy_sources.append({"path": str(path), "sha256": digest(path), "rows": len(records), "role": "diagnostic_dedup_only"})
    quarantine, matched_groups = [], set()
    for track, rows in pairs.items():
        for row in rows:
            contents = list(prompt_payload(track, row["input"]).values())
            contents.append(row["gold"]["corrected_text" if track == "clfec" else "correct_statement"])
            if any(normalized_hash(content) in old_hashes for content in contents):
                matched_groups.add(row["group"])
    for track, rows in pairs.items():
        for row in rows:
            if row["group"] in matched_groups:
                quarantine.append({"document_id": opaque_id(track, row["input"]["sample_id"]),
                                   "track": track, "source_group": row["group"],
                                   "reason": "normalized_content_overlaps_legacy_input_source_component"})
        pairs[track] = [row for row in rows if row["group"] not in matched_groups]

    assignments, artifacts, summaries = [], [], {}
    for track, rows in sorted(pairs.items()):
        summaries[track] = {}
        seen = set()
        for role in ROLES:
            selected = [row for row in rows if row["role"] == role]
            incoming, gold = [], []
            for row in selected:
                original_id = row["input"]["sample_id"]
                sid = opaque_id(track, original_id)
                if sid in seen:
                    raise ValueError("Duplicate active sample ID")
                seen.add(sid)
                incoming.append({"document_id": sid, **prompt_payload(track, row["input"])})
                # Keep task-native supervision; never invent a 15-class mapping.
                if track == claim_track:
                    answer = {key: row["gold"][key] for key in ("has_error", "label", "label_text", "correct_statement", "error_category")}
                else:
                    answer = {key: row["gold"][key] for key in ("corrected_text", "cors", "num_edits", "error_type_counts")}
                gold.append({"document_id": sid, **answer})
                assignments.append({"document_id": sid, "track": track, "role": role, "source": row["source"],
                                    "source_group": row["group"], "upstream_id": original_id,
                                    "upstream_split": row["native_split"], "exposure": row["exposure"]})
            for prefix, records in (("inputs", incoming), ("gold", gold)):
                path = output / track / f"{prefix}.{role}.jsonl"
                write_rows(path, sorted(records, key=lambda item: item["document_id"]))
                artifacts.append({"path": path.relative_to(output).as_posix(), "rows": len(records), "sha256": digest(path)})
            summaries[track][role] = {"rows": len(selected), "source_groups": len({row["group"] for row in selected}),
                                     "clean": sum(not row["gold"]["has_error"] for row in selected) if track == claim_track else sum(row["gold"]["num_edits"] == 0 for row in selected)}
    assignment_path = output / "assignments.jsonl"
    write_rows(assignment_path, sorted(assignments, key=lambda row: row["document_id"]))
    artifacts.append({"path": "assignments.jsonl", "rows": len(assignments), "sha256": digest(assignment_path)})
    legacy_path = output / "legacy_exclusions.jsonl"
    write_rows(legacy_path, [{"source": "FinED", "scope": "All historical FinED and research442 inputs, gold and predictions",
                              "role": "diagnostic_only", "active_rows": 0,
                              "reason": "User requested old examples only for problem diagnosis, never next-round validation",
                              "retention": "Keep original files and history; no deletion or rebranding as new data"}])
    artifacts.append({"path": "legacy_exclusions.jsonl", "rows": 1, "sha256": digest(legacy_path)})
    quarantine_path = output / "quarantine.legacy_overlap.jsonl"
    write_rows(quarantine_path, sorted(quarantine, key=lambda row: row["document_id"]))
    artifacts.append({"path": quarantine_path.name, "rows": len(quarantine), "sha256": digest(quarantine_path)})
    legacy_audit = {"comparison": "NFKC_plus_remove_all_whitespace_exact_full_text_hash",
                    "legacy_input_files": legacy_sources, "legacy_unique_texts": len(old_hashes),
                    "legacy_gold_read": False, "excluded_source_components": len(matched_groups),
                    "excluded_rows": len(quarantine), "active_legacy_rows": 0,
                    "limitation": "Exact normalized full-text dedup only; near-duplicates and unprovided upstream report identities are not certified"}
    manifest = {"schema_version": SCHEMA, "seed": SEED, "status": "ready_for_development_not_final_benchmark",
                "source_allowlist": ["CLFEC", "FinanceBench"], "source_files": sorted(sources.values(), key=lambda row: row["path"]),
                "artifacts": artifacts, "tracks": summaries, "prompt_fields": TRACK_FIELDS,
                "legacy_overlap_audit": legacy_audit,
                "role_policy": {"development": "Node/prompt optimization and training-data preparation permitted",
                                "validation": "Iterative checkpoint comparison; observed results make this development-exposed",
                                "final_candidate_reserved": "Frozen upstream test candidate; do not use for optimization or training; prior exposure unknown"},
                "boundaries": ["No FinED or research442 rows in active tracks",
                               "No model weight training was performed by this builder",
                               "FinanceBench claims measure given-evidence consistency, not full-document retrieval; synthetic errors are upward-only +8 percent",
                               "FinanceBench QA/correction sharing a source_group must inherit this assignment and never become a cross-role training shortcut",
                               "CLFEC was fully audited for scorer sanity; neither split is an independently unseen final test",
                               "CLFEC source IDs identify paragraphs; original-report linkage is unavailable, so report-level independence is not certified",
                               "Task-native scores remain separate; no automatic conversion into 15-class FinED or competition score",
                               "Raw source IDs and provenance are excluded from inference payloads"],
                "competition_15_class_ready": False, "pdf_evidence_retrieval_ready": False,
                "final_test_certified_unseen": False}
    write_json(output / "MANIFEST.json", manifest)
    validate_suite(output)
    return manifest


def validate_suite(output: Path) -> dict:
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or set(manifest.get("source_allowlist", [])) != {"CLFEC", "FinanceBench"}:
        raise ValueError("Old or unapproved active source")
    for source in manifest["source_files"]:
        if source["source"] not in manifest["source_allowlist"] or digest(Path(source["path"])) != source["sha256"]:
            raise ValueError("Source identity/hash drift")
    old_hashes = set()
    for source in manifest.get("legacy_overlap_audit", {}).get("legacy_input_files", []):
        path = Path(source["path"])
        if not path.name.startswith("inputs.") or source.get("role") != "diagnostic_dedup_only":
            raise ValueError("Legacy dedup source must be inputs only")
        if digest(path) != source["sha256"]:
            raise ValueError("Legacy input hash drift")
        for row in load_rows(path):
            for key in ("content", "input_text", "text"):
                if isinstance(row.get(key), str) and row[key].strip():
                    old_hashes.add(normalized_hash(row[key]))
    expected_artifacts = {f"{track}/{prefix}.{role}.jsonl" for track in TRACK_FIELDS for prefix in ("inputs", "gold") for role in ROLES}
    expected_artifacts.update({"assignments.jsonl", "legacy_exclusions.jsonl", "quarantine.legacy_overlap.jsonl"})
    paths = [item["path"] for item in manifest["artifacts"]]
    if set(paths) != expected_artifacts or len(paths) != len(expected_artifacts):
        raise ValueError("Artifact inventory missing, duplicated, or outside whitelist")
    for artifact in manifest["artifacts"]:
        path = (output / artifact["path"]).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != artifact["sha256"]:
            raise ValueError("Artifact path/hash drift")
        if len(load_rows(path)) != artifact["rows"]:
            raise ValueError("Artifact row count drift")
    assignments = load_rows(output / "assignments.jsonl")
    assignment_by_id, group_roles_seen = {}, defaultdict(set)
    for item in assignments:
        if item["source"] not in manifest["source_allowlist"] or item["track"] not in TRACK_FIELDS:
            raise ValueError("Old or unapproved active source")
        if item["document_id"] in assignment_by_id:
            raise ValueError("Duplicated active assignment ID")
        assignment_by_id[item["document_id"]] = item
        group_roles_seen[item["source_group"]].add(item["role"])
        if item["role"] == "final_candidate_reserved" and (item["track"] != "financebench_claim_verification" or item["upstream_split"] != "test" or item["exposure"] != "unknown_not_certified_unseen"):
            raise ValueError("Reserved data has an invalid provenance/exposure claim")
    if any(len(roles) != 1 for roles in group_roles_seen.values()):
        raise ValueError("Source group crosses active roles")
    found, content_roles = set(), defaultdict(set)
    for track in TRACK_FIELDS:
        for role in ROLES:
            inputs = load_rows(output / track / f"inputs.{role}.jsonl")
            gold = load_rows(output / track / f"gold.{role}.jsonl")
            input_ids = {row["document_id"] for row in inputs}
            gold_ids = {row["document_id"] for row in gold}
            answers = {row["document_id"]: row for row in gold}
            if len(input_ids) != len(inputs) or len(gold_ids) != len(gold) or input_ids != gold_ids:
                raise ValueError("Active input/gold IDs do not match")
            if found & input_ids:
                raise ValueError("Active ID appears in more than one role")
            found.update(input_ids)
            for row in inputs:
                if set(row) != {"document_id", *TRACK_FIELDS[track]}:
                    raise ValueError("Prompt field whitelist violated")
                prompt_payload(track, row)
                item = assignment_by_id[row["document_id"]]
                if item["track"] != track or item["role"] != role:
                    raise ValueError("Assignment role does not match payload")
                payload = prompt_payload(track, row)
                answer = answers[row["document_id"]]
                contents = list(payload.values()) + [answer["corrected_text" if track == "clfec" else "correct_statement"]]
                if any(normalized_hash(content) in old_hashes for content in contents):
                    raise ValueError("Active content overlaps legacy input")
                dedup_contents = [row["input_text"], answer["corrected_text"]] if track == "clfec" else [row["evidence_text"]]
                for content in dedup_contents:
                    content_roles[(track, normalized_hash(content))].add(role)
    if found != assignment_by_id.keys():
        raise ValueError("Assignments and active rows do not match")
    if any(len(roles) > 1 for roles in content_roles.values()):
        raise ValueError("Normalized content crosses active roles")
    return {"status": "passed", "rows": len(found), "source_groups": len(group_roles_seen), "final_test_certified_unseen": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, default=AUDIT_ROOT / "external/evals/external_benchmarks")
    parser.add_argument("--clfec-root", type=Path, default=AUDIT_ROOT / "clfec/evals/external_benchmarks/clfec_finance")
    parser.add_argument("--output", type=Path, default=ROOT / "data/active_suite_v1")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_suite(args.output), ensure_ascii=False, indent=2))
    else:
        manifest = prepare_suite(args.external_root, args.clfec_root, args.output)
        print(json.dumps({"status": manifest["status"], "tracks": manifest["tracks"], "final_test_certified_unseen": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
