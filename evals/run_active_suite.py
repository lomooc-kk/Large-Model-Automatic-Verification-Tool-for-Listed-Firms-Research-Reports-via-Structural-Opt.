"""Frozen new-data comparisons. Run never opens gold; score is a separate command.

The old FinED collection is excluded. Development can guide node changes;
validation is an iterative checkpoint. Final candidates are intentionally not a
CLI choice. Claim labels remain model judgments, not automatic business proof.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "factcheck/src"), str(ROOT / "pdfparse/src")]

from evals.prepare_active_suite import prompt_payload

CLAIM = "financebench_claim_verification"
ARMS = {"clfec": ("direct", "structured", "structured_fewshot"), CLAIM: ("direct", "structured", "pi_provided", "pi_openviking")}
DEFAULT_ARMS = {"clfec": ("direct", "structured"), CLAIM: ARMS[CLAIM]}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_suite():
    config = json.loads((ROOT / "evals/current_dataset.json").read_text(encoding="utf-8"))
    manifest = (ROOT / config["active_manifest"]).resolve()
    if (config.get("schema_version") != "current-dataset/1.0" or not manifest.is_relative_to(ROOT)
            or config.get("legacy_role") != "diagnostic_only" or config.get("reserved_default_run_allowed") is not False
            or sha(manifest) != config.get("active_manifest_sha256")):
        raise ValueError("current_dataset_policy_or_hash_invalid")
    return manifest.parent


def gate(suite):
    # Metadata only crosses back from the process allowed to inspect gold.
    result = subprocess.run([sys.executable, str(ROOT / "evals/prepare_active_suite.py"),
                             "--validate-only", "--output", str(suite)], capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("active_suite_admission_failed")
    return json.loads(result.stdout)


def load_examples_bank(path, suite):
    """Admit a development bank in a separate process, then bind exact bytes.

    The inference process reads only the four verified demonstrations. Source
    gold verification belongs to the preparation subprocess, never this loader.
    """
    path = Path(path).resolve()
    checked = subprocess.run([sys.executable, str(ROOT / "evals/prepare_development_examples.py"),
                              "--suite", str(Path(suite).resolve()), "--verify-bank", str(path)],
                             capture_output=True, text=True, encoding="utf-8")
    if checked.returncode:
        raise ValueError("development_examples_bank_admission_failed")
    try:
        admitted = json.loads(checked.stdout)
        if (not isinstance(admitted, dict) or admitted.get("valid") is not True
                or admitted.get("role") != "development" or admitted.get("track") != "clfec"
                or admitted.get("example_count") != 4):
            raise ValueError
        raw = path.read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        if fingerprint != admitted.get("sha256"):
            raise ValueError("development_examples_bank_changed_after_admission")
        bank = json.loads(raw)
        examples, selected_ids = bank["examples"], bank["selected_document_ids"]
        if (bank.get("schema_version") != "development-examples/1.0" or bank.get("role") != "development"
                or bank.get("track") != "clfec" or bank.get("weight_training") is not False
                or not isinstance(examples, list) or len(examples) != admitted["example_count"]
                or not isinstance(selected_ids, list) or len(selected_ids) != len(examples)
                or any(not isinstance(sid, str) for sid in selected_ids) or len(set(selected_ids)) != len(selected_ids)
                or any(not isinstance(row, dict) or set(row) != {"input_text", "corrected_text"}
                       or any(not isinstance(value, str) or not value.strip() for value in row.values()) for row in examples)):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise ValueError("development_examples_bank_invalid_or_changed") from None
    metadata = {"path": str(path), "sha256": fingerprint, "example_count": len(examples),
                "selected_document_ids": selected_ids, "role": "development", "track": "clfec",
                "weight_training": False, "admission": "separate_process_source_hash_and_exact_examples_verified"}
    return examples, metadata


class ClaimSession:
    """Same official Pi loop, dedicated text evidence contract, no fake PDF pages."""
    def __init__(self, payload, client, store=None):
        self.payload, self.client, self.store = payload, client, store
        self.result, self.hits, self.reads, self.events = None, {}, {}, []
        self.search_count, self.rechecks = 0, {}

    def view(self):
        return {"status": "completed" if self.result.get("has_error") is not None else "needs_review",
                "result_ref": "bound-claim",
                "business_status": "needs_review", "label": self.result.get("label"),
                "claim": self.payload["claim"], "question_context": self.payload.get("instruction", ""),
                "reason": self.result.get("reason", "Evidence is required"),
                "next_action": ("stop" if self.result.get("has_error") is not None or not self.store or
                                len(self.rechecks) >= 2 else
                                "Read relevant distinct sources already found before searching again; recheck only new evidence."),
                "scope": "benchmark claim candidate only; never a confirmed business finding"}

    def __call__(self, name, args):
        from yjcheck.claim_verification import verify_claim
        from yjcheck.execution_scope import check_active
        check_active()
        if name == "detect_document":
            if args:
                raise ValueError("fixed_document_only")
            if self.result is None:
                payload = dict(self.payload)
                if self.store:
                    payload["evidence_text"] = ""
                self.result = verify_claim(payload, self.client, variant="structured")
                self.events.append({"tool": name})
            return self.view()
        if self.result is None:
            raise ValueError("detect_first")
        if not self.store:
            # Provided-evidence arm may stop with insufficient evidence; no oracle
            # lookup or hidden gold is available as a rescue mechanism.
            return {"status": "unavailable", "reason": "No additional evidence beyond the supplied excerpt", "hits": []}
        if name == "search_evidence":
            if set(args) - {"query", "limit"}:
                raise ValueError("invalid_search_arguments")
            if self.search_count >= 2:
                return {"status": "search_limit_reached", "hits": [],
                        "available_evidence_ids": list(self.hits),
                        "next_action": "Read relevant existing hits and recheck; stop if none are relevant."}
            found = self.store.search(args.get("query"), args.get("limit", 5))
            self.search_count += 1
            # A read expands a chunk to its whole bound source. Showing several
            # chunks of that same source wastes context and encourages re-reading.
            distinct = {}
            for hit in found:
                distinct.setdefault(hit["source_id"], hit)
            found = list(distinct.values())
            self.hits.update({hit["evidence_id"]: hit for hit in found})
            self.events.append({"tool": name, "source_ids": [h["source_id"] for h in found], "provider": "openviking"})
            return {"status": "ok", "hits": found, "provider": "openviking",
                    "next_action": "Read relevant distinct sources, then recheck. At most two searches are available."}
        if name == "read_evidence":
            eid = args.get("evidence_id")
            if set(args) != {"evidence_id"} or eid not in self.hits:
                raise ValueError("read_requires_prior_search")
            previous = next((r for r in self.reads.values() if r["source_id"] == self.hits[eid]["source_id"]), None)
            self.reads[eid] = previous or self.store.read(self.hits[eid])
            self.events.append({"tool": name, "source_id": self.reads[eid]["source_id"]})
            return {**self.reads[eid], "evidence_id": eid}
        if name == "recheck":
            ids = args.get("evidence_ids")
            if (set(args) != {"evidence_ids"} or not isinstance(ids, list) or not 1 <= len(ids) <= 5
                    or any(not isinstance(eid, str) or eid not in self.reads for eid in ids)):
                raise ValueError("recheck_requires_verified_reads")
            text = "\n\n".join(dict.fromkeys(self.reads[eid]["text"] for eid in ids))
            key = tuple(sorted({self.reads[eid]["source_id"] for eid in ids}))
            if key in self.rechecks:
                self.result = self.rechecks[key]
                return self.view()
            if len(self.rechecks) >= 2:
                return {**self.view(), "next_action": "stop", "reason": "Evidence recheck limit reached."}
            self.result = verify_claim({**self.payload, "evidence_text": text}, self.client, variant="structured")
            self.rechecks[key] = self.result
            self.events.append({"tool": name, "source_ids": [self.reads[eid]["source_id"] for eid in ids]})
            return self.view()
        raise ValueError("unknown_tool")


def run_one(track, arm, row, config, settings, store_path=None, *, examples=None):
    from yjcheck.model_runtime import BudgetedChatClient
    payload = prompt_payload(track, row)
    started = time.monotonic()
    raw_client = BudgetedChatClient(config, settings)
    traces = []
    def client(messages, **kwargs):
        try:
            response = raw_client(messages, **kwargs)
            if isinstance(response.get("trace"), dict):
                traces.append(response["trace"])
            return response
        except Exception as error:
            if isinstance(getattr(error, "trace", None), dict):
                traces.append(error.trace)
            raise
    try:
        if track == "clfec":
            from yjcheck.correction_review import correct_text
            if arm == "structured_fewshot":
                if not examples:
                    raise ValueError("verified_development_examples_required")
                result = correct_text(payload, client, variant="structured", examples=examples)
            else:
                result = correct_text(payload, client, variant=arm)
        elif arm in {"direct", "structured"}:
            from yjcheck.claim_verification import verify_claim
            result = verify_claim(payload, client, variant=arm)
        else:
            from evals.benchmark_retrieval import ExcerptStore
            from yjcheck.pi_bridge import run_pi, AgentLimits
            store = ExcerptStore(store_path) if arm == "pi_openviking" else None
            session = ClaimSession(payload, client, store)
            agent = run_pi({"documentId": "bound-claim", "question": "Check the bound financial claim. Detect first. Search at most twice; read relevant distinct sources already found, then recheck. Do not recheck the same source set twice. Stop after a decisive tool label or when available evidence is exhausted. Respect each tool's next_action. Do not infer a verdict from your narrative."},
                           session, config, settings=settings,
                           limits=AgentLimits(max_rounds=6, max_tool_calls=10, timeout_seconds=150,
                                              tool_timeout_seconds=60, max_total_tokens=80000,
                                              max_output_tokens=2048, max_cost_cny="0.8"))
            result = session.result or {"status": "failed", "label": "insufficient", "has_error": None}
            result = {**result, "agent": agent, "tool_events": session.events,
                      "retrieved_source_ids": sorted({h["source_id"] for h in session.hits.values()}),
                      "read_source_ids": sorted({h["source_id"] for h in session.reads.values()})}
            traces.extend(agent.get("model_traces", []))
            if agent.get("status") not in {"completed", "needs_review"}:
                result.update(status="failed", label="insufficient", has_error=None)
        result["model_traces"] = traces
        return {"document_id": row["document_id"], "arm": arm, "result": result,
                "seconds": round(time.monotonic() - started, 4)}
    except Exception as error:
        # Never expose transport bodies or config secrets in progress/errors.
        return {"document_id": row["document_id"], "arm": arm,
                "result": {"status": "failed", "error": type(error).__name__, "has_error": None, "model_traces": traces},
                "seconds": round(time.monotonic() - started, 4)}


def run(args):
    from yjcheck.model import ModelConfig
    from yjcheck.model_runtime import RuntimeSettings, BudgetLedger
    if args.role not in {"development", "validation"}:
        raise ValueError("reserved_data_cannot_be_run")
    suite, out = args.suite.resolve(), args.out.resolve()
    admission = gate(suite)
    if out.exists() or out.is_relative_to(suite):
        raise ValueError("new_output_directory_outside_suite_required")
    track, role = args.track, args.role
    selected = rows(suite / track / f"inputs.{role}.jsonl")
    selected.sort(key=lambda row: row["document_id"])
    if args.limit:
        if role != "development":
            raise ValueError("validation_uses_complete_frozen_split")
        selected = selected[:args.limit]
    arms = args.arms.split(",") if args.arms else list(DEFAULT_ARMS[track])
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS[track]):
        raise ValueError("invalid_arms")
    examples, examples_metadata = None, None
    if "structured_fewshot" in arms:
        if not getattr(args, "examples_bank", None):
            raise ValueError("structured_fewshot_requires_examples_bank")
        examples, examples_metadata = load_examples_bank(args.examples_bank, suite)
        if set(examples_metadata["selected_document_ids"]) & {row["document_id"] for row in selected}:
            raise ValueError("fewshot_examples_overlap_run_documents")
    if "pi_openviking" in arms and not args.store:
        raise ValueError("real_openviking_store_required")
    if args.store:
        from evals.benchmark_retrieval import ExcerptStore, digest
        store = ExcerptStore(args.store)
        expected = {digest(row["evidence_text"]) for row in rows(suite / CLAIM / f"inputs.{role}.jsonl")}
        if set(store.sources) != expected:
            raise ValueError("retrieval_pool_must_match_current_split_only")
    config = replace(ModelConfig.from_env(), timeout=120)
    settings = replace(RuntimeSettings.from_env(), max_output_tokens=getattr(args, "node_output_tokens", 16384), max_retries=0)
    ledger = BudgetLedger(settings.ledger, settings.budget_cny)
    before = ledger.summary()
    out.mkdir(parents=True)
    code_paths = ["evals/run_active_suite.py", "evals/benchmark_retrieval.py", "evals/prepare_active_suite.py", "evals/dataset_readiness.py",
                  "evals/prepare_development_examples.py",
                  "factcheck/src/yjcheck/claim_verification.py", "factcheck/src/yjcheck/correction_review.py",
                  "factcheck/src/yjcheck/pi_bridge.py", "factcheck/src/yjcheck/model_runtime.py", "factcheck/src/yjcheck/model.py",
                  "factcheck/src/yjcheck/execution_scope.py", "pdfparse/src/yjparse/openviking.py",
                  "agent_runtime/dist/runtime.js", "agent_runtime/dist/cli.js", "agent_runtime/dist/protocol.js",
                  "agent_runtime/dist/config.js", "agent_runtime/package-lock.json"]
    plan = {"schema_version": "active-comparison/1.0", "track": track, "role": role, "arms": arms,
            "suite": str(suite), "suite_sha256": sha(suite / "MANIFEST.json"), "admission": admission,
            "input_sha256": sha(suite / track / f"inputs.{role}.jsonl"),
            "document_ids": [r["document_id"] for r in selected], "planned_documents": len(selected),
            "gold_read_by_run": False, "old_dataset_used": False, "weight_training": False,
            "optimization": "source_bound_structured_node_prompt_with_development_fewshot" if examples_metadata else "source_bound_structured_node_prompt",
            "examples_bank": examples_metadata, "model": config.model,
            "thinking": config.thinking, "reasoning_effort": config.reasoning_effort,
            "settings": {"max_output_tokens": settings.max_output_tokens, "max_retries": 0,
                         "input_cny_per_mtok": settings.input_cny_per_mtok,
                         "output_cny_per_mtok": settings.output_cny_per_mtok,
                         "price_valid_until": settings.price_valid_until},
            "code_sha256": {p: sha(ROOT / p) for p in code_paths}, "budget_before": before,
            "started_at": datetime.now(timezone.utc).isoformat()}
    save(out / "plan.json", plan)
    plan_fingerprint = sha(out / "plan.json")
    done = 0
    own_calls = {}
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = [executor.submit(run_one, track, arm, row, config, settings, args.store,
                                       **({"examples": examples} if arm == "structured_fewshot" else {}))
                       for row in selected for arm in arms]
            for future in as_completed(pending):
                value = future.result()
                for trace in value["result"].get("model_traces", []):
                    if trace.get("call_id"):
                        own_calls[trace["call_id"]] = trace
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
                stream.flush()
                done += 1
                print(json.dumps({"done": done, "planned": len(pending), "arm": value["arm"],
                                  "status": value["result"].get("status")}), flush=True)
    after = ledger.summary()
    completion = {"completed_at": datetime.now(timezone.utc).isoformat(), "planned_runs": len(selected) * len(arms),
                  "attempted_runs": done, "budget_after": after,
                  "plan_sha256": plan_fingerprint,
                  "new_model_calls": len(own_calls),
                  "new_accounted_cny": str(sum((Decimal(str(t["cost_cny"])) for t in own_calls.values()), Decimal(0))),
                  "cost_scope": "unique_call_ids_from_this_run_including_pi_and_tools_not_provider_invoice",
                  "predictions_sha256": sha(out / "predictions.jsonl"),
                  "code_unchanged": all(sha(ROOT / p) == expected for p, expected in plan["code_sha256"].items()),
                  "plan_unchanged": sha(out / "plan.json") == plan_fingerprint}
    save(out / "completion.json", completion)
    print(json.dumps(completion), flush=True)


def score(args):
    from evals.dataset_readiness import score_clfec
    out = args.out.resolve()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    done = json.loads((out / "completion.json").read_text(encoding="utf-8"))
    if (done.get("plan_unchanged") is False or
            ("plan_sha256" in done and sha(out / "plan.json") != done["plan_sha256"])):
        raise ValueError("experiment_plan_changed")
    suite, track, role = Path(plan["suite"]), plan["track"], plan["role"]
    if role not in {"development", "validation"} or plan.get("old_dataset_used") is not False:
        raise ValueError("unsupported_experiment_scope")
    if sha(suite / "MANIFEST.json") != plan["suite_sha256"] or sha(out / "predictions.jsonl") != done["predictions_sha256"]:
        raise ValueError("frozen_experiment_changed")
    if not done["code_unchanged"]:
        raise ValueError("code_changed_during_inference")
    scorer_paths = ("evals/run_active_suite.py", "evals/dataset_readiness.py")
    scorer_hashes = {p: sha(ROOT / p) for p in scorer_paths}
    # Older development runs predate scorer freezing and remain explicitly
    # marked as retrospective when re-scored. New frozen experiments fail closed.
    scorer_frozen = "evals/dataset_readiness.py" in plan.get("code_sha256", {})
    if scorer_frozen and any(plan["code_sha256"].get(p) != h for p, h in scorer_hashes.items()):
        raise ValueError("scorer_changed_after_experiment_freeze")
    gate(suite)
    if sha(suite / track / f"inputs.{role}.jsonl") != plan["input_sha256"]:
        raise ValueError("experiment_inputs_changed")
    ids = set(plan["document_ids"])
    inputs = [r for r in rows(suite / track / f"inputs.{role}.jsonl") if r["document_id"] in ids]
    gold = [r for r in rows(suite / track / f"gold.{role}.jsonl") if r["document_id"] in ids]
    if (len(ids) != len(plan["document_ids"]) or len(inputs) != len(ids) or len(gold) != len(ids)
            or {r["document_id"] for r in inputs} != ids or {r["document_id"] for r in gold} != ids):
        raise ValueError("planned_input_gold_ids_do_not_align")
    incoming = rows(out / "predictions.jsonl")
    if any(r.get("arm") not in plan["arms"] for r in incoming):
        raise ValueError("unplanned_prediction_arm")
    report = {"track": track, "role": role, "planned_documents": len(ids), "old_dataset_used": False,
              "not_competition_total_accuracy": True, "new_model_calls": done["new_model_calls"],
              "new_accounted_cny": done["new_accounted_cny"], "scorer_sha256": scorer_hashes,
              "scorer_frozen_before_inference": scorer_frozen, "arms": {}}
    for arm in plan["arms"]:
        pred = [r for r in incoming if r["arm"] == arm]
        by_id = {r["document_id"]: r["result"] for r in pred}
        if len(by_id) != len(pred) or set(by_id) - ids:
            raise ValueError("duplicate_or_unknown_prediction")
        failed = sum(r.get("status") == "failed" or r.get("coverage", {}).get("execution_complete") is False for r in by_id.values())
        if track == "clfec":
            valid = [{"document_id": key, "corrected_text": r["corrected_text"]} for key, r in by_id.items()
                     if r.get("status") != "failed" and isinstance(r.get("corrected_text"), str)]
            result = score_clfec(inputs, gold, valid)
        else:
            tp = fp = fn = tn = abstentions = wrong_clean_abstentions = 0
            for g in gold:
                prediction = by_id.get(g["document_id"], {})
                incomplete = (prediction.get("status") == "failed" or
                              prediction.get("coverage", {}).get("execution_complete") is False or
                              ("agent" in prediction and prediction["agent"].get("status") not in {"completed", "needs_review"}))
                label = None if incomplete else prediction.get("has_error")
                truth = g["has_error"]
                if label is None:
                    abstentions += 1
                    fn += truth
                    wrong_clean_abstentions += not truth
                elif type(label) is not bool:
                    raise ValueError("claim_prediction_label_not_boolean")
                elif label:
                    tp += truth
                    fp += not truth
                else:
                    fn += truth
                    tn += not truth
            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0
            clean_count = sum(not g["has_error"] for g in gold)
            result = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "abstentions": abstentions,
                      "clean_abstentions": wrong_clean_abstentions, "precision": precision, "recall": recall,
                      "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
                      "accuracy_full_denominator": (tp + tn) / len(gold),
                      "decision_coverage": (len(gold) - abstentions) / len(gold),
                      "clean_denominator": clean_count,
                      "clean_decision_coverage": (clean_count - wrong_clean_abstentions) / clean_count if clean_count else None,
                      "false_positive_rate": fp / clean_count if clean_count else None}
            if arm == "pi_openviking":
                from evals.benchmark_retrieval import digest
                result["retrieved_expected_excerpt"] = sum(digest(r["evidence_text"]) in by_id.get(r["document_id"], {}).get("retrieved_source_ids", []) for r in inputs)
                result["read_expected_excerpt"] = sum(digest(r["evidence_text"]) in by_id.get(r["document_id"], {}).get("read_source_ids", []) for r in inputs)
                result["retrieval_scope"] = "current_split_provided_excerpt_pool_not_original_pdf_corpus"
        result.update(attempted=len(pred), execution_failures=failed,
                      total_seconds=sum(r["seconds"] for r in pred))
        report["arms"][arm] = result
    save(out / "score.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def prepare_retrieval(args):
    from evals.benchmark_retrieval import ExcerptStore
    if args.role not in {"development", "validation"}:
        raise ValueError("reserved_data_cannot_enter_retrieval")
    gate(args.suite)
    data = rows(args.suite / CLAIM / f"inputs.{args.role}.jsonl")
    namespace = "active-" + args.role + "-" + sha(args.suite / "MANIFEST.json")[:12]
    store = ExcerptStore.create(args.out, [r["evidence_text"] for r in data],
             ROOT / "data/local-openviking/models/bge-small-zh-v1.5/tokenizer.json", namespace)
    print(json.dumps({"status": "ready", "documents": len(store.sources), "chunks": len(store.bindings),
                      "source_kind": "benchmark_provided_excerpts", "original_pdf_verified": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "score", "prepare-retrieval"])
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--track", choices=list(ARMS), default=CLAIM)
    parser.add_argument("--role", choices=["development", "validation"], default="development")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--arms")
    parser.add_argument("--examples-bank", type=Path, help="Verified development-only bank, required only for structured_fewshot")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--node-output-tokens", type=int, choices=[4096, 8192, 16384], default=16384)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    args.suite = args.suite or current_suite()
    {"run": run, "score": score, "prepare-retrieval": prepare_retrieval}[args.command](args)


if __name__ == "__main__":
    main()
