"""Bounded JSONL bridge to the real Pi runtime, sharing the existing ledger."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import uuid

from .model_runtime import BudgetedChatClient, BudgetExceeded, MICROS

ROOT = Path(__file__).resolve().parents[3]
MAX_LINE = 2_000_000


@dataclass(frozen=True)
class AgentLimits:
    max_rounds: int = 6
    max_tool_calls: int = 12
    timeout_seconds: float = 180
    tool_timeout_seconds: float = 60
    max_total_tokens: int = 48000
    max_output_tokens: int = 4096
    max_cost_cny: str = "2"

    def validate(self):
        for value, lower, upper in ((self.max_rounds, 1, 32), (self.max_tool_calls, 1, 128),
                                    (self.max_total_tokens, 128, 4000000), (self.max_output_tokens, 1, 131072)):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("agent_integer_limit_out_of_range")
        for value, lower, upper in ((self.timeout_seconds, 0.1, 600), (self.tool_timeout_seconds, 0.05, 300)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not lower <= value <= upper:
                raise ValueError("agent_timeout_out_of_range")
        try:
            cost = Decimal(str(self.max_cost_cny))
        except InvalidOperation:
            raise ValueError("invalid_agent_cost_limit") from None
        if not cost.is_finite() or not 0 <= cost <= 1000:
            raise ValueError("invalid_agent_cost_limit")


def runtime_command():
    node = os.environ.get("YJCHECK_NODE") or shutil.which("node")
    entry = ROOT / "agent_runtime/dist/cli.js"
    if not node or not entry.is_file():
        raise ValueError("pi_runtime_unavailable: run npm ci and npm run build in agent_runtime")
    return [node, str(entry)]


def run_pi(task, tool_handler, model_config, *, settings=None, limits=None, command=None):
    """Pi plans; Python tools own verdicts and authorize every paid request.

    An unknown/failed/interrupted request retains its maximum reservation.
    Credentials travel only over stdin, never argv or returned traces.
    """
    limits = limits or AgentLimits()
    limits.validate()
    budget = BudgetedChatClient(model_config, settings)
    config = budget.settings
    # Orchestration emits short tool calls. It must not inherit the detector's
    # 65,536-token output setting, which exceeds Pi's default total allowance.
    output_tokens = min(config.max_output_tokens, limits.max_output_tokens)
    input_tokens = min(config.context_tokens - output_tokens, limits.max_total_tokens - output_tokens)
    if input_tokens < 128:
        raise ValueError("agent_context_budget_too_small")
    thinking = ("off" if getattr(model_config, "thinking", "") == "disabled" else
                getattr(model_config, "reasoning_effort", "") or
                ("low" if getattr(model_config, "thinking", "") == "enabled" else "off"))
    request_id = uuid.uuid4().hex
    start = {
        "type": "start", "protocolVersion": 1, "requestId": request_id, "task": task,
        "model": {"baseUrl": model_config.base_url, "apiKey": model_config.api_key,
                  "modelId": model_config.model, "contextWindow": config.context_tokens,
                  "maxOutputTokens": output_tokens,
                  "thinking": thinking},
        "limits": {"maxRounds": limits.max_rounds, "maxToolCalls": limits.max_tool_calls,
                   "timeoutMs": int(limits.timeout_seconds * 1000),
                   "toolTimeoutMs": int(limits.tool_timeout_seconds * 1000),
                   "maxInputTokens": input_tokens,
                   "maxTotalTokens": limits.max_total_tokens, "maxCostCny": float(limits.max_cost_cny)},
        "pricing": {"inputCnyPerMtok": float(budget.input_rate),
                    "outputCnyPerMtok": float(budget.output_rate)},
    }
    proc = subprocess.Popen(command or runtime_command(), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, encoding="utf-8", bufsize=1,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    messages = queue.Queue()

    def reader():
        try:
            while True:
                line = proc.stdout.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE:
                    messages.put(ValueError("pi_message_too_large"))
                    break
                messages.put(json.loads(line))
        except Exception:
            messages.put(ValueError("pi_invalid_jsonl"))
        finally:
            messages.put(None)

    threading.Thread(target=reader, daemon=True).start()

    def send(message):
        proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    reservations, audit, seen_tools = {}, [], set()
    used_maximum = Decimal(0)
    rounds = 0
    final = None
    reason = "pi_terminated_without_final"
    deadline = time.monotonic() + limits.timeout_seconds
    try:
        send(start)
        while time.monotonic() < deadline:
            try:
                event = messages.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                reason = "pi_timeout"
                break
            if event is None:
                break
            if isinstance(event, Exception) or not isinstance(event, dict):
                reason = "pi_invalid_protocol"
                break
            kind = event.get("type")
            if kind == "model_request":
                eid = event.get("id")
                inp, out = event.get("estimatedInputTokens"), event.get("maxOutputTokens")
                valid = (isinstance(eid, str) and eid and eid not in reservations
                         and type(inp) is int and type(out) is int and inp > 0 and out > 0
                         and out <= output_tokens and inp <= input_tokens and inp + out <= config.context_tokens
                         and rounds < limits.max_rounds)
                maximum = (Decimal(inp) * budget.input_rate + Decimal(out) * budget.output_rate) / MICROS if valid else Decimal(0)
                if not valid or used_maximum + maximum > Decimal(limits.max_cost_cny):
                    send({"type": "model_permission", "id": eid, "allowed": False})
                    reason = "pi_request_limit"
                    continue
                try:
                    budget._check_price_validity()
                    trace = {"status": "reserved", "purpose": "pi.orchestrate", "request_id": request_id,
                             "model": model_config.model, "round": rounds,
                             "input_token_estimate": inp, "max_output_tokens": out,
                             "maximum_cny": str(maximum), "price_source": config.price_source,
                             "price_date": config.price_date, "price_valid_until": config.price_valid_until}
                    cid = budget.ledger.reserve(maximum, trace)
                except (BudgetExceeded, ValueError):
                    send({"type": "model_permission", "id": eid, "allowed": False})
                    reason = "pi_budget_or_price_blocked"
                    continue
                rounds += 1
                used_maximum += maximum
                reservations[eid] = {"call_id": cid, "maximum": maximum, "trace": trace, "settled": False}
                send({"type": "model_permission", "id": eid, "allowed": True, "reservationId": cid})
            elif kind == "model_usage":
                reservation = reservations.get(event.get("id"))
                if not reservation or reservation["settled"] or event.get("reservationId") != reservation["call_id"]:
                    reason = "pi_invalid_usage_reference"
                    break
                inp, out = event.get("inputTokens"), event.get("outputTokens")
                known = event.get("usageKnown") is True and all(type(v) is int and v >= 0 for v in (inp, out))
                cost = ((Decimal(inp) * budget.input_rate + Decimal(out) * budget.output_rate) / MICROS
                        if known else reservation["maximum"])
                exceeded = cost > reservation["maximum"]
                status = "ok" if event.get("status") == "ok" and known and not exceeded else "error"
                trace = {**reservation["trace"], "call_id": reservation["call_id"], "status": status,
                         "usage_source": "provider" if known else "maximum_reservation_no_provider_usage",
                         "cost_cny": str(cost), "input_tokens": inp if known else None,
                         "output_tokens": out if known else None}
                if exceeded:
                    trace["reservation_exceeded"] = True
                budget.ledger.settle(reservation["call_id"], cost, trace)
                reservation["settled"] = True
                used_maximum += cost - reservation["maximum"]
                audit.append(trace)
                if exceeded:
                    reason = "provider_usage_exceeds_reserved_maximum"
                    break
            elif kind == "tool_call":
                eid, name, args = event.get("id"), event.get("name"), event.get("args")
                if (not isinstance(eid, str) or eid in seen_tools or len(seen_tools) >= limits.max_tool_calls
                        or name not in {"detect_document", "search_evidence", "read_evidence", "recheck"}
                        or not isinstance(args, dict)):
                    reason = "pi_invalid_tool_call"
                    break
                seen_tools.add(eid)
                tool_deadline = min(deadline, time.monotonic() + limits.tool_timeout_seconds)
                cancelled = threading.Event()
                tool_results = queue.Queue(maxsize=1)

                def invoke_tool(tool_name=name, tool_args=args, results=tool_results,
                                token=cancelled, until=tool_deadline):
                    from .execution_scope import execution_scope
                    try:
                        with execution_scope(token, until):
                            value = tool_handler(tool_name, tool_args)
                        results.put((True, value))
                    except Exception as exc:
                        results.put((False, type(exc).__name__))

                threading.Thread(target=invoke_tool, daemon=True).start()
                try:
                    ok, value = tool_results.get(timeout=max(0.001, tool_deadline - time.monotonic()))
                except queue.Empty:
                    cancelled.set()
                    reason = "pi_tool_timeout"
                    break
                send({"type": "tool_result", "id": eid, "result" if ok else "error": value})
            elif kind == "final":
                if (event.get("requestId") != request_id or event.get("protocolVersion") != 1
                        or event.get("status") not in {"completed", "needs_review", "stopped", "failed"}):
                    reason = "pi_invalid_final_reference"
                    break
                final = event
                break
            elif kind != "event":
                reason = "pi_unknown_protocol_event"
                break
    except (OSError, ValueError, BrokenPipeError):
        reason = "pi_bridge_transport_failed"
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        for reservation in reservations.values():
            if not reservation["settled"]:
                trace = {**reservation["trace"], "call_id": reservation["call_id"], "status": "error",
                         "error_code": "pi_interrupted_unknown_usage", "cost_cny": str(reservation["maximum"])}
                budget.ledger.settle(reservation["call_id"], reservation["maximum"], trace)
                audit.append(trace)
        for stream in (proc.stdin, proc.stdout):
            if stream:
                stream.close()
    if final is None:
        final = {"type": "final", "protocolVersion": 1, "requestId": request_id,
                 "status": "failed", "stopReason": reason}
    if any(t.get("status") != "ok" for t in audit) and final.get("status") == "completed":
        final = {**final, "status": "stopped", "stopReason": "pi_model_usage_or_execution_incomplete"}
    # Keep only bounded runtime metadata; verdicts come from the Python session.
    return {"engine": "pi", "protocol_version": 1, "request_id": request_id,
            "status": final.get("status"), "stop_reason": final.get("stopReason"),
            "effective_thinking": thinking,
            "max_output_tokens": output_tokens,
            "model_calls": rounds, "tool_calls": len(seen_tools), "model_traces": audit,
            "budget": budget.ledger.summary()}
