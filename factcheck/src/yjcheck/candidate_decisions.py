"""Strict, source-independent validation of the optional detection contract.

All three verdicts are model opinions. Protocol validity neither binds a quote
to source text nor establishes business correctness. The caller retains the raw
response, checks completion/trace, anchors *every* decision, and routes results.
This module never filters decisions, changes reasons, reads files, or calls a
model. The legacy ``errors`` protocol is deliberately outside this module.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Literal, TypedDict, cast

from .text_taxonomy import FINED_ERROR_TYPES


SCHEMA_VERSION = "candidate-decisions/1.0"
VERDICTS = ("error_supported", "no_error", "insufficient_evidence")
MAX_REASON_CHARS = 400

Verdict = Literal["error_supported", "no_error", "insufficient_evidence"]


class QuoteSpan(TypedDict):
    text: str


class OffsetSpan(QuoteSpan):
    start: int
    end: int


class CandidateDecision(TypedDict):
    verdict: Verdict
    error_type: str
    spans: list[QuoteSpan | OffsetSpan]
    reason: str


class CandidateDecisionProtocolError(ValueError):
    """A whole-batch failure with a safe code and JSON-pointer location.

    No valid prefix is returned. The caller must keep its original response for
    audit and mark the stage incomplete, rather than treating failure as empty
    decisions. Messages intentionally do not echo model text or JSON key names.
    """

    def __init__(self, error_code: str, *, path: str = "",
                 decision_index: int | None = None) -> None:
        self.error_code = error_code
        self.path = path
        self.decision_index = decision_index
        super().__init__(error_code + (" at " + path if path else ""))


def parse_candidate_decisions(payload: object) -> list[CandidateDecision]:
    """Validate one decoded object and return all entries in original order.

    The returned value is a deep copy. Strings (including reason whitespace),
    span order, coordinates, and verdicts remain byte-for-byte/string-for-string
    unchanged; there is no alias conversion or inference from reason wording.
    An empty list is legal: normal documents need not enumerate every sentence.

    Quote existence, uniqueness, coordinate agreement with source, visibility,
    and issue identity require the caller's source-bound anchoring stage. For
    wire JSON use :func:`load_candidate_decisions` so duplicate JSON keys cannot
    have been silently overwritten by a generic decoder first.
    """
    if type(payload) is not dict or set(payload) != {"decisions"}:
        raise CandidateDecisionProtocolError("candidate_decisions_top_level_invalid")
    decisions = payload["decisions"]
    if type(decisions) is not list:
        raise CandidateDecisionProtocolError("candidate_decisions_not_list", path="/decisions")

    seen: set[str] = set()
    for index, decision in enumerate(decisions):
        base = f"/decisions/{index}"

        def fail(code: str, suffix: str = "") -> None:
            raise CandidateDecisionProtocolError(code, path=base + suffix,
                                                 decision_index=index)

        if type(decision) is not dict or set(decision) != {"verdict", "error_type", "spans", "reason"}:
            fail("candidate_decision_fields_invalid")
        if type(decision["verdict"]) is not str or decision["verdict"] not in VERDICTS:
            fail("candidate_decision_verdict_invalid", "/verdict")
        if type(decision["error_type"]) is not str or decision["error_type"] not in FINED_ERROR_TYPES:
            fail("candidate_decision_type_invalid", "/error_type")
        reason = decision["reason"]
        if type(reason) is not str or not reason.strip():
            fail("candidate_decision_reason_invalid", "/reason")
        if len(reason) > MAX_REASON_CHARS:
            fail("candidate_decision_reason_too_long", "/reason")
        spans = decision["spans"]
        if type(spans) is not list or not spans:
            fail("candidate_decision_spans_invalid", "/spans")
        for span_index, span in enumerate(spans):
            location = f"/spans/{span_index}"
            if type(span) is not dict or set(span) not in ({"text"}, {"start", "end", "text"}):
                fail("candidate_decision_span_fields_invalid", location)
            if type(span["text"]) is not str or not span["text"].strip():
                fail("candidate_decision_span_text_invalid", location + "/text")
            if "start" in span:
                if (type(span["start"]) is not int or type(span["end"]) is not int
                        or not 0 <= span["start"] < span["end"]):
                    fail("candidate_decision_span_coordinates_invalid", location)

        # Only exact whole-object equality is duplication. List order and string
        # whitespace are significant; JSON member ordering is not. In particular,
        # matching reasons or shared source sentences do not identify an issue.
        identity = json.dumps(decision, ensure_ascii=True, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
        if identity in seen:
            fail("candidate_decision_duplicate")
        seen.add(identity)

    return cast(list[CandidateDecision], deepcopy(decisions))


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CandidateDecisionProtocolError("candidate_decision_json_duplicate_key")
        result[key] = value
    return result


def _nonfinite_constant(_value: str) -> None:
    raise CandidateDecisionProtocolError("candidate_decision_json_nonfinite")


def load_candidate_decisions(text: str) -> list[CandidateDecision]:
    """Decode strict JSON without recovery, fence removal, or trace inference.

    Duplicate object keys and nonstandard NaN/Infinity constants fail before any
    projection. A caller must reject truncated transport *before* calling this
    function, even if the available prefix happens to be valid JSON.
    """
    if type(text) is not str:
        raise CandidateDecisionProtocolError("candidate_decision_json_text_invalid")
    try:
        payload = json.loads(text, object_pairs_hook=_unique_object,
                             parse_constant=_nonfinite_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CandidateDecisionProtocolError("candidate_decision_json_invalid") from exc
    return parse_candidate_decisions(payload)
