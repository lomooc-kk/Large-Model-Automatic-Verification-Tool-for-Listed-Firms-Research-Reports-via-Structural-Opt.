"""Merge K independent detection runs into candidates with intervals.

The model is sampled K times on the same document. Each candidate is identified
by its error type plus its exact source spans; a candidate that appears in
several runs is counted as a vote for that span. The merged candidate keeps the
first run's text and reason and gains a ``confidence`` block with a Wilson
interval.

Exact span identity is intentional. Slightly different quotes from the same
sentence are *not* merged, so the vote count can only be understated, never
inflated. Callers that need fuzzy merging should normalise spans before calling
this module and keep the original spans for audit.

Nothing here reads gold answers or changes a verdict. ``status`` stays whatever
the detector produced; the interval is additional evidence for the reviewer and
for ``uncertainty.decide``.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

from .uncertainty import Interval, wilson_interval


def _span_key(span: object) -> tuple:
    if not isinstance(span, dict):
        return ("invalid", repr(span))
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int:
        text = span.get("text")
        return ("text", text if isinstance(text, str) else repr(text))
    return ("offset", start, end)


def candidate_key(candidate: dict) -> tuple:
    """Stable identity: error type plus the exact spans, order-insensitive."""
    error_type = candidate.get("error_type", candidate.get("type"))
    spans = candidate.get("spans") or []
    return (str(error_type), tuple(sorted(_span_key(s) for s in spans)))


def merge_runs(runs: Iterable[Iterable[dict]]) -> list[dict]:
    """Count candidate occurrences across independent runs.

    ``runs`` is one iterable of candidates per run. The return value keeps the
    first-seen candidate for each identity, in first-seen order, with::

        confidence = {votes, trials, p, lo, hi, method: "wilson"}

    Duplicate candidates inside a single run count once: a model repeating the
    same finding in one response is not independent evidence.
    """
    run_list = [list(run) for run in runs]
    trials = len(run_list)
    if trials == 0:
        return []

    order: list[tuple] = []
    first: dict[tuple, dict] = {}
    votes: dict[tuple, int] = {}
    for run in run_list:
        seen_this_run: set[tuple] = set()
        for candidate in run:
            key = candidate_key(candidate)
            if key in seen_this_run:
                continue
            seen_this_run.add(key)
            if key not in first:
                first[key] = deepcopy(candidate)
                order.append(key)
                votes[key] = 0
            votes[key] += 1

    merged = []
    for key in order:
        candidate = first[key]
        interval: Interval = wilson_interval(votes[key], trials)
        candidate["confidence"] = interval.as_dict()
        merged.append(candidate)
    return merged


def merge_reports(reports: Iterable[dict]) -> dict:
    """Merge ``errors`` from K text-review reports into one consensus report.

    Only the fields this module understands are carried over. The first report
    supplies document identity and coverage; later reports contribute votes.
    Coverage is deliberately *not* recomputed: the caller must still check that
    every run completed the same range, otherwise votes are not comparable.
    """
    report_list = [deepcopy(r) for r in reports]
    if not report_list:
        raise ValueError("merge_reports requires at least one report")
    base = report_list[0]
    base["errors"] = merge_runs(r.get("errors") or [] for r in report_list)
    base["consensus"] = {
        "runs": len(report_list),
        "method": "wilson",
        "note": "interval describes run-to-run agreement only; it is not a "
                "probability that the finding is business-correct",
    }
    return base


def summary(merged: Iterable[dict]) -> dict:
    """Count merged candidates by decision band, for dashboards and logs."""
    from .uncertainty import decide

    counts = {"auto_confirm": 0, "needs_review": 0, "drop": 0}
    for candidate in merged:
        conf = candidate.get("confidence") or {}
        interval = Interval(
            votes=int(conf.get("votes", 0)),
            trials=int(conf.get("trials", 0)),
            p=float(conf.get("p", 0.0)),
            lo=float(conf.get("lo", 0.0)),
            hi=float(conf.get("hi", 0.0)),
        )
        counts[decide(interval)] += 1
    return counts
