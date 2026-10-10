"""Source-only normalization of adjacent citations within one error candidate."""
from copy import deepcopy


def normalize_candidate_spans(candidate: dict, content: str) -> dict:
    """Keep type/status/identity; merge only valid quotes with no missing words.

    Original citations remain available for audit. Never merge separate errors,
    bridge non-whitespace gaps, repair bad offsets, or consult reference answers.
    """
    result = deepcopy(candidate)
    spans = candidate.get("spans", [])
    if not spans or any(
        not isinstance(s, dict) or type(s.get("start")) is not int
        or type(s.get("end")) is not int
        or not 0 <= s["start"] < s["end"] <= len(content)
        or s.get("text") != content[s["start"]:s["end"]]
        for s in spans
    ):
        return result
    groups = []
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        start, end = span["start"], span["end"]
        if groups and (start <= groups[-1][1] or content[groups[-1][1]:start].isspace()):
            groups[-1][1] = max(groups[-1][1], end)
        else:
            groups.append([start, end])
    if len(groups) < len(spans):
        result["spans_before_normalization"] = deepcopy(spans)
        result["spans"] = [{"start": a, "end": b, "text": content[a:b]} for a, b in groups]
        result["span_normalization"] = "source_adjacent_v1"
    return result
