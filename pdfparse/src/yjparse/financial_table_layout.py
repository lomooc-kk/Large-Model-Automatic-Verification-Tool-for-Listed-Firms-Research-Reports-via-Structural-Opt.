"""Evidence-preserving candidate tables from native spans and repeated year headers.

This opt-in preparation utility does not replace the default parser or certify
financial truth. It separates adjacent financial statements before binding their
rows to years. Unassigned tokens, blank cells, unreadable source placeholders and
visibility decisions remain explicit. Unsupported layouts require review/OCR.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
from statistics import median

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

SCHEMA = "financial-table-layout/1.0"
_YEAR = re.compile(r"(?:19|20)\d{2}[AEae]?")
_NUMBER = re.compile(r"[+−-]?(?:\d[\d,，]*(?:\.\d+)?|\.\d+)(?:[%％倍xX])?|[（(][+−-]?\d[\d,]*(?:\.\d+)?[%％]?[)）]|#{2,}|[-—–/]+")
_TITLE = re.compile(r"资产负债表|损益表|利润表|现金流量表|主要财务比率|比率分析|财务指标|盈利预测|财务预测|估值指标")
_SOURCE = re.compile(r"^(?:资料来源|数据来源|来源)[：:]")


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _cx(item):
    return (item["bbox"][0] + item["bbox"][2]) / 2


def _cy(item):
    return (item["bbox"][1] + item["bbox"][3]) / 2


def _lines(tokens):
    """Cluster within a region; never join baselines across different tables."""
    groups = []
    for item in sorted(tokens, key=lambda t: (_cy(t), _cx(t))):
        tolerance = max(0.7, min(2.0, (item["bbox"][3] - item["bbox"][1]) * 0.24))
        if groups and abs(_cy(item) - median(_cy(t) for t in groups[-1])) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [sorted(row, key=_cx) for row in groups]


def _state(states):
    states = set(states)
    return next(iter(states)) if len(states) == 1 else "uncertain"


def _visibility_index(page, audit):
    if audit is None:
        return {}
    native_sha = hashlib.sha256(page.get_text("text").encode()).hexdigest()
    if (audit.get("page_number") != page.number + 1
            or audit.get("native_text_sha256") != native_sha
            or audit.get("page_rotation") != page.rotation):
        raise ValueError("visibility audit does not match this page")
    # Text alone cannot bind an audit: another PDF may paint identical text in
    # another color. Compare the exact rendered evidence before hiding tokens.
    scale = audit.get("render_scale")
    if audit.get("render_status") != "complete" or not isinstance(scale, (float, int)) or not 0 < scale <= 4:
        raise ValueError("visibility audit requires a completed bounded render")
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
    if hashlib.sha256(pix.samples_mv).hexdigest() != audit.get("render_rgb_sha256"):
        raise ValueError("visibility raster does not match this page")
    result = {}
    for span in audit["spans"]:
        for glyph in span["glyphs"]:
            key = (glyph["text"], *(round(v, 2) for v in glyph["origin"]))
            result.setdefault(key, []).append(glyph["status"])
    return {key: _state(states) for key, states in result.items()}


def _native_tokens(page, visibility):
    tokens, spans = [], []
    for bi, block in enumerate(page.get_text("rawdict")["blocks"]):
        for li, line in enumerate(block.get("lines", [])):
            for si, span in enumerate(line["spans"]):
                sid = f"b{bi}:l{li}:s{si}"
                chars = span["chars"]
                text = "".join(c["c"] for c in chars)
                spans.append({"span_id": sid, "text": text, "bbox": list(span["bbox"]),
                              "font": span["font"], "size": span["size"], "color": span["color"],
                              "direction": list(line["dir"])})
                # Split whitespace using real glyph boxes, never proportional
                # character widths. The original span remains in the sidecar.
                for match in re.finditer(r"\S+", text):
                    selected = chars[match.start():match.end()]
                    box = _union([c["bbox"] for c in selected])
                    status = _state(visibility.get((c["c"], *(round(v, 2) for v in c["origin"])), "uncertain")
                                    for c in selected)
                    tokens.append({"token_id": f"{sid}:{match.start()}:{match.end()}", "span_id": sid,
                                   "text": match.group(), "bbox": box, "visibility": status,
                                   "horizontal": abs(line["dir"][0] - 1) < 0.01 and abs(line["dir"][1]) < 0.01})
    return tokens, spans


def _headers(tokens, spans):
    titles = [s for s in spans if _TITLE.search(s["text"]) and len(s["text"].strip()) <= 45
              and abs(s["direction"][0] - 1) < 0.01 and abs(s["direction"][1]) < 0.01]
    headers = []
    for line in _lines([t for t in tokens if t["horizontal"] and _YEAR.fullmatch(t["text"])]):
        gaps = [_cx(b) - _cx(a) for a, b in zip(line, line[1:])]
        typical = median(gaps) if gaps else 0
        groups = []
        for token in line:
            crosses_title = bool(groups) and any(
                _cx(groups[-1][-1]) < title["bbox"][0] < _cx(token)
                and 0 < token["bbox"][1] - title["bbox"][1] <= 36 for title in titles)
            if (not groups or int(token["text"][:4]) <= int(groups[-1][-1]["text"][:4])
                    or _cx(token) - _cx(groups[-1][-1]) > typical * 2.5 or crosses_title):
                groups.append([token])
            else:
                groups[-1].append(token)
        for group in groups:
            if len(group) < 2:
                continue
            centers = [_cx(t) for t in group]
            step = median(b - a for a, b in zip(centers, centers[1:]))
            if step < 10 or any(not 0.65 * step <= b - a <= 1.35 * step for a, b in zip(centers, centers[1:])):
                continue
            nearby = [s for s in titles if 0 < group[0]["bbox"][1] - s["bbox"][1] <= 36
                      and s["bbox"][0] < centers[0] - step * 0.4
                      and centers[0] - s["bbox"][0] <= step * 3.5
                      and s["bbox"][2] < centers[-1] + step * 0.5]
            if not nearby:
                continue
            title = max(nearby, key=lambda s: (s["bbox"][1], s["bbox"][0]))
            headers.append({"tokens": group, "centers": centers, "step": step, "title": title,
                            "x0": title["bbox"][0], "x1": centers[-1] + step * 0.5,
                            "top": title["bbox"][1], "header_y": median(_cy(t) for t in group)})
    return headers


def extract_financial_tables(page, *, visibility_audit=None):
    """Return reviewable candidates; all coordinates are unrotated PDF points.

    The optional visibility sidecar must come from ``audit_page_visibility``.
    Hidden tokens retain raw text but have no display_text. Uncertain tokens
    are retained with a review flag. Nothing is silently promoted to gold.
    """
    visibility = _visibility_index(page, visibility_audit)
    tokens, spans = _native_tokens(page, visibility)
    headers = _headers(tokens, spans)
    used, tables = set(), []
    for header in sorted(headers, key=lambda h: (h["top"], h["x0"])):
        header_visibility = _state(t["visibility"] for t in header["tokens"])
        title_visibility = _state(t["visibility"] for t in tokens if t["span_id"] == header["title"]["span_id"])
        binding_verified = header_visibility == title_visibility == "visible"
        lower = [h["top"] for h in headers if h["top"] > header["top"] + 1
                 and max(h["x0"], header["x0"]) < min(h["x1"], header["x1"])]
        end_y = min(lower, default=page.cropbox.height)
        region = [t for t in tokens if t["horizontal"] and header["x0"] - 1 <= _cx(t) <= header["x1"] + 1
                  and header["header_y"] + 3 < _cy(t) < end_y]
        rows = []
        selected_ids = {t["token_id"] for t in header["tokens"]}
        selected_ids.update(t["token_id"] for t in tokens if t["span_id"] == header["title"]["span_id"])
        for line in _lines(region):
            if _SOURCE.match("".join(t["text"] for t in line)):
                break
            label_tokens = [t for t in line if _cx(t) < header["centers"][0] - header["step"] * 0.5]
            values = [t for t in line if t not in label_tokens]
            label = " ".join(t["text"] for t in label_tokens)
            numeric = [t for t in values if _NUMBER.fullmatch(t["text"])]
            # Prose/footer content is retained as unassigned, not made a row.
            if not label or len(label) > 45 or (values and len(numeric) != len(values)):
                continue
            cells = []
            for col, center in enumerate(header["centers"]):
                assigned = [t for t in values if min(range(len(header["centers"])),
                            key=lambda i: abs(_cx(t) - header["centers"][i])) == col]
                ambiguous = len(assigned) > 1 or any(abs(_cx(t) - center) > header["step"] * 0.5
                           or t["bbox"][2] - t["bbox"][0] > header["step"] * 1.25 for t in assigned)
                raw = " ".join(t["text"] for t in assigned) if assigned else None
                state = _state(t["visibility"] for t in assigned) if assigned else "not_applicable"
                status = ("ambiguous" if ambiguous else "blank" if not assigned else
                          "unreadable_source_placeholder" if re.fullmatch(r"#{2,}", raw) else
                          "source_dash" if re.fullmatch(r"[-—–/]+", raw) else "extracted")
                cells.append({"year": header["tokens"][col]["text"], "raw_year": header["tokens"][col]["text"],
                              "display_year": header["tokens"][col]["text"] if binding_verified else None,
                              "year_binding_status": "visible_native_header" if binding_verified else "pending_header_or_title_visibility",
                              "raw_text": raw,
                              "display_text": None if state == "invisible" or ambiguous else raw,
                              "status": status, "visibility": state,
                              "bbox": _union([t["bbox"] for t in assigned]) if assigned else None,
                              "token_ids": [t["token_id"] for t in assigned]})
            states = [t["visibility"] for t in line]
            row_state = _state(states)
            rows.append({"label": label, "label_bbox": _union([t["bbox"] for t in label_tokens]),
                         "label_visibility": _state(t["visibility"] for t in label_tokens),
                         "kind": "data" if numeric else "section_or_empty_row",
                         "bbox": _union([t["bbox"] for t in line]), "cells": cells,
                         "visibility": row_state, "display_eligible": row_state != "invisible",
                         "requires_review": not binding_verified or row_state != "visible" or any(c["status"] == "ambiguous" for c in cells),
                         "token_ids": [t["token_id"] for t in line]})
            selected_ids.update(t["token_id"] for t in line)
        if sum(r["kind"] == "data" for r in rows) < 2:
            continue
        title = header["title"]["text"].strip()
        unit = re.search(r"[（(]([^）)]+)[）)]", title)
        bounds = _union([header["title"]["bbox"], *(t["bbox"] for t in header["tokens"]), *(r["bbox"] for r in rows)])
        tables.append({"table_id": f"p{page.number + 1}:t{len(tables) + 1}", "title": title,
                       "title_span_id": header["title"]["span_id"], "title_unit_text": unit.group(1) if unit else None,
                       "title_visibility": title_visibility, "header_visibility": header_visibility,
                       "header_binding_status": "visible_native_header" if binding_verified else "pending_header_or_title_visibility",
                       "unit_scope": "title_only_not_inferred_for_each_metric", "bbox": bounds,
                       "display_bbox": list(fitz.Rect(bounds) * page.rotation_matrix),
                       "years": [t["text"] for t in header["tokens"]], "header_tokens": header["tokens"],
                       "rows": rows, "review_required": True, "business_correctness_claimed": False})
        used.update(selected_ids)
    return {"schema_version": SCHEMA, "page_number": page.number + 1, "page_rotation": page.rotation,
            "coordinate_frame": "unrotated_PDF_points", "display_page_rect": list(page.rect),
            "native_text": page.get_text("text"), "raw_spans": spans, "raw_tokens": tokens,
            "tables": tables, "unassigned_token_ids": [t["token_id"] for t in tokens if t["token_id"] not in used],
            "visibility_audited": visibility_audit is not None,
            "cell_status_counts": dict(Counter(c["status"] for table in tables for row in table["rows"] for c in row["cells"])),
            "page_visual_completeness_certified": False, "competition_ready": False,
            "limitations": ["Candidate native-text financial tables require visual review; unsupported layouts remain unassigned.",
                            "Empty means no native token in that position, not proof of a visible blank or financial zero.",
                            "Unreadable hashes and source dashes are preserved, never computed or converted into zero.",
                            "Charts, image/outlined text, wrapped labels and ambiguous spans require separate reconciliation.",
                            "A title unit is not automatically the unit of ratios, per-share metrics or row-specific quantities."]}
