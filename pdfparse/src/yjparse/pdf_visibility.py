"""Conservative visibility sidecar for native PDF text, never a text remover.

Three-way decisions describe rendered evidence, not the correctness of a report.
All native text and original trace/glyph records remain available. Missing native
glyphs, outlined/image text and OCR completeness are outside this audit's claim.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover - older supported PyMuPDF package name
    import fitz


SCHEMA = "pdf-text-visibility/1.0"
_MAX_PIXELS = 16_000_000


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _rgb(trace):
    color = trace.get("color", ())
    if trace.get("colorspace") == 1 and len(color) == 1:
        color = color * 3
    if len(color) != 3 or not all(isinstance(v, (float, int)) and 0 <= v <= 1 for v in color):
        return None
    return tuple(float(v) * 255 for v in color)


def _valid_rect(rect):
    return all(math.isfinite(v) for v in rect) and not rect.is_empty and not rect.is_infinite


def _distance(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _rect_paints(drawings, seqno, box):
    """Only a simple later opaque filled rectangle can prove full occlusion.

    Arbitrary path bounding boxes do not prove that their interiors were painted.
    Partial or transparent overlays cannot certify invisibility.
    """
    later = []
    for drawing in drawings:
        if drawing.get("seqno", -1) <= seqno or drawing.get("fill") is None:
            continue
        items = drawing.get("items", [])
        if len(items) != 1 or items[0][0] != "re":
            continue
        rect = fitz.Rect(items[0][1])
        if _valid_rect(rect) and rect.intersects(box):
            later.append({"seqno": drawing["seqno"], "bbox": list(rect),
                          "fill": _plain(drawing["fill"]), "opacity": drawing.get("fill_opacity"),
                          "fully_contains_glyph": rect.contains(box)})
    return later


def _pixel_evidence(pix, view_box, scale, rgb, opacity):
    """Compare the final glyph region with a small surrounding background ring.

    The support test is deliberately conservative: a textured ring, low contrast,
    tiny raster footprint or unsupported color cannot establish visibility.
    """
    mapped = view_box * fitz.Matrix(scale, scale)
    raw = (math.floor(mapped.x0) - pix.x, math.floor(mapped.y0) - pix.y,
           math.ceil(mapped.x1) - pix.x, math.ceil(mapped.y1) - pix.y)
    x0, y0, x1, y1 = max(0, raw[0]), max(0, raw[1]), min(pix.width, raw[2]), min(pix.height, raw[3])
    if x1 <= x0 or y1 <= y0:
        return {"status": "uncertain", "reason": "outside_rendered_page", "pixel_bbox": list(raw)}
    if (x0, y0, x1, y1) != raw:
        return {"status": "uncertain", "reason": "partially_clipped_glyph", "pixel_bbox": list(raw)}
    if (x1 - x0) * (y1 - y0) < 12:
        return {"status": "uncertain", "reason": "insufficient_raster_resolution", "pixel_bbox": list(raw)}
    margin = max(2, math.ceil(scale))
    ox0, oy0 = max(0, x0 - margin), max(0, y0 - margin)
    ox1, oy1 = min(pix.width, x1 + margin), min(pix.height, y1 + margin)
    inner, ring = [], []
    samples = pix.samples_mv
    for y in range(oy0, oy1):
        for x in range(ox0, ox1):
            offset = y * pix.stride + x * pix.n
            pixel = tuple(samples[offset:offset + 3])
            (inner if x0 <= x < x1 and y0 <= y < y1 else ring).append(pixel)
    if not ring:
        return {"status": "uncertain", "reason": "no_background_ring", "pixel_bbox": list(raw)}
    # Quantization only estimates background stability, never changes raw trace.
    bin_color, count = Counter(tuple(v // 8 for v in p) for p in ring).most_common(1)[0]
    dominant = [p for p in ring if tuple(v // 8 for v in p) == bin_color]
    background = tuple(round(sum(p[k] for p in dominant) / len(dominant)) for k in range(3))
    expected = tuple(round(opacity * f + (1 - opacity) * b) for f, b in zip(rgb, background))
    contrast = _distance(expected, background)
    purity = count / len(ring)
    support = sum(_distance(p, expected) <= max(8, contrast * 0.35)
                  and _distance(p, background) >= max(16, contrast * 0.35) for p in inner)
    uniformity = max(_distance(p, background) for p in inner + ring)
    evidence = {"pixel_bbox": list(raw), "background_rgb_255": list(background),
                "background_ring_dominant_fraction": round(purity, 6),
                "expected_composited_rgb_255": list(expected), "contrast_max_channel_255": contrast,
                "foreground_support_pixels": support, "glyph_region_pixels": len(inner),
                "max_region_deviation_from_background_255": uniformity}
    # Exact same-color foreground on a uniformly rendered field is independent
    # of hue: white on white and black on black are treated identically.
    if opacity == 1 and _distance(rgb, background) <= 0.01 and uniformity <= 3:
        return {**evidence, "status": "invisible", "reason": "same_color_on_uniform_rendered_background"}
    if contrast < 24:
        return {**evidence, "status": "uncertain", "reason": "low_contrast_requires_review"}
    if purity < 0.7:
        return {**evidence, "status": "uncertain", "reason": "nonuniform_background_requires_review"}
    if support >= max(3, len(inner) * 0.01):
        return {**evidence, "status": "visible", "reason": "rendered_foreground_contrasts_with_local_background"}
    return {**evidence, "status": "uncertain", "reason": "rendered_foreground_not_established"}


def audit_page_visibility(page, *, render_scale: float = 2.0) -> dict:
    """Audit one PyMuPDF page without mutation; positions use PDF points.

    ``bbox`` is the unrotated extraction frame; ``display_bbox`` follows the
    page rotation used by rendering. A mixed/partially obscured span is uncertain.
    Consumers must retain uncertain/raw records rather than silently filter them.
    """
    if isinstance(render_scale, bool) or not isinstance(render_scale, (int, float)) or not 0.5 <= render_scale <= 4:
        raise ValueError("render_scale must be a finite number from 0.5 to 4")
    native_text = page.get_text()
    traces = page.get_texttrace()
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    try:
        paint_log = page.get_bboxlog()
    except Exception:
        paint_log = None
    pix, render_error = None, None
    try:
        if page.rect.width * page.rect.height * render_scale ** 2 > _MAX_PIXELS:
            raise ValueError("render_pixel_budget_exceeded")
        pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), colorspace=fitz.csRGB,
                              alpha=False, annots=True)
    except Exception as exc:
        render_error = type(exc).__name__ + ": " + str(exc)[:120]
    spans = []
    for trace_index, trace in enumerate(traces):
        rgb, opacity = _rgb(trace), trace.get("opacity")
        glyphs = []
        for glyph_index, char in enumerate(trace.get("chars", ())):
            codepoint, glyph_id, origin, bbox = char
            text = chr(codepoint) if 0 <= codepoint <= 0x10ffff else "\ufffd"
            box = fitz.Rect(bbox)
            view_box = box * page.rotation_matrix
            evidence = {"glyph_index": glyph_index, "codepoint": codepoint, "glyph_id": glyph_id,
                        "text": text, "origin": list(origin), "bbox": list(box), "display_bbox": list(view_box),
                        "raw_char": _plain(char)}
            if text.isspace():
                decision = {"status": "uncertain", "reason": "whitespace_has_no_ink", "nonprinting": True}
            elif trace.get("type") == 3:
                decision = {"status": "invisible", "reason": "nonpainting_text_render_mode"}
            elif opacity == 0:
                decision = {"status": "invisible", "reason": "zero_text_opacity"}
            elif not _valid_rect(box):
                decision = {"status": "uncertain", "reason": "invalid_glyph_geometry"}
            elif trace.get("type") != 0 or rgb is None or not isinstance(opacity, (int, float)) or not 0 < opacity <= 1:
                decision = {"status": "uncertain", "reason": "unsupported_paint_attributes"}
            elif pix is None:
                decision = {"status": "uncertain", "reason": "page_render_unavailable"}
            else:
                paints = _rect_paints(drawings, trace.get("seqno", -1), box)
                later = [] if paint_log is None else [
                    {"seqno": seqno, "kind": op[0], "bbox": list(op[1])}
                    for seqno, op in enumerate(paint_log)
                    if seqno > trace.get("seqno", -1) and op[0] != "ignore-text"
                    and _valid_rect(fitz.Rect(op[1])) and fitz.Rect(op[1]).intersects(box)]
                pixels = _pixel_evidence(pix, view_box, render_scale, rgb, opacity)
                if (any(p["fully_contains_glyph"] and p["opacity"] == 1 for p in paints)
                        and pixels.get("max_region_deviation_from_background_255", 256) <= 3
                        and pixels.get("foreground_support_pixels") == 0):
                    decision = {"status": "invisible", "reason": "later_opaque_rectangle_covers_glyph",
                                "later_paints": paints, "raster_evidence": pixels}
                elif paints or later:
                    decision = {"status": "uncertain", "reason": "partial_or_transparent_overlay_requires_review",
                                "later_paints": paints, "later_operations": later, "raster_evidence": pixels}
                elif paint_log is None and pixels["status"] == "visible":
                    decision = {"status": "uncertain", "reason": "paint_order_unavailable",
                                "raster_evidence": pixels}
                else:
                    decision = pixels
            glyphs.append({**evidence, **decision})
        states = {g["status"] for g in glyphs if not g.get("nonprinting")}
        state = next(iter(states)) if len(states) == 1 else "uncertain"
        spans.append({"trace_index": trace_index, "seqno": trace.get("seqno"),
                      "text": "".join(g["text"] for g in glyphs), "bbox": list(trace["bbox"]),
                      "display_bbox": list(fitz.Rect(trace["bbox"]) * page.rotation_matrix),
                      "status": state, "visible_evidence_eligible": state == "visible",
                      "reason": "uniform_glyph_decisions" if len(states) == 1 else "mixed_or_nonprinting_glyphs_require_review",
                      "raw_trace": _plain(trace), "glyphs": glyphs})
    return {"schema_version": SCHEMA, "page_number": page.number + 1, "page_rotation": page.rotation,
            "page_rect": list(page.rect), "coordinate_unit": "PDF_points",
            "native_text": native_text, "native_text_sha256": hashlib.sha256(native_text.encode()).hexdigest(),
            "render_scale": render_scale, "render_status": "complete" if pix is not None else "failed",
            "render_error": render_error, "renderer": "PyMuPDF", "renderer_version": fitz.VersionBind,
            "render_rgb_sha256": hashlib.sha256(pix.samples_mv).hexdigest() if pix is not None else None,
            "span_counts": dict(Counter(s["status"] for s in spans)), "spans": spans,
            "raw_preserved": True, "page_visual_completeness_certified": False,
            "coverage_status": "native_glyph_visibility_only_page_coverage_unverified",
            "untraced_visible_content_status": "not_assessed_requires_visual_or_ocr_reconciliation",
            "business_correctness_claimed": False,
            "limitations": ["Only traced native text is classified; missing image/outlined text needs separate OCR or visual reconciliation.",
                            "Low contrast, complex backgrounds, unsupported paint and mixed visibility remain uncertain.",
                            "Visibility does not certify reading order, table membership, extraction completeness or business truth."]}


def audit_pdf_visibility(path, *, pages=None, render_scale: float = 2.0) -> dict:
    """Audit selected one-based pages (default all), preserving original PDF bytes."""
    path = Path(path)
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with fitz.open(path) as document:
        selected = list(range(1, len(document) + 1)) if pages is None else list(pages)
        if len(selected) != len(set(selected)) or any(type(p) is not int or not 1 <= p <= len(document) for p in selected):
            raise ValueError("pages must contain unique valid one-based page numbers")
        result = [audit_page_visibility(document[p - 1], render_scale=render_scale) for p in selected]
    return {"schema_version": SCHEMA, "source_path": str(path.resolve()), "source_pdf_sha256": source_sha256,
            "pages": result, "page_numbers": selected, "source_modified": False,
            "raw_preserved": True, "page_visual_completeness_certified": False}
