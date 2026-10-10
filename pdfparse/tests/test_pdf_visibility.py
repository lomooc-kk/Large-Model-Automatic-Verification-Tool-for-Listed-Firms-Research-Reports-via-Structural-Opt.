"""Synthetic paint-order/color counterexamples, independent of report IDs."""
import hashlib
import json

import pymupdf as fitz
import pytest

from yjparse.pdf_visibility import audit_page_visibility, audit_pdf_visibility


def document(*, color=(0, 0, 0), background=(1, 1, 1), opacity=1, mode=0, overlay=None, rotate=0):
    doc = fitz.open()
    page = doc.new_page(width=300, height=160)
    page.draw_rect(page.rect, color=None, fill=background)
    page.insert_text((40, 80), "VISIBLE", fontsize=20, color=color, fill_opacity=opacity, render_mode=mode)
    if overlay:
        box = fitz.Rect(page.get_texttrace()[-1]["bbox"])
        if overlay == "partial":
            box.x1 = box.x0 + box.width / 2
        page.draw_rect(box + (-1,-1,1,1), color=None, fill=background,
                       fill_opacity=0.5 if overlay == "transparent" else 1)
    page.set_rotation(rotate)
    return doc


def word(audit):
    return next(span for span in audit["spans"] if span["text"] == "VISIBLE")


@pytest.mark.parametrize("foreground,background,status", [
    ((1,1,1),(1,1,1),"invisible"),
    ((1,1,1),(0.02,0.17,0.35),"visible"),
    ((0,0,0),(1,1,1),"visible"),
    ((0,0,0),(0,0,0),"invisible"),
    ((0.96,0.96,0.96),(1,1,1),"uncertain"),
    ((0.999,0.999,0.999),(1,1,1),"uncertain"),
])
def test_foreground_and_local_background_jointly_determine_visibility(foreground, background, status):
    with document(color=foreground, background=background) as doc:
        audit = audit_page_visibility(doc[0])
        assert word(audit)["status"] == status
        assert "VISIBLE" in audit["native_text"] and word(audit)["raw_trace"]["chars"]
        assert audit["raw_preserved"] and not audit["page_visual_completeness_certified"]
        json.dumps(audit)


@pytest.mark.parametrize("kwargs,reason", [({"opacity":0},"zero_text_opacity"),
                                         ({"mode":3},"nonpainting_text_render_mode")])
def test_nonpainting_and_fully_transparent_glyphs_are_preserved_in_sidecar(kwargs, reason):
    with document(**kwargs) as doc:
        span = word(audit_page_visibility(doc[0]))
        assert span["status"] == "invisible"
        assert all(g["reason"] == reason for g in span["glyphs"])


@pytest.mark.parametrize("overlay,status", [("full","invisible"),("partial","uncertain"),("transparent","uncertain")])
def test_later_occlusion_is_not_mistaken_for_foreground_support(overlay, status):
    with document(overlay=overlay) as doc:
        span = word(audit_page_visibility(doc[0]))
        assert span["status"] == status
        assert any(g.get("later_paints") for g in span["glyphs"])
        assert span["text"] == "VISIBLE"


@pytest.mark.parametrize("rotation", [0,90,180,270])
def test_rotation_maps_glyphs_to_the_same_rendered_page(rotation):
    with document(color=(1,1,1),background=(0.02,0.17,0.35),rotate=rotation) as doc:
        audit=audit_page_visibility(doc[0]);span=word(audit)
        assert span["status"]=="visible"
        assert audit["page_rotation"]==rotation
        if rotation: assert span["display_bbox"]!=span["bbox"]


def test_tiny_nonzero_opacity_is_uncertain_not_deleted():
    with document(opacity=0.02) as doc:
        assert word(audit_page_visibility(doc[0]))["status"] == "uncertain"


def test_outline_text_does_not_inherit_a_fill_visibility_claim():
    with document(mode=1) as doc:
        assert word(audit_page_visibility(doc[0]))["status"] == "uncertain"


def test_white_title_is_preserved_while_white_hidden_row_is_separate():
    with document(color=(1,1,1),background=(0.02,0.17,0.35)) as doc:
        page=doc[0]
        page.draw_rect(fitz.Rect(0,100,300,160),fill=(1,1,1),color=None)
        page.insert_text((40,135),"HIDDEN",fontsize=20,color=(1,1,1))
        audit=audit_page_visibility(page)
        assert word(audit)["status"]=="visible"
        assert next(s for s in audit["spans"] if s["text"]=="HIDDEN")["status"]=="invisible"


def test_diagonal_shape_bbox_is_not_treated_as_full_opaque_cover():
    with document() as doc:
        page=doc[0];box=fitz.Rect(page.get_texttrace()[-1]["bbox"])
        shape=page.new_shape();shape.draw_polyline([box.tl,box.tr,box.br,box.tl]);shape.finish(fill=(1,1,1),color=None);shape.commit()
        span=word(audit_page_visibility(page))
        assert not any(g["reason"]=="later_opaque_rectangle_covers_glyph" for g in span["glyphs"])
        assert span["status"]=="uncertain"


def test_later_image_is_not_assumed_to_have_opaque_interior():
    with document() as doc:
        page=doc[0];box=fitz.Rect(page.get_texttrace()[-1]["bbox"])
        pix=fitz.Pixmap(fitz.csRGB,fitz.IRect(0,0,60,30),False);pix.clear_with(255)
        page.insert_image(box,pixmap=pix)
        span=word(audit_page_visibility(page))
        assert span["status"]=="uncertain"
        assert any(g.get("later_operations") for g in span["glyphs"])


def test_failed_render_preserves_all_original_text_as_pending(monkeypatch):
    with document() as doc:
        def fail(*args,**kwargs):raise RuntimeError('synthetic renderer unavailable')
        monkeypatch.setattr(fitz.Page,'get_pixmap',fail)
        audit=audit_page_visibility(doc[0])
        assert audit['render_status']=='failed' and word(audit)['status']=='uncertain'
        assert 'VISIBLE' in audit['native_text'] and audit['raw_preserved']
        assert audit['coverage_status']=='native_glyph_visibility_only_page_coverage_unverified'


def test_source_pdf_and_all_glyph_characters_remain_unchanged(tmp_path):
    path=tmp_path/'source.pdf'
    with document(color=(1,1,1)) as doc: doc.save(path)
    original=path.read_bytes()
    audit=audit_pdf_visibility(path,pages=[1])
    assert path.read_bytes()==original
    assert audit["source_pdf_sha256"]==hashlib.sha256(original).hexdigest()
    span=word(audit["pages"][0])
    assert ''.join(chr(g['raw_char'][0]) for g in span['glyphs'])=='VISIBLE'
    assert audit["page_numbers"]==[1]
    with pytest.raises(ValueError):audit_pdf_visibility(path,pages=[0])
    with pytest.raises(ValueError):audit_pdf_visibility(path,pages=[1,1])


@pytest.mark.parametrize("scale", [0,True,5,float('nan')])
def test_bad_render_scale_fails_explicitly(scale):
    with document() as doc:
        with pytest.raises(ValueError):audit_page_visibility(doc[0],render_scale=scale)
