"""Geometry, data-loss and visibility regressions; no real sample IDs in rules."""
import sys
import unittest
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yjparse.financial_table_layout import extract_financial_tables
from yjparse.pdf_visibility import audit_page_visibility


def put(page, x, y, value, *, color=(0, 0, 0), center=False):
    if center:
        x -= fitz.get_text_length(value, fontsize=9) / 2
    page.insert_text((x, y), value, fontsize=9, color=color)


def table(page, *, left=20, top=60, title="资产负债表（百万元）", rows=None, header=True, color=(0, 0, 0)):
    page.insert_text((left, top), title, fontname="china-s", fontsize=9)
    centers = [left + 120, left + 170, left + 220]
    if header:
        for center, year in zip(centers, ["2024", "2025E", "2026E"]):
            put(page, center, top + 20, year, center=True)
    for i, (label, values) in enumerate(rows or [("Cash", ["100", "200", "300"]), ("Total", ["500", "600", "700"]) ]):
        put(page, left, top + 40 + i * 20, label, color=color)
        for center, value in zip(centers, values):
            if value is not None:
                put(page, center + 5, top + 40 + i * 20, value, center=True, color=color)


class TableLayoutTests(unittest.TestCase):
    def setUp(self):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=640, height=800)

    def tearDown(self):
        self.doc.close()

    def test_parallel_tables_and_different_vertical_bands_remain_independent(self):
        table(self.page)
        table(self.page, left=340, top=61, title="利润表（百万元）", rows=[("Revenue", ["10", "20", "30"]), ("Cost", ["5", "7", "9"])])
        table(self.page, top=160, title="现金流量表（百万元）", rows=[("Operating", ["8", "9", "10"]), ("Financing", ["2", "3", "4"])])
        result = extract_financial_tables(self.page)
        self.assertEqual(len(result["tables"]), 3)
        rows = {t["title"]: [r["label"] for r in t["rows"]] for t in result["tables"]}
        self.assertEqual(rows["资产负债表（百万元）"], ["Cash", "Total"])
        self.assertEqual(rows["利润表（百万元）"], ["Revenue", "Cost"])
        self.assertEqual(rows["现金流量表（百万元）"], ["Operating", "Financing"])

    def test_empty_zero_and_unreadable_are_distinct_and_keep_their_year(self):
        table(self.page, rows=[("Growth", [None, "0", "#####"]), ("Ratio", ["-2.4%", "10.4%", "29.1%"])])
        row = extract_financial_tables(self.page)["tables"][0]["rows"][0]
        self.assertEqual([(c["year"], c["raw_text"], c["status"]) for c in row["cells"]],
                         [("2024", None, "blank"), ("2025E", "0", "extracted"), ("2026E", "#####", "unreadable_source_placeholder")])

    def test_only_years_in_prose_are_not_a_table(self):
        put(self.page, 20, 60, "Outlook for the company")
        for x, year in zip([140, 190, 240], ["2024", "2025E", "2026E"]):
            put(self.page, x, 80, year)
        self.assertEqual(extract_financial_tables(self.page)["tables"], [])

    def test_increasing_years_across_two_titles_do_not_merge(self):
        for x, title in [(10, "利润表"), (220, "现金流量表")]:
            self.page.insert_text((x, 60), title, fontname="china-s", fontsize=9)
        for x, year in zip([120, 190, 260, 330], ["2022", "2023", "2024", "2025"]):
            put(self.page, x, 80, year, center=True)
        for left, centers, y in [(10, [120, 190], 100), (220, [260, 330], 110)]:
            for delta, name in [(0, "Cash"), (20, "Total")]:
                put(self.page, left, y + delta, name)
                for x, value in zip(centers, ["10", "20"]):
                    put(self.page, x, y + delta, value, center=True)
        result = extract_financial_tables(self.page)
        # The right label is too close to its first year for this conservative
        # extractor. It may remain unassigned, but must never extend the left
        # statement into an invented four-year table.
        self.assertEqual([t["years"] for t in result["tables"]], [["2022", "2023"]])
        unassigned = {t["text"] for t in result["raw_tokens"] if t["token_id"] in result["unassigned_token_ids"]}
        self.assertTrue({"2024", "2025"} <= unassigned)

    def test_hidden_header_cannot_certify_visible_rows_or_years(self):
        table(self.page, header=False)
        for center, year in zip([140, 190, 240], ["2024", "2025E", "2026E"]):
            put(self.page, center, 80, year, center=True, color=(1, 1, 1))
        extracted = extract_financial_tables(self.page, visibility_audit=audit_page_visibility(self.page))["tables"][0]
        self.assertEqual(extracted["header_visibility"], "invisible")
        self.assertEqual(extracted["header_binding_status"], "pending_header_or_title_visibility")
        self.assertTrue(all(r["requires_review"] for r in extracted["rows"]))
        self.assertTrue(all(c["display_year"] is None for r in extracted["rows"] for c in r["cells"]))
        self.assertEqual([c["raw_year"] for c in extracted["rows"][0]["cells"]], ["2024", "2025E", "2026E"])

    def test_adjacent_untitled_table_cannot_borrow_left_title(self):
        table(self.page)
        table(self.page, left=340, title="附录资料", rows=[("Other", ["1", "2", "3"]), ("Other2", ["4", "5", "6"])])
        result = extract_financial_tables(self.page)
        self.assertEqual(len(result["tables"]), 1)
        self.assertEqual([r["label"] for r in result["tables"][0]["rows"]], ["Cash", "Total"])

    def test_hidden_text_is_preserved_but_not_displayed(self):
        table(self.page, color=(1, 1, 1))
        result = extract_financial_tables(self.page, visibility_audit=audit_page_visibility(self.page))
        row = result["tables"][0]["rows"][0]
        self.assertEqual(row["visibility"], "invisible")
        self.assertFalse(row["display_eligible"])
        self.assertEqual([c["raw_text"] for c in row["cells"]], ["100", "200", "300"])
        self.assertTrue(all(c["display_text"] is None for c in row["cells"]))
        self.assertIn("100", result["native_text"])

    def test_white_on_dark_is_not_blanket_filtered(self):
        self.page.draw_rect(fitz.Rect(10, 85, 275, 130), color=None, fill=(0, 0, 0.4))
        table(self.page, color=(1, 1, 1))
        row = extract_financial_tables(self.page, visibility_audit=audit_page_visibility(self.page))["tables"][0]["rows"][0]
        self.assertTrue(row["display_eligible"])
        self.assertEqual([c["display_text"] for c in row["cells"]], ["100", "200", "300"])

    def test_stale_same_text_different_render_audit_rejected(self):
        table(self.page)
        other = fitz.open()
        try:
            p = other.new_page(width=640, height=800)
            table(p, color=(1, 1, 1))
            with self.assertRaisesRegex(ValueError, "raster does not match"):
                extract_financial_tables(self.page, visibility_audit=audit_page_visibility(p))
        finally:
            other.close()

    def test_repeated_cell_tokens_are_ambiguous_not_summed(self):
        table(self.page)
        put(self.page, 149, 100, "99")
        cell = extract_financial_tables(self.page)["tables"][0]["rows"][0]["cells"][0]
        self.assertEqual(cell["status"], "ambiguous")
        self.assertIsNone(cell["display_text"])
        self.assertIn("99", cell["raw_text"])

    def test_raw_tokens_are_accounted_for_and_unknown_prose_retained(self):
        table(self.page)
        put(self.page, 20, 300, "This explanatory paragraph must not vanish.")
        result = extract_financial_tables(self.page)
        all_ids = {t["token_id"] for t in result["raw_tokens"]}
        assigned = {t["token_id"] for table in result["tables"] for t in table["header_tokens"]}
        assigned.update(token_id for table in result["tables"] for row in table["rows"] for token_id in row["token_ids"])
        assigned.update(t["token_id"] for t in result["raw_tokens"] if any(t["span_id"] == table["title_span_id"] for table in result["tables"]))
        self.assertEqual(all_ids, assigned | set(result["unassigned_token_ids"]))
        self.assertTrue(set(result["unassigned_token_ids"]))
        self.assertFalse(result["competition_ready"])
        self.assertFalse(result["page_visual_completeness_certified"])

    def test_rotation_changes_display_boxes_not_binding(self):
        table(self.page)
        before = extract_financial_tables(self.page)["tables"][0]
        self.page.set_rotation(90)
        after = extract_financial_tables(self.page)["tables"][0]
        self.assertEqual(before["rows"], after["rows"])
        self.assertEqual(before["bbox"], after["bbox"])
        self.assertNotEqual(before["display_bbox"], after["display_bbox"])


if __name__ == "__main__":
    unittest.main()
