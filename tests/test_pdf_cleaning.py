"""Unit tests for PDF block noise filtering (stdlib)."""

from __future__ import annotations

import unittest
from pathlib import Path

from anchor_extract.pdf_extraction import (
    FOOTER_BAND_RATIO,
    HEADER_BAND_RATIO,
    _classify_margin_bands,
    _should_drop_block,
    extract_pdf,
)
from anchor_extract.settings import EXAMPLE_HIPAA_DROP_BLOCK_PATTERNS

HIPAA_PDF = Path(__file__).resolve().parents[1] / "pdf" / "hipaa-simplification-201303.pdf"
HEADER = "HIPAA Administrative Simplification Regulation Text"


class TestShouldDropBlock(unittest.TestCase):
    def test_page_number(self):
        self.assertTrue(_should_drop_block("11", (0, 0, 1, 1), 792, False, []))
        self.assertTrue(_should_drop_block("12", None, None, False, [r"^\d{1,4}$"]))

    def test_does_not_drop_paren_enum(self):
        self.assertFalse(_should_drop_block("(1)", (0, 100, 50, 110), 792, True, []))

    def test_hipaa_header_pattern(self):
        self.assertTrue(
            _should_drop_block(HEADER, (320, 100, 540, 110), 792, False,
                               EXAMPLE_HIPAA_DROP_BLOCK_PATTERNS)
        )
        self.assertTrue(
            _should_drop_block("March 2013", (490, 100, 543, 110), 792, False,
                               EXAMPLE_HIPAA_DROP_BLOCK_PATTERNS)
        )

    def test_margin_header_footer_bands(self):
        page_h = 792.0
        header_bbox = (100, 10, 200, 20)  # y1=20 < 0.07*792
        footer_bbox = (100, 750, 200, 770)  # y0=750 > 0.93*792
        body_bbox = (72, 72, 200, 83)
        self.assertTrue(_should_drop_block("x", header_bbox, page_h, True, []))
        self.assertTrue(_should_drop_block("x", footer_bbox, page_h, True, []))
        self.assertFalse(_should_drop_block("§ 160.102 Applicability.", body_bbox, page_h, True, []))

    def test_empty(self):
        self.assertTrue(_should_drop_block("", (0, 0, 1, 1), 792, False, []))
        self.assertTrue(_should_drop_block("   ", (0, 0, 1, 1), 792, False, []))


class TestClassifyMarginBands(unittest.TestCase):
    def test_splits_header_body_footer(self):
        page_h = 100.0
        header = (0, 1, 10, 5, "h", 0, 0)   # y1=5 < 7
        body = (0, 20, 10, 30, "b", 1, 0)
        footer = (0, 95, 10, 99, "f", 2, 0)  # y0=95 > 93
        h, b, f = _classify_margin_bands([header, body, footer], page_h)
        self.assertEqual(len(h), 1)
        self.assertEqual(len(b), 1)
        self.assertEqual(len(f), 1)
        self.assertEqual(HEADER_BAND_RATIO, 0.07)
        self.assertEqual(FOOTER_BAND_RATIO, 0.93)


@unittest.skipUnless(HIPAA_PDF.is_file(), "hipaa PDF missing")
class TestExtractPdfCleaning(unittest.TestCase):
    def test_hipaa_pages_11_12_drop_running_header(self):
        doc = extract_pdf(str(HIPAA_PDF), start_page=11, end_page=12)
        self.assertNotIn(HEADER, doc.full_text)
        self.assertEqual(doc.full_text.count(HEADER), 0)
        self.assertIn("160.102", doc.full_text[:800])
        for b in doc.blocks:
            self.assertEqual(doc.full_text[b.char_start:b.char_end], b.text)
        # Standalone page numbers must not appear as their own blocks.
        for b in doc.blocks:
            self.assertNotRegex(b.text, r"^\d{1,4}$")

    def test_drop_margin_false_keeps_more_or_equal_blocks(self):
        raw = extract_pdf(
            str(HIPAA_PDF), start_page=11, end_page=12, drop_margin_blocks=False,
            drop_block_patterns=[],
        )
        clean = extract_pdf(
            str(HIPAA_PDF), start_page=11, end_page=12, drop_margin_blocks=True,
        )
        self.assertGreaterEqual(len(raw.blocks), len(clean.blocks))
