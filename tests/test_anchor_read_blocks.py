"""Unit tests for the phased public API (read -> blocks)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anchor
from anchor.document import Document
from anchor_extract.pdf_extraction import DocumentExtraction

HIPAA_PDF = Path(__file__).resolve().parents[1] / "pdf" / "hipaa-simplification-201303.pdf"


class TestReadUnsupported(unittest.TestCase):
    def test_unknown_suffix_raises(self):
        with self.assertRaises(ValueError):
            anchor.read("notes.docx")

    def test_blocks_rejects_non_document(self):
        with self.assertRaises(TypeError):
            anchor.blocks({"blocks": []})


class TestReadText(unittest.TestCase):
    def setUp(self):
        fh = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8",
        )
        fh.write("First para line one\nline two\n\n\nSecond para\n")
        fh.close()
        self.path = Path(fh.name)
        self.addCleanup(self.path.unlink)

    def test_blocks_are_paragraphs(self):
        doc = anchor.read(self.path)
        self.assertIsInstance(doc, Document)
        self.assertIsInstance(doc.extraction, DocumentExtraction)
        self.assertEqual(len(doc.extraction.blocks), 2)
        self.assertEqual(doc.extraction.blocks[0].text, "First para line one line two")
        self.assertEqual(doc.extraction.blocks[1].text, "Second para")
        self.assertEqual(len(doc.extraction.pages), 1)

    def test_offset_invariant(self):
        doc = anchor.read(self.path)
        for b in doc.extraction.blocks:
            self.assertEqual(doc.extraction.full_text[b.char_start:b.char_end], b.text)

    def test_blocks_schema(self):
        payload = anchor.read(self.path).blocks()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["source"]["kind"], "text")
        self.assertEqual(len(payload["source"]["hash_prefix"]), 12)
        self.assertNotIn("full_text", payload)
        self.assertEqual(
            set(payload["blocks"][0]),
            {"char_start", "char_end", "page", "bbox", "text", "block_no", "page_block_idx"},
        )

    def test_module_alias_matches_method(self):
        doc = anchor.read(self.path)
        self.assertEqual(anchor.blocks(doc), doc.blocks())
        self.assertEqual(anchor.blocks(doc.extraction), doc.blocks())

    def test_include_full_text(self):
        doc = anchor.read(self.path)
        payload = doc.blocks(include_full_text=True)
        self.assertEqual(payload["full_text"], doc.extraction.full_text)

    def test_page_args_are_noop(self):
        full = anchor.read(self.path)
        paged = anchor.read(self.path, start_page=5, end_page=9)
        self.assertEqual(paged.extraction.full_text, full.extraction.full_text)
        self.assertEqual(paged.extraction.start_page, 1)
        self.assertEqual(paged.extraction.end_page, 1)

    def test_summary_delegates(self):
        doc = anchor.read(self.path)
        self.assertEqual(doc.summary(), doc.extraction.summary())


@unittest.skipUnless(HIPAA_PDF.is_file(), "hipaa PDF missing")
class TestReadPdf(unittest.TestCase):
    def test_blocks_schema_and_invariant(self):
        doc = anchor.read(str(HIPAA_PDF), start_page=11, end_page=12)
        payload = doc.blocks()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["source"]["kind"], "pdf")
        self.assertGreater(len(payload["blocks"]), 0)
        self.assertEqual(len(payload["blocks"]), len(doc.extraction.blocks))
        self.assertEqual(anchor.blocks(doc), payload)
        for b in doc.extraction.blocks:
            self.assertEqual(doc.extraction.full_text[b.char_start:b.char_end], b.text)

    def test_matches_extract_pdf(self):
        from anchor_extract import extract_pdf

        direct = extract_pdf(str(HIPAA_PDF), start_page=11, end_page=12)
        via_read = anchor.read(str(HIPAA_PDF), start_page=11, end_page=12)
        self.assertEqual(via_read.extraction.full_text, direct.full_text)


if __name__ == "__main__":
    unittest.main()
