"""Unit tests for unit-aware _build_batch and section-coverage audit (stdlib)."""

from __future__ import annotations

import unittest

from anchor_extract.anchor_extraction import (
    _audit_missing_section_ids,
    _build_batch,
    _compute_boundary_blocks,
    _estimate_tokens,
)
from anchor_extract.pdf_extraction import DocumentExtraction, TextBlock
from anchor_extract.settings import (
    EXAMPLE_AI_RMF_BOUNDARY_PATTERN,
    EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN,
)


def _make_doc(texts: list[str]) -> DocumentExtraction:
    parts: list[str] = []
    blocks: list[TextBlock] = []
    pos = 0
    for i, t in enumerate(texts):
        if i:
            parts.append("\n")
            pos += 1
        start = pos
        parts.append(t)
        end = start + len(t)
        blocks.append(TextBlock(
            char_start=start,
            char_end=end,
            page=1,
            bbox=(0, 0, 1, 1),
            text=t,
            block_no=i,
            page_block_idx=i,
        ))
        pos = end
    full = "".join(parts)
    return DocumentExtraction(
        pdf_path="synthetic.pdf",
        pdf_hash="0" * 64,
        full_text=full,
        blocks=blocks,
        pages=[],
        parser="test",
        parser_version="0",
        start_page=1,
        end_page=1,
        total_pages_in_pdf=1,
    )


class TestBuildBatch(unittest.TestCase):
    def test_unit_aware_one_unit_per_batch_when_two_do_not_fit(self):
        # 3 units of 5 blocks; ~26 tokens/unit. Target 40 → one unit per batch.
        pad = "x" * 20
        texts = []
        for u in range(3):
            texts.append(f"§ 160.{102 + u} Title{u}.")
            texts.extend([pad] * 4)
        doc = _make_doc(texts)
        last = len(doc.blocks) - 1
        bounds = [0, 5, 10]
        batches = []
        cursor = 0
        while cursor <= last:
            b = _build_batch(doc, cursor, last, 40, boundary_blocks=bounds)
            self.assertIsNotNone(b)
            fi, li, text, _off = b
            starts = [x for x in bounds if fi <= x <= li]
            self.assertEqual(len(starts), 1, f"batch {fi}-{li} spans units {starts}")
            batches.append((fi, li, text))
            cursor = li + 1
        self.assertEqual(len(batches), 3)

    def test_packs_all_units_when_under_budget(self):
        pad = "x" * 20
        texts = []
        for u in range(3):
            texts.append(f"§ 160.{102 + u} Title{u}.")
            texts.extend([pad] * 4)
        doc = _make_doc(texts)
        last = len(doc.blocks) - 1
        bounds = [0, 5, 10]
        b = _build_batch(doc, 0, last, 10_000, boundary_blocks=bounds)
        self.assertIsNotNone(b)
        fi, li, text, _off = b
        self.assertEqual(fi, 0)
        self.assertEqual(li, last)
        self.assertIn("160.102", text)
        self.assertIn("160.104", text)

    def test_packs_two_small_units_then_oversized_alone(self):
        # ~100, ~100, ~5000 tok; budget 1000 → 2 batches (two small + one large).
        def unit(heading: str, target_tok: int) -> list[str]:
            body_chars = max(0, target_tok * 4 - len(heading) - 1)
            return [heading, "x" * body_chars]

        texts = []
        texts.extend(unit("§ 160.101 A.", 100))
        texts.extend(unit("§ 160.102 B.", 100))
        texts.extend(unit("§ 160.103 C.", 5000))
        doc = _make_doc(texts)
        last = len(doc.blocks) - 1
        bounds = [0, 2, 4]
        batches = []
        cursor = 0
        while cursor <= last:
            b = _build_batch(doc, cursor, last, 1000, boundary_blocks=bounds)
            self.assertIsNotNone(b)
            fi, li, _text, _off = b
            batches.append((fi, li, [x for x in bounds if fi <= x <= li]))
            cursor = li + 1
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][2], [0, 2])
        self.assertEqual(batches[1][2], [4])

    def test_ai_rmf_does_not_pack_units_over_budget(self):
        pad = "x" * 20
        texts = []
        for ident in ("GOVERN 1.1", "MAP 1.2", "MEASURE 1.3"):
            texts.append(f"{ident} Title.")
            texts.extend([pad] * 4)
        doc = _make_doc(texts)
        last = len(doc.blocks) - 1
        bounds = _compute_boundary_blocks(
            doc, EXAMPLE_AI_RMF_BOUNDARY_PATTERN, 0, last,
        )
        self.assertEqual(bounds, [0, 5, 10])
        batches = []
        cursor = 0
        while cursor <= last:
            b = _build_batch(doc, cursor, last, 40, boundary_blocks=bounds)
            self.assertIsNotNone(b)
            fi, li, _text, _off = b
            starts = [x for x in bounds if fi <= x <= li]
            self.assertEqual(len(starts), 1, f"batch {fi}-{li} spans units {starts}")
            batches.append((fi, li))
            cursor = li + 1
        self.assertEqual(len(batches), 3)

    def test_greedy_path_packs_consecutive_blocks(self):
        texts = ["aaaa " * 10] * 6
        doc = _make_doc(texts)
        last = len(doc.blocks) - 1
        one = _build_batch(doc, 0, last, 10_000, boundary_blocks=None)
        self.assertIsNotNone(one)
        self.assertEqual(one[0], 0)
        self.assertEqual(one[1], last)
        small = _build_batch(doc, 0, last, 20, boundary_blocks=None)
        self.assertIsNotNone(small)
        self.assertEqual(small[0], 0)
        self.assertLess(small[1], last)

    def test_oversized_first_unit_sent_whole(self):
        texts = ["§ 160.103 Definitions. " + ("word " * 400)]
        texts += ["body " * 50] * 4
        texts.append("§ 160.104 Modifications.")
        texts += ["more " * 20] * 4
        doc = _make_doc(texts)
        bounds = [0, 5]
        last = len(doc.blocks) - 1
        first_unit_tokens = _estimate_tokens(
            doc.text_in_range(doc.blocks[0].char_start, doc.blocks[4].char_end)
        )
        self.assertGreater(first_unit_tokens, 50)
        b = _build_batch(doc, 0, last, 50, boundary_blocks=bounds)
        self.assertIsNotNone(b)
        fi, li, text, _off = b
        self.assertEqual(fi, 0)
        self.assertEqual(li, 4)
        self.assertIn("160.103", text)
        self.assertNotIn("160.104", text)


class TestCoverageAudit(unittest.TestCase):
    def test_missing_160_103(self):
        text = "§ 160.102 Applicability.\nbody\n§ 160.103 Definitions.\nlong\n§ 160.104 Modifications.\n"
        missing = _audit_missing_section_ids(
            text,
            [{"requirement_id": "160.102"}, {"requirement_id": "160.104"}],
            EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN,
        )
        self.assertIn("160.103", missing)
        self.assertNotIn("160.102", missing)

    def test_emitted_160_103_not_missing(self):
        text = "§ 160.102 Applicability.\n§ 160.103 Definitions.\n"
        missing = _audit_missing_section_ids(
            text,
            [{"requirement_id": "160.102"}, {"requirement_id": "160.103"}],
            EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN,
        )
        self.assertEqual(missing, [])

    def test_no_pattern_empty(self):
        self.assertEqual(_audit_missing_section_ids("§ 160.103 Definitions.", [], None), [])

    def test_hipaa_pattern_on_compute_boundary_blocks(self):
        texts = [
            "preamble",
            "§ 160.102 Applicability.",
            "body",
            "§ 164.501 of this subchapter) wrapped xref",
            "§ 160.103 Definitions.",
        ]
        doc = _make_doc(texts)
        bounds = _compute_boundary_blocks(
            doc, EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN, 0, len(doc.blocks) - 1,
        )
        self.assertIn(1, bounds)
        self.assertIn(4, bounds)
        self.assertNotIn(3, bounds)


if __name__ == "__main__":
    unittest.main()
