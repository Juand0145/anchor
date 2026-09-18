"""Offline batch-plan metrics (no LLM)."""

from __future__ import annotations

import unittest
from pathlib import Path

from anchor_extract.anchor_extraction import (
    _compute_boundary_blocks,
    _estimate_tokens,
    _page_to_first_block_idx,
    _page_to_last_block_idx,
)
from anchor_extract.batch_planning import plan_batches, summarize_batch_plan
from anchor_extract.pdf_extraction import extract_pdf
from anchor_extract.settings import EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN

HIPAA_PDF = Path(__file__).resolve().parents[1] / "pdf" / "hipaa-simplification-201303.pdf"


def _llm_configured() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    try:
        from anchor_extract.llm_client import resolve_llm_provider
        resolve_llm_provider()
        return True
    except Exception:
        return False


@unittest.skipUnless(HIPAA_PDF.is_file(), "hipaa PDF missing")
class TestPlanBatchesHipaa1125(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = extract_pdf(str(HIPAA_PDF), start_page=11, end_page=25)
        cls.first = _page_to_first_block_idx(cls.doc, 11)
        cls.last = _page_to_last_block_idx(cls.doc, 25)
        cls.pattern = EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN
        cls.bounds = _compute_boundary_blocks(
            cls.doc, cls.pattern, cls.first, cls.last,
        )
        # Largest single unit (block span between consecutive boundaries).
        ends = cls.bounds + [cls.last + 1]
        cls.largest_unit_tokens = 0
        for i, start in enumerate(cls.bounds):
            end_blk = min(ends[i + 1] - 1, cls.last)
            text = cls.doc.text_in_range(
                cls.doc.blocks[start].char_start,
                cls.doc.blocks[end_blk].char_end,
            )
            cls.largest_unit_tokens = max(
                cls.largest_unit_tokens, _estimate_tokens(text),
            )

    def test_block_coverage_no_gaps_no_overlaps(self):
        # Multi-unit packing on 11–25: 24 § headings, 2 batches at 12000.
        plans = plan_batches(
            self.doc, self.first, self.last, 12000, self.pattern,
        )
        covered = []
        for p in plans:
            covered.extend(range(p.first_block_idx, p.last_block_idx + 1))
        self.assertEqual(covered, list(range(self.first, self.last + 1)))
        self.assertEqual(len(self.bounds), 24)
        self.assertEqual(len(plans), 2)
        self.assertLess(len(plans), 24)

    def test_multi_unit_packing_respects_budget(self):
        plans_12k = plan_batches(
            self.doc, self.first, self.last, 12000, self.pattern,
        )
        self.assertLessEqual(len(plans_12k), 6)
        self.assertGreaterEqual(max(p.n_units for p in plans_12k), 2)
        total_12k = sum(p.est_input_tokens for p in plans_12k)
        self.assertGreaterEqual(total_12k, 14000)
        self.assertLessEqual(total_12k, 14600)

        plans_6k = plan_batches(
            self.doc, self.first, self.last, 6000, self.pattern,
        )
        self.assertGreaterEqual(len(plans_6k), 3)
        hits_6k = [p for p in plans_6k if p.contains_section_id("160.103")]
        self.assertEqual(len(hits_6k), 1, [p.section_ids for p in plans_6k])
        self.assertGreaterEqual(hits_6k[0].est_input_tokens, 5000)
        self.assertEqual(hits_6k[0].n_units, 1)

        plans_3k = plan_batches(
            self.doc, self.first, self.last, 3000, self.pattern,
        )
        hits_3k = [p for p in plans_3k if p.contains_section_id("160.103")]
        self.assertEqual(len(hits_3k), 1, [p.section_ids for p in plans_3k])
        self.assertGreaterEqual(hits_3k[0].est_input_tokens, 5000)

    def test_103_not_packed_with_104_below_12000(self):
        for budget in (3000, 4000, 6000):
            with self.subTest(budget=budget):
                plans = plan_batches(
                    self.doc, self.first, self.last, budget, self.pattern,
                )
                for p in plans:
                    ids = set(p.section_ids)
                    self.assertFalse(
                        {"160.103", "160.104"} <= ids,
                        f"budget={budget} packed 103+104 in {p.section_ids}",
                    )

    def test_budgets_oversized_unit_may_exceed(self):
        for budget in (3000, 4000, 6000, 12000):
            with self.subTest(budget=budget):
                plans = plan_batches(
                    self.doc, self.first, self.last, budget, self.pattern,
                )
                self.assertTrue(plans)
                cap = max(budget, self.largest_unit_tokens)
                for p in plans:
                    self.assertLessEqual(p.est_input_tokens, cap)
                hits = [p for p in plans if p.contains_section_id("160.103")]
                self.assertEqual(len(hits), 1, [p.section_ids for p in plans])
                self.assertGreaterEqual(hits[0].est_input_tokens, 5000)

    def test_summarize_keys(self):
        plans = plan_batches(
            self.doc, self.first, self.last, 12000, self.pattern,
        )
        s = summarize_batch_plan(plans)
        self.assertEqual(
            set(s),
            {
                "n_batches",
                "max_est_tokens",
                "min_est_tokens",
                "median_est_tokens",
                "max_units_per_batch",
            },
        )
        self.assertGreater(s["max_units_per_batch"], 1)


@unittest.skipUnless(_llm_configured(), "Azure/Anthropic not configured")
class TestLiveProviderConfigured(unittest.TestCase):
    def test_resolve_llm_provider(self):
        from anchor_extract.llm_client import resolve_llm_provider
        self.assertIn(resolve_llm_provider(), ("azure", "anthropic"))
