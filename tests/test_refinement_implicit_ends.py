"""Opt-in refinement pass: windows, merge policy, and id preservation."""

from __future__ import annotations

import unittest
from unittest import mock

from anchor_extract.anchor_extraction import (
    ExtractedUnit,
    ResolvedSegment,
    SegmentSpec,
)
from anchor_extract.refinement import (
    RefinementWindow,
    build_refinement_system_prompt,
    merge_refined_unit,
    needs_refinement,
    refine_implicit_end_units,
    refinement_windows,
)
from tests.test_anchor_id_and_metadata import make_doc
from tests.test_inspect_requirement import _result

TEXT = "AAA BBB CCC"
RANGE_END = len(TEXT)


def _unit(start, end, **overrides) -> ExtractedUnit:
    """A stitched unit with an inferred end, as the primary pass leaves it."""
    kwargs = dict(
        sequence=1,
        span_key=f"a322e3195123:{start}",
        original_text=TEXT[start:end] if start >= 0 else "",
        requirement_text=TEXT[start:end] if start >= 0 else "",
        start_anchor="AAA",
        end_anchor=None,
        status="complete",
        verbatim_match=True,
        chunk_offset_start=-1,
        chunk_offset_end=-1,
        doc_offset_start=start,
        doc_offset_end=end,
        page=1,
        bbox=None,
        segments=[ResolvedSegment(start, end, "requirement")] if start >= 0 else [],
        n_segments=1,
        n_segments_resolved=1,
        segment_anchor_pairs=[SegmentSpec("AAA", None, metadata={"req_id": "A"})],
        source_chunk_id=1,
        source_chunk_ids=[1],
        end_resolved=True,
        end_inferred_from_next_start=True,
    )
    kwargs.update(overrides)
    return ExtractedUnit(**kwargs)


def _chunk_unit(doc_start, doc_end, window_start=0, **overrides) -> ExtractedUnit:
    """A chunk-local unit as returned by a refinement call on a window."""
    kwargs = dict(
        original_text=TEXT[doc_start:doc_end],
        start_anchor="AAA",
        end_anchor="BBB",
        status="complete",
        verbatim_match=True,
        chunk_offset_start=doc_start - window_start,
        chunk_offset_end=doc_end - window_start,
        doc_offset_start=doc_start,
        doc_offset_end=doc_end,
        page=1,
        bbox=None,
        segments=[ResolvedSegment(
            doc_start - window_start, doc_end - window_start, "requirement",
        )],
        n_segments=1,
        n_segments_resolved=1,
        segment_anchor_pairs=[SegmentSpec("AAA", "BBB", metadata={"req_id": "A"})],
        end_resolved=True,
    )
    kwargs.update(overrides)
    return ExtractedUnit(**kwargs)


class TestEligibility(unittest.TestCase):
    def test_inferred_end_and_unresolved_anchor_both_qualify(self):
        self.assertTrue(needs_refinement(_unit(0, 4)))
        self.assertTrue(needs_refinement(_unit(
            0, 4, end_inferred_from_next_start=False, end_anchor_unresolved=True,
        )))

    def test_confirmed_end_does_not_qualify(self):
        self.assertFalse(needs_refinement(_unit(
            0, 4, end_inferred_from_next_start=False, end_anchor="BBB",
        )))


class TestRefinementWindows(unittest.TestCase):
    def test_window_ends_at_the_next_unit_start(self):
        units = [_unit(0, 4), _unit(8, 11, sequence=2)]
        windows = refinement_windows(units, RANGE_END)
        self.assertEqual([(w.index, w.span_start, w.span_end) for w in windows],
                         [(0, 0, 8), (1, 8, 11)])

    def test_last_unit_uses_the_range_end(self):
        windows = refinement_windows([_unit(0, 4)], RANGE_END)
        self.assertEqual([(w.span_start, w.span_end) for w in windows], [(0, 11)])

    def test_last_unit_without_an_end_is_skipped(self):
        self.assertEqual(refinement_windows([_unit(0, -1)], RANGE_END), [])

    def test_unresolved_start_is_skipped(self):
        units = [_unit(-1, -1, sequence=0, span_key=""), _unit(8, 11, sequence=2)]
        self.assertEqual([w.index for w in refinement_windows(units, RANGE_END)], [1])

    def test_confirmed_units_produce_no_windows(self):
        units = [_unit(0, 4, end_inferred_from_next_start=False, end_anchor="x"),
                 _unit(8, 11, sequence=2, end_inferred_from_next_start=False,
                       end_anchor="y")]
        self.assertEqual(refinement_windows(units, RANGE_END), [])

    def test_empty_window_is_skipped(self):
        """A successor starting at or before the candidate leaves nothing to cut."""
        units = [_unit(4, 8), _unit(4, 11, sequence=2)]
        self.assertEqual([w.index for w in refinement_windows(units, RANGE_END)], [1])


class TestMergeRefinedUnit(unittest.TestCase):
    def setUp(self):
        self.doc = make_doc(TEXT)
        self.unit = _unit(0, 4)
        self.window = RefinementWindow(0, 0, 8)

    def _merge(self, *chunk_units):
        return merge_refined_unit(
            self.doc, self.unit, _result(chunk_units=list(chunk_units)), self.window,
        )

    def test_explicit_end_clears_the_flags_and_moves_the_end(self):
        merged = self._merge(_chunk_unit(0, 7))
        self.assertIsNotNone(merged)
        self.assertEqual(merged.doc_offset_end, 7)
        self.assertEqual(merged.end_anchor, "BBB")
        self.assertTrue(merged.end_resolved)
        self.assertFalse(merged.end_inferred_from_next_start)
        self.assertFalse(merged.end_anchor_unresolved)
        self.assertEqual(merged.original_text, "AAA BBB")

    def test_reading_order_ids_are_preserved(self):
        merged = self._merge(_chunk_unit(0, 7))
        self.assertEqual(merged.sequence, self.unit.sequence)
        self.assertEqual(merged.anchor_id, self.unit.anchor_id)
        self.assertEqual(merged.span_key, self.unit.span_key)
        self.assertEqual(merged.doc_offset_start, self.unit.doc_offset_start)
        self.assertEqual(merged.source_chunk_ids, self.unit.source_chunk_ids)

    def test_end_past_the_window_is_rejected(self):
        self.assertIsNone(self._merge(_chunk_unit(0, 11)))

    def test_end_at_the_window_edge_is_accepted(self):
        merged = self._merge(_chunk_unit(0, 8))
        self.assertIsNotNone(merged)
        self.assertEqual(merged.doc_offset_end, 8)

    def test_several_returned_units_are_rejected(self):
        self.assertIsNone(self._merge(_chunk_unit(0, 7), _chunk_unit(0, 4)))

    def test_different_start_is_rejected(self):
        self.assertIsNone(self._merge(_chunk_unit(4, 7)))

    def test_still_unresolved_end_is_rejected(self):
        self.assertIsNone(self._merge(_chunk_unit(0, 7, end_anchor=None)))
        self.assertIsNone(self._merge(_chunk_unit(0, 7, end_resolved=False)))

    def test_empty_result_is_rejected(self):
        self.assertIsNone(self._merge())

    def test_window_local_segments_are_shifted_to_document_offsets(self):
        unit = _unit(4, 8, sequence=2, span_key="a322e3195123:4")
        merged = merge_refined_unit(
            self.doc,
            unit,
            _result(chunk_units=[_chunk_unit(4, 7, window_start=4)]),
            RefinementWindow(0, 4, 11),
        )
        self.assertEqual(
            [(s.start, s.end) for s in merged.segments], [(4, 7)],
        )


class TestRefineImplicitEndUnits(unittest.TestCase):
    """End-to-end through the entry point with the LLM call patched out."""

    def setUp(self):
        self.doc = make_doc(TEXT)

    def test_merges_and_reports_one_result_per_call(self):
        units = [_unit(0, 4), _unit(8, 11, sequence=2,
                                    end_inferred_from_next_start=False,
                                    end_anchor="CCC")]
        with mock.patch(
            "anchor_extract.refinement.extract_requirements_from_chunk",
            return_value=_result(chunk_units=[_chunk_unit(0, 7)]),
        ) as call:
            refined, results = refine_implicit_end_units(
                self.doc, units, "SYSTEM", RANGE_END,
            )
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(refined), len(units))
        self.assertEqual(len(results), 1)
        self.assertEqual(refined[0].doc_offset_end, 7)
        self.assertFalse(refined[0].end_inferred_from_next_start)
        self.assertIs(refined[1], units[1])

    def test_call_receives_the_window_text_and_offset(self):
        with mock.patch(
            "anchor_extract.refinement.extract_requirements_from_chunk",
            return_value=_result(chunk_units=[_chunk_unit(0, 7)]),
        ) as call:
            refine_implicit_end_units(self.doc, [_unit(0, 4)], "SYSTEM", 8)
        kwargs = call.call_args.kwargs
        self.assertEqual(kwargs["chunk_text"], TEXT[:8])
        self.assertEqual(kwargs["chunk_doc_offset_start"], 0)
        self.assertTrue(kwargs["system_prompt"].startswith("SYSTEM"))

    def test_results_carry_the_window_as_their_batch_span(self):
        with mock.patch(
            "anchor_extract.refinement.extract_requirements_from_chunk",
            return_value=_result(chunk_units=[_chunk_unit(0, 7)]),
        ):
            _, results = refine_implicit_end_units(
                self.doc, [_unit(0, 4)], "SYSTEM", 8,
            )
        self.assertEqual(results[0].batch_doc_offset_start, 0)
        self.assertEqual(results[0].batch_doc_offset_end, 8)

    def test_rejected_merge_leaves_the_unit_untouched(self):
        units = [_unit(0, 4)]
        with mock.patch(
            "anchor_extract.refinement.extract_requirements_from_chunk",
            return_value=_result(chunk_units=[_chunk_unit(0, 11)]),
        ):
            refined, results = refine_implicit_end_units(
                self.doc, units, "SYSTEM", 8,
            )
        self.assertIs(refined[0], units[0])
        self.assertTrue(refined[0].end_inferred_from_next_start)
        self.assertEqual(len(results), 1)

    def test_failed_call_is_recorded_without_touching_the_unit(self):
        units = [_unit(0, 4)]
        with mock.patch(
            "anchor_extract.refinement.extract_requirements_from_chunk",
            side_effect=RuntimeError("all models exhausted"),
        ):
            refined, results = refine_implicit_end_units(
                self.doc, units, "SYSTEM", 8,
            )
        self.assertIs(refined[0], units[0])
        self.assertEqual(results[0].batch_status, "api_error")
        self.assertEqual(results[0].batch_doc_offset_start, 0)

    def test_nothing_flagged_issues_no_calls(self):
        units = [_unit(0, 11, end_inferred_from_next_start=False, end_anchor="CCC")]
        with mock.patch(
            "anchor_extract.refinement.extract_requirements_from_chunk",
        ) as call:
            refined, results = refine_implicit_end_units(
                self.doc, units, "SYSTEM", RANGE_END,
            )
        call.assert_not_called()
        self.assertEqual(results, [])
        self.assertEqual(refined, units)


class TestRefinementPrompt(unittest.TestCase):
    def test_addendum_is_appended_to_the_composed_prompt(self):
        built = build_refinement_system_prompt("BASE PROMPT")
        self.assertTrue(built.startswith("BASE PROMPT"))
        self.assertIn("BOUNDED-FRAGMENT REFINEMENT PASS", built)
        self.assertGreater(len(built), len("BASE PROMPT"))


if __name__ == "__main__":
    unittest.main()
