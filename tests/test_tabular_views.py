"""Tabular views: segments, units, anchors and calls over a RangeExtraction."""

from __future__ import annotations

import copy
import unittest

from anchor_extract.anchor_extraction import (
    RangeExtraction,
    ResolvedSegment,
    SegmentSpec,
)
from anchor_extract.tabular import tabular_views
from tests.test_anchor_id_and_metadata import BODY, make_doc
from tests.test_inspect_requirement import _req, _result

VIEW_KEYS = {"segments", "units", "anchors", "calls"}


class TestViewShape(unittest.TestCase):
    def test_empty_extraction_yields_four_empty_lists(self):
        self.assertEqual(
            tabular_views(RangeExtraction([], [])),
            {"segments": [], "units": [], "anchors": [], "calls": []},
        )

    def test_keys_and_row_counts(self):
        extraction = RangeExtraction(
            [_result(), _result()], [_req(sequence=1), _req(sequence=2)],
        )
        views = tabular_views(extraction, doc=make_doc())
        self.assertEqual(set(views), VIEW_KEYS)
        self.assertEqual(len(views["units"]), len(extraction.units))
        self.assertEqual(len(views["calls"]), len(extraction.results))
        self.assertEqual([r["anchor_id"] for r in views["units"]], ["1", "2"])

    def test_rows_share_one_key_set_per_view(self):
        views = tabular_views(
            RangeExtraction([_result()], [_req(sequence=1), _req(sequence=2)]),
            doc=make_doc(),
        )
        for name, rows in views.items():
            with self.subTest(view=name):
                self.assertTrue(rows)
                self.assertEqual(len({tuple(sorted(r)) for r in rows}), 1)

    def test_accepts_an_anchor_document(self):
        import anchor

        positional = make_doc()
        views = tabular_views(
            RangeExtraction([], [_req()]), doc=anchor.Document(positional),
        )
        self.assertEqual(views["segments"][0]["text"], BODY)


class TestSegmentRows(unittest.TestCase):
    def test_text_is_the_full_text_slice(self):
        views = tabular_views(RangeExtraction([], [_req()]), doc=make_doc())
        row = views["segments"][0]
        self.assertEqual(row["text"], BODY)
        self.assertEqual(row["text"], make_doc().full_text[
            row["doc_offset_start"]:row["doc_offset_end"]
        ])
        self.assertEqual(row["role"], "requirement")
        self.assertEqual(row["segment_index"], 0)
        self.assertEqual(row["metadata"], {"req_id": "160.102"})
        self.assertEqual(row["page_start"], 1)

    def test_without_doc_text_is_empty_and_page_is_none(self):
        row = tabular_views(RangeExtraction([], [_req()]))["segments"][0]
        self.assertEqual(row["text"], "")
        self.assertIsNone(row["page_start"])
        self.assertEqual(row["doc_offset_start"], 0)

    def test_one_row_per_resolved_segment_with_aligned_metadata(self):
        req = _req(
            n_segments=2,
            n_segments_resolved=2,
            segments=[
                ResolvedSegment(0, 9, "requirement"),
                ResolvedSegment(10, 24, "context"),
            ],
            segment_anchor_pairs=[
                SegmentSpec("a", "b", metadata={"req_id": "160.102"}),
                SegmentSpec("c", "d", role="context"),
            ],
        )
        rows = tabular_views(RangeExtraction([], [req]), doc=make_doc())["segments"]
        self.assertEqual([r["segment_index"] for r in rows], [0, 1])
        self.assertEqual([r["role"] for r in rows], ["requirement", "context"])
        self.assertEqual([r["metadata"] for r in rows],
                         [{"req_id": "160.102"}, None])
        self.assertEqual(rows[1]["text"], make_doc().full_text[10:24])

    def test_metadata_is_none_when_spec_count_differs(self):
        req = _req(
            segments=[ResolvedSegment(0, 9, "requirement"),
                      ResolvedSegment(10, 24, "context")],
            spec=SegmentSpec("a", "b", metadata={"req_id": "160.102"}),
        )
        rows = tabular_views(RangeExtraction([], [req]))["segments"]
        self.assertEqual([r["metadata"] for r in rows],
                         [{"req_id": "160.102"}, None])


class TestUnitRows(unittest.TestCase):
    def test_engine_ids_and_flags(self):
        row = tabular_views(RangeExtraction([], [_req()]))["units"][0]
        self.assertEqual(row["anchor_id"], "1")
        self.assertEqual(row["sequence"], 1)
        self.assertEqual(row["span_key"], "a322e3195123:0")
        self.assertEqual(row["status"], "complete")
        self.assertTrue(row["verbatim_match"])
        self.assertTrue(row["end_resolved"])
        self.assertEqual(row["source_chunk_ids"], [1])
        self.assertEqual(row["warnings"], [])

    def test_warnings_come_from_the_shared_helper(self):
        req = _req(
            doc_offset_start=-1,
            segments=[],
            n_segments_resolved=0,
            end_resolved=False,
            original_text="",
        )
        row = tabular_views(RangeExtraction([], [req]))["units"][0]
        self.assertEqual(
            row["warnings"], ["start_unresolved", "end_unresolved", "empty_text"],
        )


class TestAnchorRows(unittest.TestCase):
    def test_one_row_per_spec_with_boundary_fields(self):
        req = _req(segment_anchor_pairs=[
            SegmentSpec("§ 160.102", None, end_before_anchor="§ 160.103",
                        metadata={"req_id": "160.102"}),
            SegmentSpec("About", "questions.", role="context",
                        start_after_anchor="About"),
        ])
        rows = tabular_views(RangeExtraction([], [req]))["anchors"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["start_anchor"], "§ 160.102")
        self.assertEqual(rows[0]["end_before_anchor"], "§ 160.103")
        self.assertEqual(rows[0]["metadata"], {"req_id": "160.102"})
        self.assertEqual(rows[1]["role"], "context")
        self.assertEqual(rows[1]["start_after_anchor"], "About")

    def test_emitted_even_when_no_segment_resolved(self):
        req = _req(segments=[], n_segments_resolved=0)
        views = tabular_views(RangeExtraction([], [req]))
        self.assertEqual(views["segments"], [])
        self.assertEqual(len(views["anchors"]), 1)


class TestCallRows(unittest.TestCase):
    def test_telemetry_and_batch_position(self):
        raw = [{"status": "complete", "start_anchor": "§ 160.102"}]
        extraction = RangeExtraction([_result(raw=raw)], [_req()])
        row = tabular_views(extraction, doc=make_doc())["calls"][0]
        self.assertEqual(row["chunk_id"], 1)
        self.assertEqual(row["model_used"], "test-model")
        self.assertEqual(row["input_tokens"], 100)
        self.assertEqual(row["output_tokens"], 40)
        self.assertEqual(row["stop_reason"], "end_turn")
        self.assertEqual(row["batch_status"], "ok")
        self.assertEqual(row["batch_first_block"], 0)
        self.assertEqual(row["batch_doc_offset_end"], len(BODY))
        self.assertEqual(row["page_range"], [1, 1])
        self.assertEqual(row["model_items"], raw)

    def test_chunk_ids_are_one_based(self):
        extraction = RangeExtraction([_result(), _result(), _result()], [])
        self.assertEqual(
            [r["chunk_id"] for r in tabular_views(extraction)["calls"]], [1, 2, 3],
        )

    def test_model_items_empty_without_a_payload_list(self):
        extraction = RangeExtraction([_result(raw_payload={})], [])
        self.assertEqual(tabular_views(extraction)["calls"][0]["model_items"], [])

    def test_model_items_are_copies(self):
        raw = [{"status": "complete"}]
        extraction = RangeExtraction([_result(raw=raw)], [])
        row = tabular_views(extraction)["calls"][0]
        row["model_items"][0]["status"] = "mutated"
        self.assertEqual(raw[0]["status"], "complete")

    def test_page_range_is_none_without_doc(self):
        row = tabular_views(RangeExtraction([_result()], []))["calls"][0]
        self.assertEqual(row["page_range"], [None, None])


class TestPurity(unittest.TestCase):
    def test_views_do_not_mutate_the_extraction(self):
        extraction = RangeExtraction(
            [_result(raw=[{"status": "complete"}])], [_req()],
        )
        before = copy.deepcopy(extraction)
        tabular_views(extraction, doc=make_doc())
        self.assertEqual(len(extraction.results), len(before.results))
        self.assertEqual(len(extraction.units), len(before.units))
        self.assertEqual(
            extraction.results[0].raw_payload, before.results[0].raw_payload,
        )
        req, req_before = extraction.units[0], before.units[0]
        self.assertEqual(req.segments, req_before.segments)
        self.assertEqual(
            req.segment_anchor_pairs[0].metadata,
            req_before.segment_anchor_pairs[0].metadata,
        )

    def test_method_matches_the_function(self):
        extraction = RangeExtraction([_result()], [_req()])
        doc = make_doc()
        self.assertEqual(
            extraction.tabular(doc), tabular_views(extraction, doc),
        )


if __name__ == "__main__":
    unittest.main()
