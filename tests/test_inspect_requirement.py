"""Inspection API: metadata, model anchors, and linked LLM calls."""

from __future__ import annotations

import unittest

from anchor_extract.anchor_extraction import (
    ExtractedRequirement,
    ExtractionResult,
    PendingRequirement,
    RangeExtraction,
    ResolvedSegment,
    SegmentSpec,
    _SegmentResolution,
    _build_chunk_requirement_from_segments,
    _stitch_chunk,
)
from anchor_extract.inspect import (
    get_requirement,
    inspect_requirement,
    segment_metadatas,
    unit_metadata,
)
from tests.test_anchor_id_and_metadata import BODY, make_doc


def _req(**overrides) -> ExtractedRequirement:
    spec = overrides.pop(
        "spec",
        SegmentSpec("§ 160.102", "health plans.", metadata={"req_id": "160.102"}),
    )
    kwargs = dict(
        sequence=1,
        span_key="a322e3195123:0",
        original_text=BODY,
        requirement_text=BODY,
        start_anchor="§ 160.102",
        end_anchor="health plans.",
        status="complete",
        verbatim_match=True,
        chunk_offset_start=0,
        chunk_offset_end=len(BODY),
        doc_offset_start=0,
        doc_offset_end=len(BODY),
        page=1,
        bbox=(0.0, 0.0, 0.0, 0.0),
        segments=[ResolvedSegment(0, len(BODY), "requirement")],
        n_segments=1,
        n_segments_resolved=1,
        segment_anchor_pairs=[spec] if spec is not None else [],
        source_chunk_id=1,
        source_chunk_ids=[1],
    )
    kwargs.update(overrides)
    return ExtractedRequirement(**kwargs)


def _result(requirements=None, raw=None, **overrides) -> ExtractionResult:
    kwargs = dict(
        requirements=requirements or [],
        model_used="test-model",
        input_tokens=100,
        output_tokens=40,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        response_time_s=0.2,
        stop_reason="end_turn",
        chunk_hash="abc",
        timestamp="2026-01-01T00:00:00+00:00",
        batch_first_block=0,
        batch_last_block=0,
        batch_doc_offset_start=0,
        batch_doc_offset_end=len(BODY),
        batch_status="ok",
        raw_payload={"requirements": raw or []},
    )
    kwargs.update(overrides)
    return ExtractionResult(**kwargs)


class TestMetadataHelpers(unittest.TestCase):
    def test_unit_metadata_is_a_shallow_copy_of_the_first_nonempty(self):
        second = {"req_id": "160.103", "title": "Definitions"}
        req = _req(segment_anchor_pairs=[
            SegmentSpec("a", "b"),
            SegmentSpec("c", "d", metadata=second),
        ])
        meta = unit_metadata(req)
        self.assertEqual(meta, second)
        meta["req_id"] = "changed"
        self.assertEqual(second["req_id"], "160.103")

    def test_segment_metadatas_follow_spec_order(self):
        req = _req(segment_anchor_pairs=[
            SegmentSpec("a", "b", metadata={"req_id": "160.102"}),
            SegmentSpec("c", "d", role="context"),
        ])
        self.assertEqual(
            segment_metadatas(req),
            [{"req_id": "160.102"}, None],
        )


class TestGetRequirement(unittest.TestCase):
    def test_matches_int_and_str(self):
        extraction = RangeExtraction([], [_req(sequence=2), _req(sequence=1)])
        self.assertEqual(get_requirement(extraction, 1).anchor_id, "1")
        self.assertEqual(get_requirement(extraction, "2").sequence, 2)

    def test_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            get_requirement(RangeExtraction([], [_req()]), 9)


class TestInspectRequirement(unittest.TestCase):
    def test_view_links_call_and_metadata(self):
        raw = {
            "status": "complete",
            "start_anchor": "§ 160.102",
            "end_anchor": "health plans.",
            "segments": [{
                "start_anchor": "§ 160.102",
                "end_anchor": "health plans.",
                "role": "requirement",
                "metadata": {"req_id": "160.102"},
            }],
        }
        req = _req(model_requirement=raw, model_requirement_parts=[raw])
        doc = make_doc()
        extraction = RangeExtraction([_result(raw=[raw])], [req])
        view = inspect_requirement(extraction, "1", doc=doc)
        self.assertEqual(view["metadata"]["unit"]["req_id"], "160.102")
        self.assertEqual(view["metadata"]["segments"][0]["req_id"], "160.102")
        self.assertEqual(view["model_anchors"]["status"], "complete")
        self.assertEqual(view["model_anchors"]["start_anchor"], "§ 160.102")
        self.assertEqual(
            view["model_anchors"]["segments"][0]["end_anchor"], "health plans.",
        )
        self.assertEqual(len(view["llm_calls"]), 1)
        call = view["llm_calls"][0]
        self.assertEqual(call["chunk_id"], 1)
        self.assertEqual(call["model_used"], "test-model")
        self.assertEqual(call["tokens"]["input"], 100)
        self.assertEqual(call["stop_reason"], "end_turn")
        self.assertEqual(call["batch_status"], "ok")
        self.assertEqual(call["block_range"], [0, 0])
        self.assertEqual(call["doc_offset"], [0, len(BODY)])
        self.assertEqual(call["page_range"], [1, 1])
        self.assertEqual(call["model_output"], raw)
        self.assertEqual(view["resolution"]["verbatim_match"], True)
        self.assertEqual(view["resolution"]["warnings"], [])
        self.assertEqual(view["resolution"]["extracted_slice"], BODY)
        self.assertIn("start", view["resolution"]["segments"][0])

    def test_fallback_matches_raw_payload_by_start_anchor(self):
        raw = {"start_anchor": "§ 160.102", "status": "complete", "metadata": {"req_id": "160.102"}}
        req = _req()
        extraction = RangeExtraction([_result(raw=[raw])], [req])
        view = inspect_requirement(extraction, 1)
        self.assertEqual(view["llm_calls"][0]["model_output"], raw)
        self.assertNotIn("extracted_slice", view["resolution"])

    def test_multi_span_omits_bounding_slice(self):
        req = _req(
            n_segments=2,
            segments=[
                ResolvedSegment(0, 10, "requirement"),
                ResolvedSegment(20, 30, "context"),
            ],
        )
        view = inspect_requirement(RangeExtraction([], [req]), 1, doc=make_doc())
        self.assertNotIn("extracted_slice", view["resolution"])
        self.assertIn("multi_span", view["resolution"]["warnings"])


class TestModelRequirementCarry(unittest.TestCase):
    def test_chunk_builder_stores_a_copy_of_the_raw_item(self):
        raw = {"status": "complete", "start_anchor": "§ 160.102", "end_anchor": "plans."}
        spec = SegmentSpec("§ 160.102", "plans.", metadata={"req_id": "160.102"})
        built = _build_chunk_requirement_from_segments(
            raw, [spec], [_SegmentResolution()], BODY, 0, make_doc(),
        )
        raw["status"] = "mutated"
        self.assertEqual(built.model_requirement["status"], "complete")
        self.assertEqual(built.model_requirement_parts, [built.model_requirement])
        self.assertEqual(built.segment_anchor_pairs[0].metadata["req_id"], "160.102")

    def test_stitch_merges_parts_across_chunks(self):
        doc = make_doc()
        opened = {
            "status": "truncated_at_end",
            "start_anchor": "§ 160.102",
        }
        closed = {
            "status": "truncated_at_start",
            "end_anchor": "health plans.",
        }
        final: list = []
        pending = _stitch_chunk(
            doc,
            [_req(
                status="truncated_at_end",
                doc_offset_end=-1,
                chunk_offset_end=-1,
                end_anchor=None,
                model_requirement=opened,
                model_requirement_parts=[opened],
                segment_anchor_pairs=[
                    SegmentSpec("§ 160.102", None, metadata={"req_id": "160.102"}),
                ],
            )],
            0, None, final, len(doc.full_text), chunk_id=1,
        )
        self.assertIsInstance(pending, PendingRequirement)
        self.assertEqual(pending.model_requirement_parts, [opened])
        _stitch_chunk(
            doc,
            [_req(
                status="truncated_at_start",
                start_anchor=None,
                doc_offset_start=-1,
                chunk_offset_start=-1,
                doc_offset_end=40,
                chunk_offset_end=20,
                model_requirement=closed,
                model_requirement_parts=[closed],
                segment_anchor_pairs=[],
            )],
            0, pending, final, len(doc.full_text), chunk_id=2,
        )
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].model_requirement_parts, [opened, closed])
        self.assertEqual(final[0].model_requirement, opened)
        self.assertEqual(final[0].source_chunk_ids, [1, 2])
        self.assertEqual(
            final[0].segment_anchor_pairs[0].metadata, {"req_id": "160.102"},
        )


if __name__ == "__main__":
    unittest.main()
