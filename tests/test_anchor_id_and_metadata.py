"""Schema 2.2: engine ids (sequence, span_key) and opaque segment metadata."""

from __future__ import annotations

import unittest

from anchor_extract import anchor_extraction
from anchor_extract.anchor_extraction import (
    ANCHOR_INPUT_SCHEMA,
    ExtractedRequirement,
    PendingRequirement,
    RangeExtraction,
    ResolvedSegment,
    SegmentSpec,
    _Sequencer,
    _stitch_chunk,
    generate_anchor_id,
    generate_span_key,
)
from anchor_extract.pdf_extraction import DocumentExtraction, PageInfo, TextBlock
from anchor_extract.trace_serialization import (
    SCHEMA_VERSION,
    _requirement_warnings,
    _serialize_segment_specs,
    build_requirements_doc,
)

BODY = "§ 160.102 Applicability. This subchapter applies to health plans."
DOC_HASH = "a322e3195123" + "0" * 52


def make_doc(text: str = BODY) -> DocumentExtraction:
    """Minimal DocumentExtraction; no PDF needed."""
    return DocumentExtraction(
        pdf_path="synthetic.pdf",
        pdf_hash=DOC_HASH,
        full_text=text + "\n\n",
        blocks=[TextBlock(0, len(text), 1, (0.0, 0.0, 0.0, 0.0), text, 0, 0)],
        pages=[PageInfo(1, 612.0, 792.0, 0, len(text) + 2, 1)],
        parser="test",
        parser_version="1",
        start_page=1,
        end_page=1,
        total_pages_in_pdf=1,
    )


class TestGenerateSpanKey(unittest.TestCase):
    def test_hash_and_offset(self):
        doc = make_doc()
        self.assertEqual(generate_span_key(doc, 0), "a322e3195123:0")
        self.assertEqual(generate_span_key(doc, 42), "a322e3195123:42")

    def test_deterministic(self):
        self.assertEqual(generate_span_key(make_doc(), 7), generate_span_key(make_doc(), 7))

    def test_unresolved_start_is_empty(self):
        self.assertEqual(generate_span_key(make_doc(), -1), "")
        self.assertEqual(generate_span_key(None, 0), "")

    def test_legacy_name_is_an_alias(self):
        self.assertIs(generate_anchor_id, generate_span_key)


class TestEngineHoldsNoDomainId(unittest.TestCase):
    """Metadata is carried, never interpreted."""

    def test_business_id_helpers_are_gone(self):
        for name in ("derive_business_id", "validate_metadata_business_id",
                     "BUSINESS_ID_METADATA_KEYS", "_audit_missing_section_ids"):
            self.assertFalse(hasattr(anchor_extraction, name), name)

    def test_requirement_has_no_domain_fields(self):
        req = ExtractedRequirement(
            original_text="x", start_anchor=None, end_anchor=None,
            status="complete", verbatim_match=True,
            chunk_offset_start=0, chunk_offset_end=1,
            doc_offset_start=0, doc_offset_end=1, page=1, bbox=None,
            sequence=3,
        )
        for name in ("business_id", "id_mismatch", "metadata_id_mismatch"):
            self.assertFalse(hasattr(req, name), name)

    def test_requirement_id_is_the_display_id(self):
        req = ExtractedRequirement(
            original_text="x", start_anchor=None, end_anchor=None,
            status="complete", verbatim_match=True,
            chunk_offset_start=0, chunk_offset_end=1,
            doc_offset_start=0, doc_offset_end=1, page=1, bbox=None,
            sequence=3,
        )
        self.assertEqual(req.anchor_id, "3")
        self.assertEqual(req.requirement_id, "3")

    def test_pending_has_no_domain_id(self):
        pending = PendingRequirement(start=0, start_anchor="a", sequence=1)
        self.assertFalse(hasattr(pending, "business_id"))


class TestSchemaRelaxation(unittest.TestCase):
    def test_requirement_id_not_required(self):
        item = ANCHOR_INPUT_SCHEMA["properties"]["requirements"]["items"]
        self.assertNotIn("requirement_id", item["required"])
        self.assertIn("requirement_id", item["properties"])
        self.assertEqual(item["required"], ["start_anchor", "end_anchor", "status"])

    def test_metadata_documented_as_opaque(self):
        seg = ANCHOR_INPUT_SCHEMA["properties"]["requirements"]["items"]["properties"]
        description = seg["segments"]["items"]["properties"]["metadata"]["description"]
        self.assertIn("opaque", description.lower())
        self.assertIn("VERBATIM", description)


class TestSerializeSegmentSpecs(unittest.TestCase):
    def test_metadata_included_as_objects(self):
        rows = _serialize_segment_specs([
            SegmentSpec("a", "b", role="requirement", metadata={"req_id": "160.102"}),
            SegmentSpec("c", "d", role="context"),
        ])
        self.assertEqual(rows[0]["metadata"], {"req_id": "160.102"})
        self.assertIsNone(rows[1]["metadata"])
        self.assertEqual(
            set(rows[0]),
            {"start_anchor", "end_anchor", "end_before_anchor", "role",
             "start_after_anchor", "metadata"},
        )

    def test_dict_spec(self):
        rows = _serialize_segment_specs([{"start_anchor": "a", "metadata": {"chapter": "IV"}}])
        self.assertEqual(rows[0]["metadata"], {"chapter": "IV"})
        self.assertEqual(rows[0]["role"], "requirement")


class TestRequirementsDocExport(unittest.TestCase):
    def build(self, **overrides):
        doc = make_doc()
        spec = SegmentSpec("§ 160.102", "health plans.", role="requirement",
                           metadata={"req_id": "160.102", "title": "Applicability."})
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
            segment_anchor_pairs=[spec],
            source_chunk_id=1,
            source_chunk_ids=[1],
        )
        kwargs.update(overrides)
        req = ExtractedRequirement(**kwargs)
        return build_requirements_doc("hipaa", "run#1", doc, RangeExtraction([], [req]))

    def test_schema_version_is_2_2(self):
        self.assertEqual(SCHEMA_VERSION, "2.2")
        self.assertEqual(self.build()["schema_version"], "2.2")

    def test_requirement_id_is_the_sequence(self):
        unit = self.build()["requirements"][0]
        self.assertEqual(unit["requirement_id"], "1")
        self.assertEqual(unit["anchor_id"], "1")
        self.assertEqual(unit["sequence"], 1)
        self.assertEqual(unit["span_key"], "a322e3195123:0")

    def test_no_business_id_key(self):
        self.assertNotIn("business_id", self.build()["requirements"][0])

    def test_metadata_round_trips_untouched(self):
        unit = self.build()["requirements"][0]
        self.assertEqual(unit["metadata"], {"req_id": "160.102", "title": "Applicability."})
        self.assertEqual(unit["segments"][0]["metadata"], unit["metadata"])
        self.assertEqual(
            unit["anchors"]["segment_anchor_pairs"][0]["metadata"]["req_id"], "160.102",
        )

    def test_title_needs_no_metadata(self):
        unit = self.build(original_text="Applicability. Applies to plans.",
                          segment_anchor_pairs=[])["requirements"][0]
        self.assertEqual(unit["title"], "Applicability")
        self.assertEqual(unit["metadata"], {})

    def test_unresolved_start_has_no_display_id(self):
        unit = self.build(sequence=0, span_key="", doc_offset_start=-1,
                          segments=[])["requirements"][0]
        self.assertEqual(unit["requirement_id"], "")
        self.assertEqual(unit["anchor_id"], "")
        self.assertEqual(unit["sequence"], 0)
        self.assertIn("start_unresolved", unit["validation"]["warnings"])

    def test_segment_metadata_null_when_lengths_differ(self):
        unit = self.build(segments=[
            ResolvedSegment(0, 10, "requirement"),
            ResolvedSegment(11, 20, "context"),
        ])["requirements"][0]
        self.assertIsNone(unit["segments"][0]["metadata"])

    def test_no_identifier_warnings(self):
        warnings = self.build(segment_anchor_pairs=[])["requirements"][0]["validation"]["warnings"]
        for key in ("empty_business_id", "metadata_id_mismatch", "id_mismatch"):
            self.assertNotIn(key, warnings)


class TestRequirementWarningsHelper(unittest.TestCase):
    def test_flags_resolution_only(self):
        req = ExtractedRequirement(
            original_text="", start_anchor=None, end_anchor=None,
            status="complete", verbatim_match=False,
            chunk_offset_start=-1, chunk_offset_end=-1,
            doc_offset_start=-1, doc_offset_end=-1, page=None, bbox=None,
            end_resolved=False,
        )
        self.assertEqual(
            _requirement_warnings(req),
            ["start_unresolved", "end_unresolved", "empty_text"],
        )


class TestStitchContinuation(unittest.TestCase):
    """A continuation closes a pending by position alone."""

    def chunk_req(self, **overrides):
        kwargs = dict(
            original_text="",
            start_anchor=None,
            end_anchor="health plans.",
            status="truncated_at_start",
            verbatim_match=True,
            chunk_offset_start=-1,
            chunk_offset_end=20,
            doc_offset_start=-1,
            doc_offset_end=40,
            page=None,
            bbox=None,
        )
        kwargs.update(overrides)
        return ExtractedRequirement(**kwargs)

    def make_pending(self):
        return PendingRequirement(
            start=0,
            start_anchor="§ 160.102",
            sequence=1,
            span_key="a322e3195123:0",
            start_chunk_id=1,
            chunk_ids=[1],
        )

    def test_closes_pending_and_keeps_engine_ids(self):
        doc = make_doc()
        final: list = []
        pending = _stitch_chunk(doc, [self.chunk_req()], 0, self.make_pending(),
                                final, len(doc.full_text), chunk_id=2)
        self.assertIsNone(pending)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0].anchor_id, "1")
        self.assertEqual(final[0].span_key, "a322e3195123:0")
        self.assertEqual(final[0].doc_offset_start, 0)
        self.assertEqual(final[0].doc_offset_end, 40)
        self.assertEqual(final[0].source_chunk_ids, [1, 2])

    def test_pending_metadata_survives_the_close(self):
        """Metadata is the only carrier of the source id: never drop it."""
        doc = make_doc()
        final: list = []
        pending = self.make_pending()
        pending.segment_anchor_pairs = [
            SegmentSpec("§ 160.102", None, metadata={"req_id": "160.102"}),
        ]
        _stitch_chunk(doc, [self.chunk_req()], 0, pending, final,
                      len(doc.full_text), chunk_id=2)
        self.assertEqual(
            final[0].segment_anchor_pairs[0].metadata, {"req_id": "160.102"},
        )

    def test_open_pending_gets_engine_ids(self):
        doc = make_doc()
        final: list = []
        opened = _stitch_chunk(
            doc,
            [self.chunk_req(
                status="truncated_at_end", doc_offset_start=0, doc_offset_end=-1,
                chunk_offset_start=0, chunk_offset_end=-1, end_anchor=None,
                span_key="a322e3195123:0",
            )],
            0, None, final, len(doc.full_text), chunk_id=1,
        )
        self.assertIsNotNone(opened)
        self.assertEqual(opened.sequence, 1)
        self.assertEqual(opened.span_key, "a322e3195123:0")


class TestSequentialNumbering(unittest.TestCase):
    """anchor_id is 1..N in stitch encounter order, shared with continuations."""

    def chunk_req(self, start, end, metadata=None, **overrides):
        kwargs = dict(
            original_text="",
            start_anchor="a",
            end_anchor="b",
            status="complete",
            verbatim_match=True,
            chunk_offset_start=start,
            chunk_offset_end=end,
            doc_offset_start=start,
            doc_offset_end=end,
            page=1,
            bbox=None,
            segment_anchor_pairs=[SegmentSpec("a", "b", metadata=metadata)],
        )
        kwargs.update(overrides)
        return ExtractedRequirement(**kwargs)

    def test_two_units_numbered_one_and_two(self):
        doc = make_doc()
        final: list = []
        _stitch_chunk(
            doc,
            [self.chunk_req(0, 20, {"req_id": "160.102"}),
             self.chunk_req(20, 40, {"req_id": "160.103"})],
            0, None, final, len(doc.full_text), chunk_id=1, sequencer=_Sequencer(),
        )
        self.assertEqual([r.anchor_id for r in final], ["1", "2"])
        self.assertEqual([r.span_key for r in final],
                         ["a322e3195123:0", "a322e3195123:20"])

    def test_single_span_units_keep_their_metadata(self):
        doc = make_doc()
        final: list = []
        _stitch_chunk(
            doc,
            [self.chunk_req(0, 20, {"req_id": "160.102"}),
             self.chunk_req(20, 40, {"req_id": "160.103"})],
            0, None, final, len(doc.full_text), chunk_id=1, sequencer=_Sequencer(),
        )
        self.assertEqual(
            [r.segment_anchor_pairs[0].metadata for r in final],
            [{"req_id": "160.102"}, {"req_id": "160.103"}],
        )

    def test_numbering_ignores_metadata(self):
        """Repeated, conflicting or absent req_id does not affect numbering."""
        doc = make_doc()
        final: list = []
        _stitch_chunk(
            doc,
            [self.chunk_req(0, 20, {"req_id": "160.102"}),
             self.chunk_req(20, 40, {"req_id": "160.102"}),
             self.chunk_req(40, 60, None)],
            0, None, final, len(doc.full_text), chunk_id=1, sequencer=_Sequencer(),
        )
        self.assertEqual([r.anchor_id for r in final], ["1", "2", "3"])

    def test_sequence_continues_across_chunks(self):
        doc = make_doc()
        final: list = []
        sequencer = _Sequencer()
        _stitch_chunk(doc, [self.chunk_req(0, 20)], 0, None, final,
                      len(doc.full_text), chunk_id=1, sequencer=sequencer)
        _stitch_chunk(doc, [self.chunk_req(20, 40)], 0, None, final,
                      len(doc.full_text), chunk_id=2, sequencer=sequencer)
        self.assertEqual([r.anchor_id for r in final], ["1", "2"])

    def test_continuation_does_not_consume_a_number(self):
        doc = make_doc()
        final: list = []
        sequencer = _Sequencer()
        pending = _stitch_chunk(
            doc,
            [self.chunk_req(0, -1, status="truncated_at_end",
                            chunk_offset_end=-1, end_anchor=None)],
            0, None, final, len(doc.full_text), chunk_id=1, sequencer=sequencer,
        )
        self.assertEqual(pending.sequence, 1)
        _stitch_chunk(
            doc,
            [self.chunk_req(-1, 30, status="truncated_at_start",
                            chunk_offset_start=-1, start_anchor=None),
             self.chunk_req(30, 50)],
            0, pending, final, len(doc.full_text), chunk_id=2, sequencer=sequencer,
        )
        self.assertEqual([r.anchor_id for r in final], ["1", "2"])
        self.assertEqual(len(final), 2)

    def test_unresolved_start_gets_no_number(self):
        doc = make_doc()
        final: list = []
        sequencer = _Sequencer()
        _stitch_chunk(
            doc,
            [self.chunk_req(-1, -1, chunk_offset_start=-1, chunk_offset_end=-1)],
            0, None, final, len(doc.full_text), chunk_id=1, sequencer=sequencer,
        )
        self.assertEqual(final[0].sequence, 0)
        self.assertEqual(final[0].anchor_id, "")
        self.assertEqual(sequencer.next, 1)


if __name__ == "__main__":
    unittest.main()
