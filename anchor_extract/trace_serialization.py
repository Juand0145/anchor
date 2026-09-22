"""Traceability serialization for the anchor extraction pipeline.

Builds two JSON artifacts from a ``RangeExtraction``:

* ``extraction_run.json`` -- FULL execution trace: run + document + model
  metadata, and for every chunk the call telemetry, the anchors the model
  returned, and how each anchor resolved. This is the debugging surface.
* ``requirements.json`` -- the LEAN deliverable: the final stitched units
  with just enough traceability (run id, page, requirement id, anchors, source
  chunk) to link each one back to its extraction call in the run file.

Design rule: store offsets + hashes, never the raw chunk text (it is fully
reproducible from the PDF + ``DocumentExtraction.blocks``). Body text lives
only in ``requirements.json``. Optional/noisy fields (attempts, malformed
items) are emitted only when non-empty.

ID model:
    document_id        = pdf sha256
    run_id             = "<UTC compact>Z-<pdf_hash[:8]>"   e.g. 20260609T140801Z-a322e319
    chunk_id           = 1-based ordinal of the chunk in RangeExtraction.results
    extraction_call_id = "<run_id>#c<chunk_id>"
    requirement uid    = "<run_id>#r<NNNN>"
    anchor_id          = "1", "2", "3", ...  (1-based reading order)
    span_key           = "<pdf_hash[:12]>:<doc_offset_start>"  (positional key)

Schema 2.0 breaking changes vs 1.0:

* Every unit carries ``anchor_id`` (engine-assigned). ``requirement_id`` no
  longer holds a model-supplied value.
* ``segment_anchor_pairs`` entries changed from a 5-element list to an object
  ``{start_anchor, end_anchor, end_before_anchor, role, start_after_anchor,
  metadata}``.
* Segment ``metadata`` is exported: per unit (from the first segment that has
  it) and per entry in ``segments[]`` when the spec and resolved lists align.
* New warning keys: ``metadata_id_mismatch``, ``empty_business_id``.

Schema 2.1 changes vs 2.0:

* ``requirement_id`` and ``anchor_id`` are the 1-based READING-ORDER sequence
  ("1", "2", "3", ...), empty when the unit's start never resolved. ``sequence``
  carries the same value as an int.
* The positional key ``{pdf_hash[:12]}:{doc_offset_start}`` moved from
  ``anchor_id`` to ``span_key``, emitted per unit and per chunk anchor.

Schema 2.2 changes vs 2.1:

* ``business_id`` is gone from every artifact. The engine holds no domain
  identifier: the source's own id (``req_id``, ``subcategory_id``, ...) travels
  only inside the untouched ``metadata`` object, per unit and per segment.
* Warnings ``metadata_id_mismatch``, ``empty_business_id`` and ``id_mismatch``
  are gone: nothing in the engine validates domain identifiers.
* ``missing_section_ids`` / the ``missing_section`` call warning are gone from
  the trace; heading-coverage checks belong to consumers of the artifacts.
* ``totals`` no longer reports ``n_id_mismatches`` or ``n_missing_section_ids``.
"""

import re
import datetime
from dataclasses import asdict
from pathlib import Path

from .anchor_extraction import (
    SegmentSpec,
    ResolvedSegment,
    _coerce_resolved_segment,
)

SCHEMA_VERSION = "2.2"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def make_run_id(doc, results) -> str:
    """Stable, human-readable run id from the first call's timestamp + doc hash."""
    ts = results[0].timestamp if results else _now_iso()
    base = ts[:19].replace("-", "").replace(":", "")   # 2026-06-09T14:08:01 -> 20260609T140801
    return f"{base}Z-{doc.pdf_hash[:8]}"


def _block_page(doc, idx: int):
    if 0 <= idx < len(doc.blocks):
        return doc.blocks[idx].page
    return None


def _offset_page(doc, offset: int):
    if offset is None or offset < 0:
        return None
    return doc.locate(offset).get("page")


def _derive_title(text: str):
    """Best-effort heading: the first sentence fragment of the unit text."""
    if not text:
        return None
    head = text[:160].strip()
    head = head.strip(" .-:\u2014")
    dot = head.find(".")
    title = (head[:dot] if dot > 0 else head).strip()
    return title[:80] or None


def _requirement_warnings(req) -> list:
    """Controlled-vocabulary warnings for filtering/debugging."""
    w = []
    if req.doc_offset_start is None or req.doc_offset_start < 0:
        w.append("start_unresolved")
    if not req.end_resolved:
        w.append("end_unresolved")
    if not req.original_text:
        w.append("empty_text")
    if getattr(req, "ambiguous", False):
        w.append("ambiguous_anchor")
    if getattr(req, "start_via_fallback", False):
        w.append("resolved_via_prefix_fallback")
    if getattr(req, "segments_partial", False):
        w.append("segments_partial")
    if getattr(req, "n_segments", 1) > 1:
        w.append("multi_span")
    if len(getattr(req, "source_chunk_ids", []) or []) > 1:
        w.append("spans_multiple_chunks")
    return w


def _spec_metadata(spec):
    """Metadata dict of a SegmentSpec or raw model segment dict, else None.

    Serialization-only accessor: the engine never looks inside this object.
    """
    meta = spec.get("metadata") if isinstance(spec, dict) else getattr(spec, "metadata", None)
    return meta if isinstance(meta, dict) else None


def _segment_spec_row(spec) -> dict:
    """One segment spec as a schema 2.0 object (metadata included)."""
    if isinstance(spec, SegmentSpec):
        return {
            "start_anchor": spec.start_anchor,
            "end_anchor": spec.end_anchor,
            "end_before_anchor": spec.end_before_anchor,
            "role": spec.role,
            "start_after_anchor": spec.start_after_anchor,
            "metadata": _spec_metadata(spec),
        }
    if isinstance(spec, dict):
        return {
            "start_anchor": spec.get("start_anchor"),
            "end_anchor": spec.get("end_anchor"),
            "end_before_anchor": spec.get("end_before_anchor"),
            "role": spec.get("role", "requirement"),
            "start_after_anchor": spec.get("start_after_anchor"),
            "metadata": _spec_metadata(spec),
        }
    return {
        "start_anchor": spec[0] if len(spec) > 0 else None,
        "end_anchor": spec[1] if len(spec) > 1 else None,
        "end_before_anchor": spec[2] if len(spec) > 2 else None,
        "role": spec[3] if len(spec) > 3 else "requirement",
        "start_after_anchor": spec[4] if len(spec) > 4 else None,
        "metadata": spec[5] if len(spec) > 5 and isinstance(spec[5], dict) else None,
    }


def _serialize_segment_specs(specs) -> list:
    """Emit segment anchor fields for each segment spec (schema 2.0 objects)."""
    return [
        _segment_spec_row(spec)
        for spec in specs or []
        if isinstance(spec, (SegmentSpec, dict, list, tuple))
    ]


def _spec_metadata_list(specs) -> list:
    """Per-segment metadata dicts (or None) in spec order."""
    return [row["metadata"] for row in _serialize_segment_specs(specs)]


def _unit_metadata(specs) -> dict:
    """Unit-level metadata: shallow copy of the first segment that carries it."""
    for meta in _spec_metadata_list(specs):
        if meta:
            return dict(meta)
    return {}


def _serialize_resolved_segments(segments) -> list:
    """Emit resolved segment offsets with role for trace JSON."""
    out = []
    for seg in segments or []:
        rs = _coerce_resolved_segment(seg)
        out.append({
            "start": rs.start,
            "end": rs.end,
            "role": rs.role,
        })
    return out


def _serialize_resolved_segments_with_doc(doc, batch_offset_start, segments) -> list:
    """Chunk-local resolved segments with doc offsets for extraction_run.json."""
    rows = []
    for seg in segments or []:
        rs = _coerce_resolved_segment(seg)
        rows.append({
            "chunk_offset": [rs.start, rs.end],
            "doc_offset": [
                batch_offset_start + rs.start if rs.start >= 0 else rs.start,
                batch_offset_start + rs.end if rs.end >= 0 else rs.end,
            ],
            "role": rs.role,
            "resolved": rs.start >= 0 and rs.end > rs.start,
        })
    return rows


def _chunk_block(doc, run_id, ordinal, res):
    """One chunk entry for extraction_run.json (call + per-anchor resolution)."""
    anchors = []
    for i, r in enumerate(res.requirements):
        seg_list = getattr(r, "segments", []) or []
        anchors.append({
            "idx": i,
            # Chunk-local anchors carry no reading-order number yet (assigned by
            # the stitcher); span_key links them to the final unit.
            "span_key": getattr(r, "span_key", ""),
            "status": r.status,
            "start_anchor": r.start_anchor,
            "end_anchor": r.end_anchor,
            "segment_anchor_pairs": _serialize_segment_specs(
                getattr(r, "segment_anchor_pairs", []) or []
            ),
            "resolution": {
                "start_found": r.chunk_offset_start >= 0,
                "end_found": r.chunk_offset_end >= 0,
                "chunk_offset": [r.chunk_offset_start, r.chunk_offset_end],
                "doc_offset": [r.doc_offset_start, r.doc_offset_end],
                "segments": _serialize_resolved_segments_with_doc(
                    doc, res.batch_doc_offset_start, seg_list,
                ),
                "n_segments": getattr(r, "n_segments", 1),
                "n_segments_resolved": getattr(r, "n_segments_resolved", 0),
                "segments_partial": getattr(r, "segments_partial", False),
                "verbatim_match": r.verbatim_match,
                "resolved_via_prefix_fallback": getattr(r, "start_via_fallback", False),
                "ambiguous": getattr(r, "ambiguous", False),
            },
        })

    span = max(res.batch_doc_offset_end - res.batch_doc_offset_start, 0)
    call = {
        "model_used": res.model_used,
        "status": res.batch_status,
        "stop_reason": res.stop_reason,
        "response_time_s": res.response_time_s,
        "tokens": {
            "input": res.input_tokens,
            "output": res.output_tokens,
            "cache_creation": res.cache_creation_tokens,
            "cache_read": res.cache_read_tokens,
        },
        "timestamp": res.timestamp,
    }
    # Only surface retries when there was more than one attempt or any non-ok one.
    if res.attempts and (len(res.attempts) > 1 or any(a.outcome != "ok" for a in res.attempts)):
        call["attempts"] = [asdict(a) for a in res.attempts]

    model_output = {
        "n_returned": len(res.requirements),
        "n_malformed": len(res.malformed_items),
        "anchors": anchors,
    }
    if res.malformed_items:
        model_output["malformed"] = res.malformed_items

    return {
        "chunk_id": ordinal,
        "extraction_call_id": f"{run_id}#c{ordinal}",
        "input": {
            "block_range": [res.batch_first_block, res.batch_last_block],
            "page_range": [_block_page(doc, res.batch_first_block),
                           _block_page(doc, res.batch_last_block)],
            "doc_offset": [res.batch_doc_offset_start, res.batch_doc_offset_end],
            "estimated_tokens": span // 4 + 1,
            "chunk_hash": res.chunk_hash,
        },
        "call": call,
        "model_output": model_output,
    }


def build_extraction_run(framework, run_id, doc, extraction, stats,
                         model_config, target_input_tokens) -> dict:
    """Assemble the full execution-trace document."""
    results = extraction.results
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "framework": framework,
        "timestamp_start": results[0].timestamp if results else _now_iso(),
        "timestamp_end": results[-1].timestamp if results else _now_iso(),
        "generated_at": _now_iso(),
        "document": {
            "document_id": doc.pdf_hash,
            "filename": Path(doc.pdf_path).name,
            "pages_processed": [doc.start_page, doc.end_page],
            "total_pages": doc.total_pages_in_pdf,
            "n_blocks": len(doc.blocks),
            "n_chars": len(doc.full_text),
            "parser": f"{doc.parser} {doc.parser_version}".strip(),
        },
        "model_config": {
            "primary_model": model_config.get("model"),
            "fallback_models": model_config.get("fallback_models", []) or [],
            "temperature": model_config.get("temperature", 0.0),
            "max_output_tokens": model_config.get("max_output_tokens"),
            "target_input_tokens": target_input_tokens,
            "dynamic_budget": target_input_tokens is None,
        },
        "totals": stats.to_dict(),
        "chunks": [_chunk_block(doc, run_id, i, res)
                   for i, res in enumerate(results, start=1)],
    }


def build_requirements_doc(framework, run_id, doc, extraction) -> dict:
    """Assemble the lean requirements deliverable."""
    reqs = []
    for k, q in enumerate(extraction.requirements):
        chunk_id = getattr(q, "source_chunk_id", -1)
        spans = getattr(q, "source_chunk_ids", []) or ([chunk_id] if chunk_id >= 0 else [])
        end_off = q.doc_offset_end
        page_end_off = (end_off - 1) if (end_off is not None and end_off > 0) else q.doc_offset_start
        specs = getattr(q, "segment_anchor_pairs", []) or []
        segment_metas = _spec_metadata_list(specs)
        resolved_segments = [
            _coerce_resolved_segment(s) for s in (getattr(q, "segments", []) or [])
        ]
        metas_aligned = len(segment_metas) == len(resolved_segments)
        # The only ids the engine owns: reading order and position. Whatever the
        # source calls this unit stays in `metadata`, unread.
        anchor_id = getattr(q, "anchor_id", "")
        reqs.append({
            "uid": f"{run_id}#r{k:04d}",
            "anchor_id": anchor_id,
            "requirement_id": anchor_id,
            "sequence": getattr(q, "sequence", 0),
            "span_key": getattr(q, "span_key", ""),
            "metadata": _unit_metadata(specs),
            "title": _derive_title(q.original_text),
            "extracted_text": q.original_text,
            "requirement_text": getattr(q, "requirement_text", q.original_text),
            "questionnaire_text": getattr(q, "questionnaire_text", ""),
            "context_text": getattr(q, "context_text", ""),
            "source": {
                "doc_offset": [q.doc_offset_start, q.doc_offset_end],
                "page_range": [_offset_page(doc, q.doc_offset_start),
                               _offset_page(doc, page_end_off)],
            },
            "anchors": {
                "start_anchor": q.start_anchor,
                "end_anchor": q.end_anchor,
                "segment_anchor_pairs": _serialize_segment_specs(specs),
            },
            "segments": [
                {
                    "doc_offset": [rs.start, rs.end],
                    "page_range": [
                        _offset_page(doc, rs.start),
                        _offset_page(doc, rs.end - 1 if rs.end > 0 else rs.start),
                    ],
                    "role": rs.role,
                    "resolved": rs.start >= 0 and rs.end > rs.start,
                    "metadata": segment_metas[i] if metas_aligned else None,
                }
                for i, rs in enumerate(resolved_segments)
            ],
            "trace": {
                "chunk_id": chunk_id,
                "extraction_call_id": f"{run_id}#c{chunk_id}" if chunk_id >= 0 else None,
                "spans_chunks": spans,
            },
            "validation": {
                "status": q.status,
                "end_resolved": q.end_resolved,
                "end_inferred_from_next_start": getattr(q, "end_inferred_from_next_start", False),
                "end_anchor_unresolved": getattr(q, "end_anchor_unresolved", False),
                "verbatim_match": q.verbatim_match,
                "n_segments": getattr(q, "n_segments", 1),
                "n_segments_resolved": getattr(q, "n_segments_resolved", 0),
                "segments_partial": getattr(q, "segments_partial", False),
                "warnings": _requirement_warnings(q),
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "document_id": doc.pdf_hash,
        "framework": framework,
        "generated_at": _now_iso(),
        "n_requirements": len(reqs),
        "requirements": reqs,
    }
