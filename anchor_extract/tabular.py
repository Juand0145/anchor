"""Flat tabular views over a stitched ``RangeExtraction``.

Four lists of plain dicts, one row per thing, ready for ``pd.DataFrame`` or
``json.dumps``. Nothing here is required by the engine: these are read-only
projections of what the pipeline already produced.

| View | One row per | Answers |
|---|---|---|
| ``segments`` | resolved segment of a unit | what text was extracted, with which role and metadata |
| ``units`` | stitched unit (``extraction.units``) | how the unit resolved (offsets, flags, warnings) |
| ``anchors`` | segment spec the model emitted | what the LLM asked for, before resolution |
| ``calls`` | ``ExtractionResult`` | what each LLM call cost and returned |

``RangeExtraction.results`` keeps its meaning: the per-chunk
``ExtractionResult`` objects. ``calls`` is a projection of that list, not a
replacement for it.

Segment metadata is passed through untouched; no key inside it is read.
"""

from __future__ import annotations

from .inspect import _positional_doc
from .trace_serialization import (
    _block_page,
    _offset_page,
    _requirement_warnings,
    _serialize_resolved_segments,
    _serialize_segment_specs,
    _spec_metadata_list,
)


def _text_slice(doc, start, end) -> str:
    """``full_text[start:end]``, or ``""`` when there is no doc or no span."""
    text = getattr(doc, "full_text", None) if doc is not None else None
    if not isinstance(text, str):
        return ""
    if not (isinstance(start, int) and isinstance(end, int)):
        return ""
    if start < 0 or end < start:
        return ""
    return text[start:end]


def segment_rows(extraction, doc=None) -> list:
    """One row per resolved segment: text, role and metadata in document order.

    ``text`` is the ``full_text`` slice when ``doc`` is given and the segment
    resolved; otherwise the empty string. ``page_start`` is ``None`` without a
    ``doc``. ``metadata`` comes from the segment spec at the same index, or
    ``None`` when the resolved and spec lists have different lengths.
    """
    doc = _positional_doc(doc)
    rows = []
    for req in getattr(extraction, "units", None) or []:
        metadatas = _spec_metadata_list(getattr(req, "segment_anchor_pairs", None))
        resolved = _serialize_resolved_segments(getattr(req, "segments", None))
        for i, seg in enumerate(resolved):
            start, end = seg["start"], seg["end"]
            rows.append({
                "anchor_id": getattr(req, "anchor_id", ""),
                "segment_index": i,
                "role": seg["role"],
                "text": _text_slice(doc, start, end),
                "metadata": metadatas[i] if i < len(metadatas) else None,
                "doc_offset_start": start,
                "doc_offset_end": end,
                "page_start": _offset_page(doc, start) if doc is not None else None,
            })
    return rows


def unit_rows(extraction) -> list:
    """One row per stitched unit: engine ids, resolution flags and warnings."""
    rows = []
    for req in getattr(extraction, "units", None) or []:
        rows.append({
            "anchor_id": getattr(req, "anchor_id", ""),
            "sequence": getattr(req, "sequence", 0),
            "span_key": getattr(req, "span_key", ""),
            "status": req.status,
            "verbatim_match": req.verbatim_match,
            "end_resolved": req.end_resolved,
            "end_inferred_from_next_start": getattr(
                req, "end_inferred_from_next_start", False,
            ),
            "end_anchor_unresolved": getattr(req, "end_anchor_unresolved", False),
            "n_segments": getattr(req, "n_segments", 1),
            "n_segments_resolved": getattr(req, "n_segments_resolved", 0),
            "segments_partial": getattr(req, "segments_partial", False),
            "doc_offset_start": req.doc_offset_start,
            "doc_offset_end": req.doc_offset_end,
            "source_chunk_id": getattr(req, "source_chunk_id", -1),
            "source_chunk_ids": list(getattr(req, "source_chunk_ids", None) or []),
            "warnings": _requirement_warnings(req),
        })
    return rows


def anchor_rows(extraction) -> list:
    """One row per segment spec the model emitted, before resolution.

    Emitted from the specs alone, so a unit whose segments failed to resolve
    still shows the anchors the LLM asked for.
    """
    rows = []
    for req in getattr(extraction, "units", None) or []:
        specs = _serialize_segment_specs(getattr(req, "segment_anchor_pairs", None))
        for i, spec in enumerate(specs):
            rows.append({
                "anchor_id": getattr(req, "anchor_id", ""),
                "segment_index": i,
                "start_anchor": spec["start_anchor"],
                "end_anchor": spec["end_anchor"],
                "end_before_anchor": spec["end_before_anchor"],
                "start_after_anchor": spec["start_after_anchor"],
                "role": spec["role"],
                "metadata": spec["metadata"],
            })
    return rows


def _model_items(res) -> list:
    """Shallow copies of the raw tool items of one call; ``[]`` when absent."""
    payload = getattr(res, "raw_payload", None)
    items = payload.get("requirements") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [dict(item) if isinstance(item, dict) else item for item in items]


def call_rows(extraction, doc=None) -> list:
    """One row per LLM call: telemetry, batch position and raw model items.

    ``chunk_id`` is the 1-based ordinal used everywhere else (``source_chunk_id``,
    ``extraction_call_id``). Chunk user-message text is never included;
    ``page_range`` needs a ``doc`` and is ``[None, None]`` without one.
    """
    doc = _positional_doc(doc)
    has_blocks = isinstance(getattr(doc, "blocks", None), list)
    rows = []
    for ordinal, res in enumerate(getattr(extraction, "results", None) or [], start=1):
        rows.append({
            "chunk_id": ordinal,
            "model_used": res.model_used,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "cache_creation_tokens": res.cache_creation_tokens,
            "cache_read_tokens": res.cache_read_tokens,
            "stop_reason": res.stop_reason,
            "batch_status": res.batch_status,
            "batch_first_block": res.batch_first_block,
            "batch_last_block": res.batch_last_block,
            "batch_doc_offset_start": res.batch_doc_offset_start,
            "batch_doc_offset_end": res.batch_doc_offset_end,
            "page_range": [
                _block_page(doc, res.batch_first_block) if has_blocks else None,
                _block_page(doc, res.batch_last_block) if has_blocks else None,
            ],
            "model_items": _model_items(res),
        })
    return rows


def tabular_views(extraction, doc=None) -> dict:
    """The four views as ``{"segments", "units", "anchors", "calls"}``.

    Every value is a list of dicts with the same keys, so each one maps to a
    DataFrame directly::

        views = extraction.tabular(doc.extraction)
        pd.DataFrame(views["units"])

    ``doc`` is optional and accepts a ``DocumentExtraction`` or an
    ``anchor.Document``; it adds segment ``text``, ``page_start`` and call
    ``page_range``. An empty extraction yields four empty lists.
    """
    return {
        "segments": segment_rows(extraction, doc),
        "units": unit_rows(extraction),
        "anchors": anchor_rows(extraction),
        "calls": call_rows(extraction, doc),
    }
