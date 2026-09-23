"""Public inspection views over a stitched ``RangeExtraction``.

Reads unit metadata, model-emitted boundary anchors, and the LLM call that
produced each unit. Does not interpret metadata keys.
"""

from __future__ import annotations

from .trace_serialization import (
    _block_page,
    _requirement_warnings,
    _serialize_resolved_segments,
    _serialize_segment_specs,
    _spec_metadata_list,
    _unit_metadata,
)


def unit_metadata(requirement) -> dict:
    """Shallow copy of the first non-empty segment metadata dict, else ``{}``."""
    return _unit_metadata(getattr(requirement, "segment_anchor_pairs", None))


def segment_metadatas(requirement) -> list:
    """Per-segment metadata (or None), aligned with ``segment_anchor_pairs``."""
    return _spec_metadata_list(getattr(requirement, "segment_anchor_pairs", None))


def get_requirement(extraction, anchor_id):
    """Return the stitched unit whose ``sequence`` or ``anchor_id`` matches.

    ``anchor_id`` may be an int or a str. Raises ``KeyError`` when no unit matches.
    """
    wanted_seq = None
    if isinstance(anchor_id, bool):
        wanted_str = str(anchor_id)
    elif isinstance(anchor_id, int):
        wanted_seq = anchor_id
        wanted_str = str(anchor_id) if anchor_id > 0 else ""
    else:
        wanted_str = str(anchor_id).strip()
        if wanted_str.isdigit():
            wanted_seq = int(wanted_str)
    for req in getattr(extraction, "requirements", None) or []:
        if wanted_seq is not None and getattr(req, "sequence", None) == wanted_seq:
            return req
        if getattr(req, "anchor_id", None) == wanted_str:
            return req
    raise KeyError(f"no requirement with anchor_id {anchor_id!r}")


def _positional_doc(doc):
    """``DocumentExtraction``, unwrapping ``anchor.Document`` when passed."""
    if doc is None:
        return None
    inner = getattr(doc, "extraction", None)
    blocks = getattr(inner, "blocks", None) if inner is not None else None
    if inner is not None and isinstance(blocks, list) and isinstance(
        getattr(inner, "full_text", None), str
    ):
        return inner
    return doc


def _first_start_anchor(requirement):
    specs = getattr(requirement, "segment_anchor_pairs", None) or []
    if specs:
        spec = specs[0]
        if isinstance(spec, dict):
            return spec.get("start_anchor")
        return getattr(spec, "start_anchor", None)
    return getattr(requirement, "start_anchor", None)


def _fallback_model_item(res, requirement):
    """Fallback only: match a raw tool item by the first segment start_anchor.

    Production units carry ``model_requirement_parts`` from the chunk builder.
    This scan is for requirements that only have ``ExtractionResult.raw_payload``.
    """
    payload = getattr(res, "raw_payload", None)
    if not isinstance(payload, dict):
        return None
    items = payload.get("requirements")
    if not isinstance(items, list):
        return None
    start = _first_start_anchor(requirement)
    if not start:
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("start_anchor") == start:
            return item
        segments = item.get("segments") or []
        if (
            segments
            and isinstance(segments[0], dict)
            and segments[0].get("start_anchor") == start
        ):
            return item
    return None


def _model_parts(requirement) -> list:
    parts = [
        p for p in (getattr(requirement, "model_requirement_parts", None) or [])
        if isinstance(p, dict)
    ]
    raw = getattr(requirement, "model_requirement", None)
    if isinstance(raw, dict) and raw not in parts:
        parts.append(raw)
    return parts


def _llm_call(extraction, requirement, chunk_id, part, doc):
    results = getattr(extraction, "results", None) or []
    res = None
    if isinstance(chunk_id, int) and 1 <= chunk_id <= len(results):
        res = results[chunk_id - 1]
    model_output = part if part is not None else (
        _fallback_model_item(res, requirement) if res is not None else None
    )
    if res is None:
        return {
            "chunk_id": chunk_id,
            "model_used": None,
            "tokens": None,
            "stop_reason": None,
            "batch_status": None,
            "block_range": None,
            "doc_offset": None,
            "page_range": [None, None],
            "model_output": model_output,
        }
    return {
        "chunk_id": chunk_id,
        "model_used": res.model_used,
        "tokens": {
            "input": res.input_tokens,
            "output": res.output_tokens,
            "cache_creation": res.cache_creation_tokens,
            "cache_read": res.cache_read_tokens,
        },
        "stop_reason": res.stop_reason,
        "batch_status": res.batch_status,
        "block_range": [res.batch_first_block, res.batch_last_block],
        "doc_offset": [res.batch_doc_offset_start, res.batch_doc_offset_end],
        "page_range": [
            _block_page(doc, res.batch_first_block) if doc is not None else None,
            _block_page(doc, res.batch_last_block) if doc is not None else None,
        ],
        "model_output": model_output,
    }


def inspect_requirement(extraction, anchor_id, doc=None) -> dict:
    """Metadata, model anchors, linked LLM calls, and resolution for one unit.

    ``doc`` is optional. When it is a ``DocumentExtraction`` or ``anchor.Document``
    and the unit is a single span with valid offsets, ``resolution`` includes
    ``extracted_slice`` (``full_text[doc_offset_start:doc_offset_end]``).
    Chunk user-message text is omitted; call entries carry block and offset ranges.
    """
    req = get_requirement(extraction, anchor_id)
    positional = _positional_doc(doc)
    chunk_ids = list(getattr(req, "source_chunk_ids", None) or [])
    if not chunk_ids and getattr(req, "source_chunk_id", -1) >= 0:
        chunk_ids = [req.source_chunk_id]
    parts = _model_parts(req)
    resolution = {
        "doc_offset_start": req.doc_offset_start,
        "doc_offset_end": req.doc_offset_end,
        "chunk_offset_start": req.chunk_offset_start,
        "chunk_offset_end": req.chunk_offset_end,
        "verbatim_match": req.verbatim_match,
        "end_resolved": req.end_resolved,
        "end_inferred_from_next_start": getattr(req, "end_inferred_from_next_start", False),
        "end_anchor_unresolved": getattr(req, "end_anchor_unresolved", False),
        "n_segments": getattr(req, "n_segments", 1),
        "n_segments_resolved": getattr(req, "n_segments_resolved", 0),
        "segments_partial": getattr(req, "segments_partial", False),
        "segments": _serialize_resolved_segments(getattr(req, "segments", None)),
        "warnings": _requirement_warnings(req),
    }
    n_segments = getattr(req, "n_segments", 1)
    start = req.doc_offset_start
    end = req.doc_offset_end
    text = getattr(positional, "full_text", None) if positional is not None else None
    if (
        isinstance(text, str)
        and n_segments == 1
        and isinstance(start, int)
        and isinstance(end, int)
        and start >= 0
        and end >= start
    ):
        resolution["extracted_slice"] = text[start:end]
    return {
        "metadata": {
            "unit": unit_metadata(req),
            "segments": segment_metadatas(req),
        },
        "model_anchors": {
            "status": req.status,
            "start_anchor": req.start_anchor,
            "end_anchor": req.end_anchor,
            "segments": _serialize_segment_specs(
                getattr(req, "segment_anchor_pairs", None)
            ),
        },
        "llm_calls": [
            _llm_call(
                extraction,
                req,
                chunk_id,
                parts[i] if i < len(parts) else None,
                positional if positional is not None and isinstance(
                    getattr(positional, "blocks", None), list
                ) else None,
            )
            for i, chunk_id in enumerate(chunk_ids)
        ],
        "resolution": resolution,
    }
