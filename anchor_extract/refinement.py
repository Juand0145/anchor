"""Optional second pass: confirm inconclusive unit ends on a narrow window.

The forward-only primary pass closes a unit at the next resolved start when the
model supplied no usable ``end_anchor`` (``end_inferred_from_next_start``) or
when the ``end_anchor`` it supplied never resolved (``end_anchor_unresolved``).
Both outcomes bound the unit deterministically but leave its terminal text
unconfirmed. This pass re-asks the model for an explicit end on a fragment
pre-cut between two consecutive unit starts, where the ordering ambiguity that
made the anchor fail inside a full-size chunk is gone.

Cost: v1 issues ONE LLM call per flagged unit and does not group adjacent
windows, so N flagged units cost N extra calls. The pass is opt-in
(``refine_implicit_ends=False`` everywhere by default) and never runs as part of
the primary path.

Reading-order ids are untouched: a merged unit keeps its ``sequence``,
``anchor_id`` and ``span_key``, and the units list keeps its order and length.
A refinement that cannot be matched back to its target is skipped and the
original unit is returned unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .anchor_extraction import (
    _doc_segments_from_chunk_req,
    _failed_batch_result,
    _finalize_requirement,
    _merge_model_parts,
    extract_requirements_from_chunk,
)
from .pdf_extraction import DocumentExtraction

# The refinement instructions are an addendum to the composed primary system
# prompt, authored alongside the generic anchor contract rather than hardcoded.
REFINEMENT_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prompts",
    "refinement.txt",
)


def _load_refinement_addendum(path: str = REFINEMENT_PROMPT_PATH) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Refinement prompt addendum not found at {path}. It must live in "
            "anchor_extract/prompts/refinement.txt (do not hardcode it)."
        )
    if not content:
        raise ValueError(f"Refinement prompt addendum at {path} is empty.")
    return content


REFINEMENT_ADDENDUM = _load_refinement_addendum()


def build_refinement_system_prompt(system_prompt: str) -> str:
    """Append the refinement addendum to an already-composed system prompt.

    The orchestrator only holds the composed prompt (generic contract + profile
    detection rules), so the addendum is layered on top of it instead of being
    recomposed from the raw detection text.
    """
    return (
        system_prompt
        + "\n\n=== BOUNDED-FRAGMENT REFINEMENT PASS ===\n"
        + REFINEMENT_ADDENDUM
    )


@dataclass
class RefinementWindow:
    """A flagged unit's position in the list plus the span its call may use."""
    index: int
    span_start: int
    span_end: int


def needs_refinement(unit) -> bool:
    """True when the unit's end was inferred or its end anchor never resolved."""
    return bool(
        getattr(unit, "end_inferred_from_next_start", False)
        or getattr(unit, "end_anchor_unresolved", False)
    )


def refinement_windows(units: list, range_end_offset: int) -> list:
    """Windows for every refinable unit, bounded by the next unit's start.

    A unit qualifies when its start resolved, it carries an inconclusive end,
    and the span up to the next unit's start -- or ``range_end_offset`` for the
    last unit, which must already have an end -- is non-empty. Units with an
    unresolved start are skipped: there is nothing to cut from.
    """
    windows: list = []
    for i, unit in enumerate(units):
        span_start = getattr(unit, "doc_offset_start", -1)
        if span_start < 0 or not needs_refinement(unit):
            continue
        if i + 1 < len(units):
            span_end = getattr(units[i + 1], "doc_offset_start", -1)
        elif getattr(unit, "doc_offset_end", -1) >= 0:
            span_end = range_end_offset
        else:
            continue
        if span_end <= span_start:
            continue
        windows.append(RefinementWindow(i, span_start, span_end))
    return windows


def merge_refined_unit(
    doc: DocumentExtraction,
    unit,
    result,
    window: RefinementWindow,
):
    """The target unit with its refined end, or ``None`` when not mergeable.

    Conservative by construction: the call must return exactly one unit, that
    unit must start exactly where the target starts, its end must be
    anchor-resolved, and that end must not fall outside the window. Any other
    outcome returns ``None`` so the caller keeps the original unit and its
    flags. ``sequence``, ``span_key`` and the start side are carried over
    untouched; only the end, the segments and the end flags change.
    """
    chunk_units = getattr(result, "chunk_units", None) or []
    if len(chunk_units) != 1:
        return None
    refined = chunk_units[0]
    if refined.doc_offset_start != unit.doc_offset_start:
        return None
    if not (refined.end_resolved and refined.end_anchor):
        return None
    if not (unit.doc_offset_start < refined.doc_offset_end <= window.span_end):
        return None

    return _finalize_requirement(
        doc,
        unit.doc_offset_start,
        refined.doc_offset_end,
        unit.status,
        True,
        unit.start_anchor,
        refined.end_anchor,
        source_chunk_id=unit.source_chunk_id,
        source_chunk_ids=list(unit.source_chunk_ids),
        end_inferred_from_next_start=False,
        end_anchor_unresolved=False,
        segment_offsets=_doc_segments_from_chunk_req(refined, window.span_start),
        n_segments_expected=refined.n_segments,
        segments_partial=refined.segments_partial,
        segment_anchor_pairs=refined.segment_anchor_pairs,
        start_via_fallback=unit.start_via_fallback,
        ambiguous=unit.ambiguous,
        sequence=unit.sequence,
        span_key=unit.span_key,
        model_requirement_parts=_merge_model_parts(unit, refined),
    )


def refine_implicit_end_units(
    doc: DocumentExtraction,
    units: list,
    system_prompt: str,
    range_end_offset: int,
    *,
    model: Optional[str] = None,
    max_output_tokens: int = 4096,
    temperature: float = 0.0,
    verbose: bool = False,
) -> tuple:
    """Re-run extraction on a narrow window for each unit with an inconclusive end.

    Returns ``(units, results)``: a new list of the same length and order with
    merged units replaced in place, and one ``ExtractionResult`` per refinement
    call for the caller to append to ``RangeExtraction.results``. A call that
    raises is recorded as an ``api_error`` result and leaves its unit alone.
    """
    refined_units = list(units)
    results: list = []
    windows = refinement_windows(refined_units, range_end_offset)
    if not windows:
        return refined_units, results

    refinement_prompt = build_refinement_system_prompt(system_prompt)
    for window in windows:
        unit = refined_units[window.index]
        window_text = doc.full_text[window.span_start:window.span_end]
        try:
            result = extract_requirements_from_chunk(
                chunk_text=window_text,
                system_prompt=refinement_prompt,
                doc=doc,
                chunk_doc_offset_start=window.span_start,
                model=model,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            results.append(_failed_batch_result(
                exc, window_text, -1, -1, window.span_start, window.span_end,
            ))
            if verbose:
                print(f"  refine #{unit.anchor_id}: "
                      f"{len(window_text)} chars | FAILED ({exc})")
            continue

        result.batch_doc_offset_start = window.span_start
        result.batch_doc_offset_end = window.span_end
        results.append(result)

        merged = merge_refined_unit(doc, unit, result, window)
        if merged is not None:
            refined_units[window.index] = merged
        if verbose:
            print(f"  refine #{unit.anchor_id}: {len(window_text)} chars | "
                  f"{'merged' if merged is not None else 'skipped'}")
    return refined_units, results
