"""Anchor-based requirement extraction (Anthropic tool-use + deterministic recovery).

Production port of the validated ``anchor.ipynb`` pipeline. The LLM identifies
ONLY semantic boundaries; Python owns offsets, slicing, validation, and
provenance.

Contract:

* For each requirement the model returns anchors only -- never the requirement body.
  A requirement may be a single contiguous span (start_anchor/end_anchor) or an
  ordered ``segments`` array of disjoint spans.
* ``status`` is one of ``complete`` / ``truncated_at_end`` / ``truncated_at_start``.
* Anchors are resolved (whitespace-tolerant) to chunk offsets, translated to
  document offsets, and segment text is sliced from ``doc.full_text``.
  Multi-span ``original_text`` is ``SEGMENT_SEPARATOR.join(segment slices)``.
  Offsets are the source of truth.
* A requirement that spans chunks is stitched via a single ``PendingRequirement``
  state as the orchestrator walks chunks FORWARD ONLY (no jumpback).
* Anything that cannot be closed by a confirmed end anchor is emitted, bounded
  deterministically, and flagged (``end_resolved=False``) -- never dropped,
  never looped on.

The Anthropic client and model-fallback chain are reused from
``anchor_extract.llm_client``.
"""

import os
import time
import json
import datetime
import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import anthropic

from .llm_client import get_anthropic_client, _build_model_chain
from .pdf_extraction import DocumentExtraction


# Join resolved segment slices into role-grouped text. Single slice uses no separator.
SEGMENT_SEPARATOR = "\n\n"

ROLE_REQUIREMENT = "requirement"
ROLE_QUESTIONNAIRE = "questionnaire"
ROLE_CONTEXT = "context"
VALID_SEGMENT_ROLES = frozenset({ROLE_REQUIREMENT, ROLE_QUESTIONNAIRE, ROLE_CONTEXT})


def _normalize_role(role: Optional[str]) -> str:
    if role in VALID_SEGMENT_ROLES:
        return role
    return ROLE_REQUIREMENT


@dataclass
class SegmentSpec:
    """One contiguous segment boundary spec from the model."""
    start_anchor: Optional[str]
    end_anchor: Optional[str] = None
    end_before_anchor: Optional[str] = None
    role: str = ROLE_REQUIREMENT
    start_after_anchor: Optional[str] = None

    def to_list(self) -> list:
        return [
            self.start_anchor,
            self.end_anchor,
            self.end_before_anchor,
            self.role,
            self.start_after_anchor,
        ]


@dataclass
class ResolvedSegment:
    """One resolved contiguous span with document/chunk offsets and role."""
    start: int
    end: int
    role: str = ROLE_REQUIREMENT


def _coerce_resolved_segment(seg) -> ResolvedSegment:
    if isinstance(seg, ResolvedSegment):
        return seg
    if isinstance(seg, (list, tuple)):
        start = seg[0] if len(seg) > 0 else -1
        end = seg[1] if len(seg) > 1 else -1
        role = _normalize_role(seg[2] if len(seg) > 2 else None)
        return ResolvedSegment(start, end, role)
    raise TypeError(f"Cannot coerce segment: {seg!r}")


# --------------------------------------------------------------------------- #
# Tool-use schema: anchors only (framework-agnostic). The framework-specific
# detection rules live in the system prompt composed by build_anchor_system_prompt.
# --------------------------------------------------------------------------- #
ANCHOR_TOOL_NAME = "emit_requirement_anchors"

ANCHOR_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "description": "Requirement boundaries found in the chunk. Empty list is valid when the chunk contains none.",
            "items": {
                "type": "object",
                "properties": {
                    "requirement_id": {
                        "type": "string",
                        "description": "Identifier of the requirement as it appears verbatim in the chunk (e.g. '5.8', '164.312')."
                    },
                    "start_anchor": {
                        "type": ["string", "null"],
                        "description": "PRIMARY anchor. Verbatim, UNIQUE substring marking where the requirement BEGINS. ALWAYS span from the heading/ID line through the first complete sentence (or ~12-25 words) of the body. A bare heading is too short and may collide with a table-of-contents entry or cross-reference; the extra body context is mandatory, not optional. Copy it EXACTLY (including punctuation and section symbols); never summarize. Null ONLY for a 'truncated_at_start' continuation (chunk begins inside a requirement whose heading was in a previous chunk)."
                    },
                    "end_anchor": {
                        "type": ["string", "null"],
                        "description": "REQUIRED by default when the requirement's terminal text is visible in the chunk. Provide a verbatim, UNIQUE end substring: the COMPLETE final sentence, final clause, or regulatory-history bracket through its terminal punctuation/bracket (~12-25 words). It MUST occur after start_anchor. Copy it EXACTLY; never summarize. Leave null ONLY when the end is genuinely not visible because the requirement continues past the chunk/document boundary, or when the next requirement's heading starts immediately after this requirement's final clause/punctuation with no intervening non-requirement prose. For multi-segment requirements, prefer the segments array instead; legacy top-level start_anchor/end_anchor describe a single contiguous span only."
                    },
                    "segments": {
                        "type": "array",
                        "description": "OPTIONAL. Ordered list of disjoint contiguous spans that together form one requirement. Each segment has start_anchor, end_anchor (inclusive end), and optional end_before_anchor (exclusive stop-before heading). Text BETWEEN segments is intentionally excluded. Segments must be in document order and must not overlap. When omitted, the requirement is a single span described by top-level start_anchor/end_anchor.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_anchor": {
                                    "type": ["string", "null"],
                                    "description": "Verbatim, UNIQUE substring marking where this segment BEGINS (~12-25 words). Null only for truncated_at_start continuations of this segment."
                                },
                                "end_anchor": {
                                    "type": ["string", "null"],
                                    "description": "Verbatim, UNIQUE substring marking where this segment ENDS (~12-25 words). Null only when this segment continues past the chunk boundary."
                                },
                                "end_before_anchor": {
                                    "type": ["string", "null"],
                                    "description": "OPTIONAL exclusive end. The segment ends immediately BEFORE the first occurrence (after this segment's start) of this verbatim anchor; the anchor text itself is NOT included in the segment. Use it when the natural boundary is the beginning of the next labeled section/heading. Mutually exclusive with end_anchor; if both are given, end_before_anchor takes precedence. Leave null when not used."
                                },
                                "role": {
                                    "type": "string",
                                    "enum": ["requirement", "questionnaire", "context"],
                                    "description": "Segment role. Default 'requirement'. Segments are grouped by role into separate outputs; different roles are never concatenated."
                                },
                                "start_after_anchor": {
                                    "type": ["string", "null"],
                                    "description": "OPTIONAL exclusive start. The segment begins immediately AFTER the first occurrence (after the previous segment's start) of this verbatim anchor; the anchor text is NOT included. Mutually exclusive with start_anchor for defining the start position; if both are given, start_after_anchor wins for the slice start but start_anchor may still be used for ordering. Use for 'start right after this heading/phrase'."
                                }
                            },
                            "required": ["start_anchor", "end_anchor"]
                        }
                    },
                    "status": {
                        "type": "string",
                        "enum": ["complete", "truncated_at_end", "truncated_at_start"],
                        "description": "'complete': start is in this chunk and the end is determinable here, either via an explicit end_anchor or immediate adjacency where the next requirement starts exactly at this requirement's end. 'truncated_at_end': ONLY valid for the LAST requirement in the chunk -- it starts here but its body runs past the chunk/document boundary with no visible terminal text (end_anchor null); the next chunk continues it. A middle requirement is complete, not truncated_at_end. 'truncated_at_start': began in a previous chunk (start_anchor null); set end_anchor if it ends in this chunk."
                    }
                },
                "required": ["requirement_id", "start_anchor", "end_anchor", "status"]
            }
        }
    },
    "required": ["requirements"]
}


# The generic anchor contract (output shape, boundary model, anchor rules) is
# NOT hardcoded here: it is authored in ``Prompts/anchor.txt`` so it can be
# edited without touching code. The framework-specific detection prompts live
# alongside it in ``Prompts/requirements/``.
ANCHOR_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "prompts",
    "anchor.txt",
)


def _load_anchor_system_prompt(path: str = ANCHOR_PROMPT_PATH) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Generic anchor prompt not found at {path}. It must live in "
            "Prompts/anchor.txt (do not hardcode it)."
        )
    if not content:
        raise ValueError(f"Generic anchor prompt at {path} is empty.")
    return content


ANCHOR_SYSTEM_PROMPT_GENERIC = _load_anchor_system_prompt()


def build_anchor_system_prompt(framework_detection_prompt: str) -> str:
    """Compose the generic anchor contract with a framework-specific detection
    prompt.

    The framework prompt decides ONLY which spans are requirements and how
    ``requirement_id`` is formatted. The output format stays anchors-only as
    defined by the generic block, which explicitly takes precedence over any
    conflicting output instruction the framework prompt may still contain
    (legacy prompts ask for full ``original_text``/``source_quote``).
    """
    return (
        ANCHOR_SYSTEM_PROMPT_GENERIC
        + "\n\n=== FRAMEWORK-SPECIFIC REQUIREMENT DETECTION RULES ===\n"
        + "Use the rules below ONLY to decide which spans are requirements and "
        + "how requirement_id is formatted. IGNORE any instruction below about "
        + "output shape, original_text, source_quote, or returning full "
        + "requirement bodies: the output is anchors-only as defined above.\n\n"
        + framework_detection_prompt
    )


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class ExtractedRequirement:
    """One requirement located by anchors, after deterministic text recovery.

    ``original_text`` is NOT produced by the LLM: for a single span it is the slice
    of ``doc.full_text`` between resolved anchors; for multi-span requirements it
    is ``SEGMENT_SEPARATOR.join(slice for each resolved segment)``. The contiguous
    invariant ``full_text[start:end] == original_text`` holds only for single-span
    requirements; multi-span uses per-segment slices instead.

    ``doc_offset_start``/``doc_offset_end`` are the BOUNDING span (first segment
    start through last segment end). When ``n_segments > 1`` they are NOT a clean
    slice of ``original_text``.
    """
    requirement_id: str
    original_text: str
    start_anchor: Optional[str]
    end_anchor: Optional[str]
    status: str

    # Validation outcomes
    verbatim_match: bool                # the anchors required by `status` all resolved

    # Position within the chunk that was sent to the LLM (-1 if not matched)
    chunk_offset_start: int
    chunk_offset_end: int

    # Position within doc.full_text (-1 if that bound is unresolved)
    doc_offset_start: int
    doc_offset_end: int

    # Physical location of the requirement start (None if not resolved)
    page: Optional[int]
    bbox: Optional[tuple]

    # Resolved segment offsets (document coordinates after stitching; chunk-local
    # before). Provenance source of truth for multi-span requirements.
    segments: list = field(default_factory=list)
    n_segments: int = 1
    n_segments_resolved: int = 0
    segments_partial: bool = False
    # All segment specs from the model (SegmentSpec list; for trace / pending carry-forward).
    segment_anchor_pairs: list = field(default_factory=list)

    # Linear-stitcher observability:
    #   end_resolved=False -> doc_offset_end is a containment bound, not a
    #                         confirmed terminal end.
    #   end_inferred_from_next_start=True -> the requirement was closed at the
    #                         next resolved start in the same chunk (valid for
    #                         immediate adjacency when end_anchor was null;
    #                         suspicious when end_anchor was supplied but failed).
    #   end_anchor_unresolved=True -> the model supplied an end_anchor but it
    #                         did not resolve after the start, so the end is not
    #                         anchor-confirmed (bounded by next-start inference
    #                         or carried as pending).
    #   id_mismatch=True   -> a continuation closed a pending requirement whose id differed.
    end_resolved: bool = True
    end_inferred_from_next_start: bool = False
    end_anchor_unresolved: bool = False
    id_mismatch: bool = False

    # True when the start_anchor did not match exactly and was resolved by the
    # strict unique-prefix fallback (see _find_anchor_unique_prefix). Recorded
    # for traceability; the start OFFSET is identical to an exact match, so the
    # recovered text is unaffected.
    start_via_fallback: bool = False
    # True when the start_anchor could not be resolved deterministically because
    # its text (or longest matching prefix) occurs more than once within the
    # ordering window -- left unresolved on purpose (no guessing).
    ambiguous: bool = False

    # Traceability (set by the stitcher on document-level requirements only):
    # the chunk ordinal(s) -- 1-based index into RangeExtraction.results -- whose
    # call(s) produced this requirement. ``source_chunk_id`` is the START chunk;
    # ``source_chunk_ids`` lists every chunk that contributed text (len > 1 means
    # the requirement was stitched across chunks).
    source_chunk_id: int = -1
    source_chunk_ids: list = field(default_factory=list)
    requirement_text: str = ""
    questionnaire_text: str = ""
    context_text: str = ""


@dataclass
class CallAttempt:
    """Record of a single API call attempt within extract_requirements_from_chunk."""
    model: Optional[str]
    attempt_no: int                    # 1-based within the model's retry budget
    duration_s: float
    outcome: str                       # "ok" | "not_found" | "api_error" | "connection_error"
    error_message: str = ""            # truncated to ~200 chars


@dataclass
class ExtractionResult:
    """Output of one LLM extraction call (chunk-local requirements + provenance)."""
    requirements: list                  # list[ExtractedRequirement]

    # Provenance / observability
    model_used: Optional[str]
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    response_time_s: float
    stop_reason: str
    chunk_hash: str
    timestamp: str

    # Batch position in the document (for provenance / mapping requirements back
    # to the batch that produced them). -1 when not applicable.
    batch_first_block: int = -1
    batch_last_block: int = -1
    batch_doc_offset_start: int = -1
    batch_doc_offset_end: int = -1

    # Explicit per-batch status: "ok" | "empty" | "verbatim_failure" | "json_error" | "api_error"
    batch_status: str = "ok"

    raw_payload: dict = field(default_factory=dict)
    malformed_items: list = field(default_factory=list)
    attempts: list = field(default_factory=list)


def _derive_batch_status(requirements: list, malformed: list, stop_reason: str = "") -> str:
    """Collapse a batch outcome into one explicit status label.

    ``overflow`` (output hit max_tokens, JSON truncated, nothing usable parsed)
    is distinguished from a genuinely ``empty`` chunk -- they look identical in
    the requirement count but mean opposite things for debugging.
    """
    if malformed:
        return "json_error"
    if stop_reason == "max_tokens" and not requirements:
        return "overflow"
    if any(not r.verbatim_match for r in requirements):
        return "verbatim_failure"
    if not requirements:
        return "empty"
    return "ok"


# --------------------------------------------------------------------------- #
# Whitespace-tolerant anchor resolution
# --------------------------------------------------------------------------- #
# Punctuation the model routinely "straightens" when it echoes an anchor, while
# the PDF text layer keeps the typographic form (e.g. "business's" vs the
# source's "business’s"). Folding both sides to a canonical ASCII form makes
# anchor matching tolerant to these swaps. Every mapping is 1 char -> 1 char so
# the normalized index map stays aligned with the raw text and the recovered
# original_text is still sliced verbatim from the source.
_CONFUSABLE_CHARS = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u0060": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-", "\u2012": "-",
    "\u2015": "-", "\u2212": "-",
}
_CONFUSABLE_TABLE = {ord(k): v for k, v in _CONFUSABLE_CHARS.items()}


def _fold_confusables(text: str) -> str:
    """Map typographic quote/dash variants to canonical ASCII (1:1, length-preserving)."""
    return text.translate(_CONFUSABLE_TABLE)


def _normalize_ws_with_map(text: str) -> tuple:
    """Collapse every run of whitespace to a single space and fold confusable
    punctuation to canonical ASCII, returning the normalized string plus a map
    from each normalized index to the raw index in ``text``. Lets us match
    anchors that straddle block boundaries (a PDF newline inside what the model
    sees as one phrase) or that differ only by curly-vs-straight quotes/dashes,
    and map back to exact raw offsets. ``idx_map`` has length len(norm)+1;
    ``idx_map[len(norm)] == len(text)``.
    """
    norm_chars: list = []
    idx_map: list = []
    prev_space = False
    for i, c in enumerate(text):
        if c.isspace():
            if not prev_space:
                norm_chars.append(" ")
                idx_map.append(i)
                prev_space = True
        else:
            norm_chars.append(_CONFUSABLE_CHARS.get(c, c))
            idx_map.append(i)
            prev_space = False
    idx_map.append(len(text))
    return "".join(norm_chars), idx_map


def _find_anchor_raw(norm: str, idx_map: list, anchor: Optional[str],
                     from_norm: int = 0, hi_norm: Optional[int] = None) -> Optional[tuple]:
    """Whitespace/punctuation-tolerant search for ``anchor`` inside the window
    ``[from_norm, hi_norm)``. Returns (raw_start, raw_end, norm_start, norm_end)
    for the first match, else None. Strictly forward from ``from_norm`` (no
    global fallback), so an end anchor searched after a resolved start can never
    bind to an earlier occurrence.
    """
    if not anchor:
        return None
    needle = _fold_confusables(re.sub(r"\s+", " ", anchor).strip())
    if not needle:
        return None
    if hi_norm is None:
        hi_norm = len(norm)
    p = norm.find(needle, from_norm)
    if p < 0 or p + len(needle) > hi_norm:
        return None
    norm_end = p + len(needle)
    raw_start = idx_map[p]
    raw_end = idx_map[norm_end - 1] + 1
    return (raw_start, raw_end, p, norm_end)


_FALLBACK_MIN_WORDS = 4
_FALLBACK_MIN_CHARS = 24


def _find_anchor_unique_prefix(norm: str, idx_map: list, anchor: Optional[str],
                               from_norm: int = 0, hi_norm: Optional[int] = None,
                               min_words: int = _FALLBACK_MIN_WORDS,
                               min_chars: int = _FALLBACK_MIN_CHARS) -> Optional[tuple]:
    """Strict prefix fallback for an anchor that did NOT match exactly, bounded
    to the window ``[from_norm, hi_norm)``.

    Shortens the (whitespace-collapsed, confusable-folded) anchor word by word
    from the END and accepts a match ONLY when EXACTLY ONE occurrence of the
    prefix exists inside the window. Returns (raw_start, raw_end, norm_start,
    norm_end) or None.

    This is deterministic, NOT fuzzy. Occurrence count is monotonically
    non-decreasing as the prefix shrinks, so the first (longest) prefix that
    occurs is taken; if that longest-matching prefix is AMBIGUOUS (>=2 matches
    in the window) the anchor is left UNRESOLVED rather than guessing -- a false
    positive is worse than a failed extraction. Shortening stops at
    ``min_words`` / ``min_chars`` to avoid degenerate short matches.

    The window is what disambiguates repeated anchor text: the caller passes the
    previous requirement's start as the lower bound and the next requirement's
    start as the upper bound, so only a candidate satisfying
    previous_start < current_start < next_start is accepted.

    Intended for START anchors only: every prefix shares the anchor's start
    position, so recovery is loss-free and never shrinks the requirement body.
    """
    if not anchor:
        return None
    if hi_norm is None:
        hi_norm = len(norm)
    needle = _fold_confusables(re.sub(r"\s+", " ", anchor).strip())
    words = needle.split(" ")
    for k in range(len(words), min_words - 1, -1):
        prefix = " ".join(words[:k])
        if len(prefix) < min_chars:
            break
        # Count occurrences whose match falls fully inside [from_norm, hi_norm),
        # stopping at 2 (we only need unique vs ambiguous).
        first = -1
        count = 0
        p = norm.find(prefix, from_norm)
        while p != -1 and p + len(prefix) <= hi_norm:
            count += 1
            if first < 0:
                first = p
            if count > 1:
                break
            p = norm.find(prefix, p + 1)
        if count == 0:
            continue
        if count == 1:
            norm_end = first + len(prefix)
            return (idx_map[first], idx_map[norm_end - 1] + 1, first, norm_end)
        # Longest matching prefix is ambiguous within the window; shorter
        # prefixes can only be MORE ambiguous. Do not guess.
        return None
    return None


def _find_anchor_unique_suffix(norm: str, idx_map: list, anchor: Optional[str],
                               from_norm: int = 0, hi_norm: Optional[int] = None,
                               min_words: int = _FALLBACK_MIN_WORDS,
                               min_chars: int = _FALLBACK_MIN_CHARS) -> Optional[tuple]:
    """Strict suffix fallback for an end anchor that did NOT match exactly,
    bounded to the window ``[from_norm, hi_norm)``.

    Shortens the (whitespace-collapsed, confusable-folded) anchor word by word
    from the FRONT (drops leading words) and accepts a match ONLY when EXACTLY
    ONE occurrence of the suffix exists inside the window. Returns
    (raw_start, raw_end, norm_start, norm_end) or None.

    Intended for END anchors only: every suffix shares the anchor's terminal
    position, so recovery is loss-free and never truncates the requirement body
    at the tail.
    """
    if not anchor:
        return None
    if hi_norm is None:
        hi_norm = len(norm)
    needle = _fold_confusables(re.sub(r"\s+", " ", anchor).strip())
    words = needle.split(" ")
    for k in range(len(words), min_words - 1, -1):
        suffix = " ".join(words[len(words) - k:])
        if len(suffix) < min_chars:
            break
        first = -1
        count = 0
        p = norm.find(suffix, from_norm)
        while p != -1 and p + len(suffix) <= hi_norm:
            count += 1
            if first < 0:
                first = p
            if count > 1:
                break
            p = norm.find(suffix, p + 1)
        if count == 0:
            continue
        if count == 1:
            norm_end = first + len(suffix)
            return (idx_map[first], idx_map[norm_end - 1] + 1, first, norm_end)
        return None
    return None


def _normalize_model_segments(raw: dict) -> tuple:
    """Normalize a model requirement dict to an ordered list of SegmentSpec.

    Returns (segment_specs, malformed_reason). Prefer non-empty ``segments``;
    otherwise fall back to legacy top-level start_anchor/end_anchor as one segment.
    """
    status = raw.get("status", "")
    segments_raw = raw.get("segments")
    if isinstance(segments_raw, list) and len(segments_raw) > 0:
        specs: list = []
        for seg in segments_raw:
            if not isinstance(seg, dict):
                return [], "invalid_segment_item"
            specs.append(SegmentSpec(
                start_anchor=seg.get("start_anchor"),
                end_anchor=seg.get("end_anchor"),
                end_before_anchor=seg.get("end_before_anchor"),
                role=_normalize_role(seg.get("role")),
                start_after_anchor=seg.get("start_after_anchor"),
            ))
        return specs, None
    sa = raw.get("start_anchor")
    ea = raw.get("end_anchor")
    if sa is not None or ea is not None:
        return [SegmentSpec(sa, ea, None)], None
    if status == "truncated_at_start":
        return [SegmentSpec(None, raw.get("end_anchor"), None)], None
    return [], "missing_anchors"


def _join_segment_slices(slices: list) -> str:
    """Join segment text slices; single-span uses direct slice (no separator)."""
    if not slices:
        return ""
    if len(slices) == 1:
        return slices[0]
    return SEGMENT_SEPARATOR.join(slices)


def _join_role_slices(full_text: str, segments: list, role: str) -> str:
    """Join slices for one role from resolved segments."""
    slices = [
        full_text[s.start:s.end]
        for s in (_coerce_resolved_segment(seg) for seg in segments)
        if s.role == role and s.start >= 0 and s.end > s.start
    ]
    return _join_segment_slices(slices)


def build_role_texts(full_text: str, segments: list) -> dict:
    """Return joined text per role from resolved segments."""
    coerced = [_coerce_resolved_segment(s) for s in segments]
    return {
        ROLE_REQUIREMENT: _join_role_slices(full_text, coerced, ROLE_REQUIREMENT),
        ROLE_QUESTIONNAIRE: _join_role_slices(full_text, coerced, ROLE_QUESTIONNAIRE),
        ROLE_CONTEXT: _join_role_slices(full_text, coerced, ROLE_CONTEXT),
    }


def _clip_overlapping_segments(segments: list) -> tuple:
    """Sort segments by start; clip later overlaps. Returns (clipped, was_partial)."""
    partial = False
    coerced = [_coerce_resolved_segment(s) for s in segments]
    valid = [s for s in coerced if s.start >= 0 and s.end > s.start]
    if not valid:
        return [], partial
    valid.sort(key=lambda x: x.start)
    clipped: list = []
    for seg in valid:
        if clipped and seg.start < clipped[-1].end:
            partial = True
            new_start = clipped[-1].end
            if seg.end <= new_start:
                continue
            seg = ResolvedSegment(new_start, seg.end, seg.role)
        clipped.append(seg)
    return clipped, partial


def _resolve_anchors(norm: str, idx_map: list, start_anchor: Optional[str],
                     end_anchor: Optional[str],
                     allow_prefix_fallback: bool = True) -> tuple:
    """Resolve a SINGLE requirement's boundary anchors against a PRECOMPUTED
    normalized chunk, with no ordering context. Returns (start_off, end_off,
    start_found, end_found, start_via_fallback) as chunk-local offsets.

    Kept as a primitive; the production path uses ``_resolve_chunk_anchors``,
    which adds the previous/next ordering window. ``end_off`` is exclusive and
    INCLUDES the end anchor text. The end anchor is searched strictly AFTER the
    start, exact-only (trimming its tail would silently drop requirement text).
    """
    start_off, end_off = -1, -1
    start_found = end_found = False
    start_via_fallback = False
    search_from = 0

    s = _find_anchor_raw(norm, idx_map, start_anchor)
    if s is None and start_anchor and allow_prefix_fallback:
        s = _find_anchor_unique_prefix(norm, idx_map, start_anchor)
        if s is not None:
            start_via_fallback = True
    if s is not None:
        start_off, search_from = s[0], s[3]
        start_found = True

    e = _find_anchor_raw(norm, idx_map, end_anchor, from_norm=search_from)
    if e is not None:
        end_off = e[1]
        end_found = True

    return (start_off, end_off, start_found, end_found, start_via_fallback)


@dataclass
class _SegmentResolution:
    """Chunk-local resolution outcome for one segment anchor spec."""
    start_off: int = -1
    end_off: int = -1
    start_found: bool = False
    end_found: bool = False
    start_via_fallback: bool = False
    start_exclusive: bool = False
    end_exclusive: bool = False
    ambiguous: bool = False
    start_norm: int = -1
    start_norm_end: int = -1


def _resolve_chunk_anchors(norm: str, idx_map: list,
                           requirements_segments: list) -> list:
    """Resolve segment anchors for every requirement in a chunk WITH global ordering.

    ``requirements_segments`` is a list (one per requirement) of SegmentSpec
    lists. Returns, per requirement, a list of ``_SegmentResolution`` in segment
    order.

    All segment starts across all requirements share one forward ordering cursor
    so ``previous_start < this_start < next_start`` holds globally.
    """
    flat: list = []
    for req_i, segs in enumerate(requirements_segments):
        for seg_i, spec in enumerate(segs):
            flat.append((req_i, seg_i, spec))

    recs = [_SegmentResolution() for _ in flat]

    cursor = 0
    for idx, (_ri, _si, spec) in enumerate(flat):
        if spec.start_after_anchor:
            saa = spec.start_after_anchor
            m = _find_anchor_raw(norm, idx_map, saa, from_norm=cursor)
            if m is not None:
                recs[idx].start_off = m[1]
                recs[idx].start_norm = m[2]
                recs[idx].start_norm_end = m[3]
                recs[idx].start_found = True
                recs[idx].start_exclusive = True
                cursor = m[2] + 1
            continue
        sa = spec.start_anchor
        if not sa:
            continue
        m = _find_anchor_raw(norm, idx_map, sa, from_norm=cursor)
        if m is not None:
            recs[idx].start_off = m[0]
            recs[idx].start_norm = m[2]
            recs[idx].start_norm_end = m[3]
            recs[idx].start_found = True
            cursor = m[2] + 1

    for idx, (_ri, _si, spec) in enumerate(flat):
        if recs[idx].start_found or spec.start_after_anchor:
            continue
        sa = spec.start_anchor
        if not sa:
            continue
        lo = 0
        for j in range(idx - 1, -1, -1):
            if recs[j].start_found:
                lo = recs[j].start_norm + 1
                break
        hi = len(norm)
        for j in range(idx + 1, len(flat)):
            if recs[j].start_found:
                hi = recs[j].start_norm
                break
        m = _find_anchor_unique_prefix(norm, idx_map, sa, from_norm=lo, hi_norm=hi)
        if m is None:
            recs[idx].ambiguous = True
            continue
        recs[idx].start_off = m[0]
        recs[idx].start_norm = m[2]
        recs[idx].start_norm_end = m[3]
        recs[idx].start_found = True
        recs[idx].start_via_fallback = True

    for idx, (_ri, _si, spec) in enumerate(flat):
        eba = spec.end_before_anchor
        ea = spec.end_anchor if not eba else None
        if not eba and not ea:
            continue
        frm = recs[idx].start_norm_end if recs[idx].start_found else 0
        hi = len(norm)
        for j in range(idx + 1, len(flat)):
            if recs[j].start_found:
                # end_before may target the same heading as the next segment's
                # start anchor; use start_norm_end so the boundary is inside [frm, hi).
                hi = recs[j].start_norm_end if eba else recs[j].start_norm
                break
        seg_start_raw = recs[idx].start_off if recs[idx].start_found else -1
        if eba:
            m = _find_anchor_raw(norm, idx_map, eba, from_norm=frm, hi_norm=hi)
            if m is not None and m[0] > seg_start_raw:
                recs[idx].end_off = m[0]
                recs[idx].end_found = True
                recs[idx].end_exclusive = True
        else:
            m = _find_anchor_raw(norm, idx_map, ea, from_norm=frm, hi_norm=hi)
            if m is None:
                m = _find_anchor_unique_suffix(norm, idx_map, ea, from_norm=frm, hi_norm=hi)
            if m is not None:
                recs[idx].end_off = m[1]
                recs[idx].end_found = True

    grouped: list = [[] for _ in range(len(requirements_segments))]
    for idx, (req_i, _seg_i, _spec) in enumerate(flat):
        grouped[req_i].append(recs[idx])
    return grouped


def _build_chunk_requirement_from_segments(
    raw: dict,
    segment_specs: list,
    segment_resolutions: list,
    chunk_text: str,
    chunk_doc_offset_start: int,
    doc: Optional[DocumentExtraction],
) -> ExtractedRequirement:
    """Assemble one chunk-local ExtractedRequirement from resolved segments."""
    status = raw.get("status", "")
    n_segments = len(segment_specs)
    chunk_segments: list = []
    n_resolved = 0
    segments_partial = False
    any_ambiguous = False
    any_start_fallback = False
    any_end_unresolved = False

    for spec, res in zip(segment_specs, segment_resolutions):
        if res.ambiguous:
            any_ambiguous = True
        if res.start_via_fallback:
            any_start_fallback = True
        if res.start_found and res.end_found and res.end_off > res.start_off:
            chunk_segments.append(ResolvedSegment(
                res.start_off, res.end_off, _normalize_role(spec.role),
            ))
            n_resolved += 1
        else:
            has_start_spec = bool(spec.start_anchor or spec.start_after_anchor)
            if has_start_spec and not res.start_found:
                segments_partial = True
            has_end_spec = bool(spec.end_before_anchor or spec.end_anchor)
            if has_end_spec and res.start_found and not res.end_found:
                segments_partial = True
                any_end_unresolved = True
            elif res.start_found and not res.end_found:
                segments_partial = True

    chunk_segments, overlap_partial = _clip_overlapping_segments(chunk_segments)
    if overlap_partial:
        segments_partial = True

    role_texts = build_role_texts(chunk_text, chunk_segments)
    requirement_text = role_texts[ROLE_REQUIREMENT]
    questionnaire_text = role_texts[ROLE_QUESTIONNAIRE]
    context_text = role_texts[ROLE_CONTEXT]
    original_text = requirement_text

    if chunk_segments:
        chunk_start = chunk_segments[0].start
        chunk_end = chunk_segments[-1].end
        doc_start = chunk_doc_offset_start + chunk_start
        doc_end = chunk_doc_offset_start + chunk_end
    else:
        chunk_start = chunk_end = -1
        doc_start = doc_end = -1
        for res in segment_resolutions:
            if res.start_found:
                chunk_start = res.start_off
                doc_start = chunk_doc_offset_start + res.start_off
                break

    first_sa = segment_specs[0].start_anchor if segment_specs else None
    last_spec = segment_specs[-1] if segment_specs else None
    last_ea = None
    if last_spec is not None:
        last_ea = last_spec.end_before_anchor or last_spec.end_anchor

    all_segments_resolved = (
        n_resolved == n_segments
        and not segments_partial
        and n_segments > 0
    )
    if n_segments == 1:
        spec = segment_specs[0]
        res = segment_resolutions[0]
        start_ok = (
            (not (spec.start_anchor or spec.start_after_anchor)) or res.start_found
        )
        has_end = bool(spec.end_before_anchor or spec.end_anchor)
        end_ok = (not has_end) or res.end_found
        order_ok = (not (res.start_found and res.end_found)) or (res.end_off > res.start_off)
        verbatim = start_ok and end_ok and order_ok
    else:
        verbatim = all_segments_resolved and bool(original_text)

    page = bbox = None
    if doc is not None and doc_start >= 0:
        loc = doc.locate(doc_start)
        page = loc.get("page")
        bbox = loc.get("bbox")

    return ExtractedRequirement(
        requirement_id=raw.get("requirement_id", ""),
        original_text=original_text,
        requirement_text=requirement_text,
        questionnaire_text=questionnaire_text,
        context_text=context_text,
        start_anchor=first_sa,
        end_anchor=last_ea,
        status=status,
        verbatim_match=verbatim,
        segments=list(chunk_segments),
        n_segments=n_segments,
        n_segments_resolved=n_resolved,
        segments_partial=segments_partial,
        segment_anchor_pairs=list(segment_specs),
        chunk_offset_start=chunk_start,
        chunk_offset_end=chunk_end,
        doc_offset_start=doc_start,
        doc_offset_end=doc_end,
        page=page,
        bbox=bbox,
        end_resolved=all_segments_resolved,
        end_anchor_unresolved=any_end_unresolved,
        start_via_fallback=any_start_fallback,
        ambiguous=any_ambiguous,
    )


# --------------------------------------------------------------------------- #
# Single-chunk extraction
# --------------------------------------------------------------------------- #
def extract_requirements_from_chunk(
    chunk_text: str,
    system_prompt: str,
    doc: Optional[DocumentExtraction] = None,
    chunk_doc_offset_start: int = 0,
    model: Optional[str] = None,
    max_output_tokens: int = 4096,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_wait_s: int = 10,
) -> ExtractionResult:
    """Locate requirement boundaries in a chunk via Anthropic tool-use, then
    resolve each anchor to chunk-local offsets (whitespace-tolerant).

    Raises:
        RuntimeError: if every model in the fallback chain is exhausted. The
        orchestrator catches this and records an explicit ``api_error`` batch.
    """
    chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

    tools = [{
        "name": ANCHOR_TOOL_NAME,
        "description": "Emit the boundary anchors of the regulatory requirements found in the provided chunk.",
        "input_schema": ANCHOR_INPUT_SCHEMA,
    }]
    system_blocks = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]
    messages = [{"role": "user", "content": chunk_text}]

    client = get_anthropic_client()
    last_exception = None
    response = None
    used_model = None
    response_time = 0.0
    attempts: list = []

    for current_model in _build_model_chain(model):
        for attempt_no in range(1, max_retries + 1):
            t0 = time.time()
            try:
                response = client.messages.create(
                    model=current_model,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    system=system_blocks,
                    tools=tools,
                    tool_choice={"type": "tool", "name": ANCHOR_TOOL_NAME},
                    messages=messages,
                )
                duration = time.time() - t0
                response_time = duration
                used_model = current_model
                attempts.append(CallAttempt(current_model, attempt_no, round(duration, 3), "ok"))
                break
            except anthropic.NotFoundError as e:
                duration = time.time() - t0
                attempts.append(CallAttempt(current_model, attempt_no, round(duration, 3),
                                            "not_found", str(e)[:200]))
                last_exception = e
                break
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                duration = time.time() - t0
                is_connection = isinstance(e, anthropic.APIConnectionError)
                status_code = getattr(e, "status_code", None)
                # Retry ONLY transient failures; non-transient 4xx fail fast.
                retryable = is_connection or status_code in {408, 409, 429, 500, 502, 503, 504, 529}
                outcome = "connection_error" if is_connection else "api_error"
                attempts.append(CallAttempt(current_model, attempt_no, round(duration, 3),
                                            outcome, str(e)[:200]))
                last_exception = e
                if retryable and attempt_no < max_retries:
                    time.sleep(retry_wait_s)
                else:
                    break
        if response is not None:
            break

    if response is None:
        raise RuntimeError(f"All models exhausted. Last error: {last_exception}")

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise RuntimeError("Model returned no tool_use block despite tool_choice forcing it.")

    payload = tool_block.input
    raw_requirements = payload.get("requirements", [])

    # Recover the recurring small-model failure where `requirements` is a single
    # JSON-formatted string instead of an array.
    if isinstance(raw_requirements, str):
        try:
            parsed = json.loads(raw_requirements)
            if isinstance(parsed, list):
                raw_requirements = parsed
        except json.JSONDecodeError:
            raw_requirements = [{"_raw_string": raw_requirements}]

    validated: list = []
    malformed: list = []
    norm_chunk, idx_map_chunk = _normalize_ws_with_map(chunk_text)

    dict_items = [r for r in raw_requirements if isinstance(r, dict)]
    normalized_segments: list = []
    valid_dict_items: list = []
    for r in dict_items:
        pairs, reason = _normalize_model_segments(r)
        if reason:
            malformed.append({"reason": reason, "requirement_id": r.get("requirement_id"), "value": r})
            normalized_segments.append([])
            valid_dict_items.append(r)
        else:
            normalized_segments.append(pairs)
            valid_dict_items.append(r)

    grouped_resolutions = _resolve_chunk_anchors(norm_chunk, idx_map_chunk, normalized_segments)

    for r, specs, seg_res in zip(valid_dict_items, normalized_segments, grouped_resolutions):
        if not specs:
            continue
        validated.append(_build_chunk_requirement_from_segments(
            r, specs, seg_res, chunk_text, chunk_doc_offset_start, doc,
        ))

    for r in raw_requirements:
        if not isinstance(r, dict):
            malformed.append({"reason": "not-a-dict", "value": r})

    usage = response.usage
    return ExtractionResult(
        requirements=validated,
        model_used=used_model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        response_time_s=round(response_time, 3),
        stop_reason=response.stop_reason,
        chunk_hash=chunk_hash[:16],
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        batch_status=_derive_batch_status(validated, malformed, response.stop_reason),
        raw_payload=payload,
        malformed_items=malformed,
        attempts=attempts,
    )


# --------------------------------------------------------------------------- #
# Dynamic chunk sizing
# --------------------------------------------------------------------------- #
CHARS_PER_TOKEN_HEURISTIC = 4   # rough average for English with the Anthropic tokenizer


def _estimate_tokens(text: str) -> int:
    """Cheap, slightly conservative heuristic (no tokenizer round-trip)."""
    return (len(text) // CHARS_PER_TOKEN_HEURISTIC) + 1


# Per-model total context windows. Unknown or local models fall back to
# DEFAULT_CONTEXT_WINDOW.
MODEL_CONTEXT_WINDOWS = {
    "claude-haiku-4-5-20251001": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
}
DEFAULT_CONTEXT_WINDOW = 200_000


def _model_context_window(model: Optional[str]) -> int:
    chain = _build_model_chain(model)
    primary = model or (chain[0] if chain else None)
    return MODEL_CONTEXT_WINDOWS.get(primary, DEFAULT_CONTEXT_WINDOW)


def compute_input_token_budget(
    system_prompt: str,
    model: Optional[str] = None,
    expected_output_tokens: int = 8000,
    safety_margin: int = 8000,
    schema_overhead_tokens: int = 1200,
    max_input_tokens_cap: Optional[int] = None,
) -> int:
    """Dynamically size the chunk-text budget from the model context window:

        available = context_window - system_prompt_tokens - schema_overhead
                    - expected_output_tokens - safety_margin

    The system prompt + tool schema are fixed per-call overhead; the result is
    the budget for the CHUNK TEXT only (the prompt is sent under cache_control).
    """
    ctx = _model_context_window(model)
    available = (
        ctx
        - _estimate_tokens(system_prompt)
        - schema_overhead_tokens
        - expected_output_tokens
        - safety_margin
    )
    available = max(available, 0)
    if max_input_tokens_cap is not None:
        available = min(available, max_input_tokens_cap)
    return available


def _page_to_first_block_idx(doc: DocumentExtraction, page_num: int) -> int:
    for i, b in enumerate(doc.blocks):
        if b.page >= page_num:
            return i
    return len(doc.blocks)


def _page_to_last_block_idx(doc: DocumentExtraction, page_num: int) -> int:
    last = -1
    for i, b in enumerate(doc.blocks):
        if b.page <= page_num:
            last = i
    return last


_BOUNDARY_PREFIX_TOLERANCE = 8


def _compute_boundary_blocks(
    doc: DocumentExtraction,
    pattern: Optional[str],
    first_block_idx: int,
    last_block_idx: int,
) -> list[int]:
    """Return sorted block indices (within [first, last]) whose text STARTS a new
    unit, i.e. block text matches ``pattern`` at its beginning. Empty list if
    ``pattern`` is None."""
    if not pattern:
        return []
    compiled = re.compile(pattern)
    boundaries: list[int] = []
    for i in range(first_block_idx, last_block_idx + 1):
        stripped = doc.blocks[i].text.lstrip()
        if not stripped:
            continue
        if compiled.match(stripped):
            boundaries.append(i)
            continue
        # Tolerate a few leading artifact characters before the unit id.
        prefix = stripped[:60]
        m = compiled.search(prefix)
        if m is not None and m.start() <= _BOUNDARY_PREFIX_TOLERANCE:
            boundaries.append(i)
    return boundaries


def _build_batch(
    doc: DocumentExtraction,
    start_block_idx: int,
    end_block_idx: int,
    target_input_tokens: int,
    boundary_blocks: Optional[list[int]] = None,
) -> Optional[tuple]:
    """Greedily extend a batch from ``start_block_idx`` while staying under
    ``target_input_tokens``. Always includes at least one block (a single block
    larger than the budget is sent whole rather than dropped).

    When ``boundary_blocks`` is provided, packs whole requirement units (delimited
    by those block indices) and never splits a unit across chunks.

    Returns (first_idx, last_idx, batch_text, batch_doc_offset_start) or None.
    """
    if start_block_idx > end_block_idx or start_block_idx >= len(doc.blocks):
        return None

    batch_doc_offset_start = doc.blocks[start_block_idx].char_start

    if not boundary_blocks:
        last_idx = start_block_idx
        upper = min(end_block_idx, len(doc.blocks) - 1)
        for i in range(start_block_idx, upper + 1):
            candidate_end = doc.blocks[i].char_end
            candidate_tokens = (
                (candidate_end - batch_doc_offset_start) // CHARS_PER_TOKEN_HEURISTIC + 1
            )
            if candidate_tokens > target_input_tokens and i > start_block_idx:
                last_idx = i - 1
                break
            last_idx = i
        batch_text = doc.text_in_range(
            batch_doc_offset_start, doc.blocks[last_idx].char_end
        )
        return (start_block_idx, last_idx, batch_text, batch_doc_offset_start)

    nexts = [
        b for b in boundary_blocks
        if b > start_block_idx and b <= end_block_idx
    ]
    first_unit_last = (nexts[0] - 1) if nexts else end_block_idx
    chosen_last = first_unit_last

    for b in nexts:
        candidate_last = b - 1
        candidate_tokens = (
            (doc.blocks[candidate_last].char_end - batch_doc_offset_start)
            // CHARS_PER_TOKEN_HEURISTIC + 1
        )
        if candidate_tokens > target_input_tokens and candidate_last > first_unit_last:
            break
        chosen_last = candidate_last

    if nexts and nexts[-1] <= end_block_idx:
        candidate_last = end_block_idx
        candidate_tokens = (
            (doc.blocks[candidate_last].char_end - batch_doc_offset_start)
            // CHARS_PER_TOKEN_HEURISTIC + 1
        )
        if candidate_tokens <= target_input_tokens:
            chosen_last = candidate_last

    last_idx = min(max(chosen_last, start_block_idx), end_block_idx)
    batch_text = doc.text_in_range(
        batch_doc_offset_start, doc.blocks[last_idx].char_end
    )
    return (start_block_idx, last_idx, batch_text, batch_doc_offset_start)


def _failed_batch_result(exc: Exception, batch_text: str,
                         first_idx: int, last_idx: int,
                         offset_start: int, offset_end: int) -> ExtractionResult:
    """Explicit ``api_error`` result so an exhausted fallback chain on one batch
    does not abort the whole run (no silent failure)."""
    return ExtractionResult(
        requirements=[],
        model_used=None,
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        response_time_s=0.0,
        stop_reason="error",
        chunk_hash=hashlib.sha256(batch_text.encode("utf-8")).hexdigest()[:16],
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        batch_first_block=first_idx,
        batch_last_block=last_idx,
        batch_doc_offset_start=offset_start,
        batch_doc_offset_end=offset_end,
        batch_status="api_error",
        attempts=[CallAttempt(model=None, attempt_no=1, duration_s=0.0,
                              outcome="api_error", error_message=str(exc)[:200])],
    )


@dataclass
class PendingRequirement:
    """A requirement opened in one chunk and not yet closed. Offsets are the
    source of truth: only the document ``start`` offset persists across chunks;
    the closing chunk supplies the end. At most one is open at a time.

    For multi-span requirements, ``segments`` holds resolved document segment
    offsets collected so far and ``segment_anchor_pairs`` holds the full model
    spec (including unresolved segments). Only one open multi-span pending is
    supported at a time."""
    requirement_id: str
    start: int
    start_anchor: Optional[str]
    start_chunk_id: int = -1
    chunk_ids: list = field(default_factory=list)
    end_anchor_unresolved: bool = False
    end_anchor: Optional[str] = None
    segments: list = field(default_factory=list)
    segment_anchor_pairs: list = field(default_factory=list)
    n_segments: int = 1
    segments_partial: bool = False


@dataclass
class RangeExtraction:
    """Output of the linear orchestrator.

    ``results`` keeps every per-chunk ExtractionResult for telemetry; the
    deliverable is ``requirements`` -- the stitched, document-level list.
    """
    results: list           # list[ExtractionResult] in batch order
    requirements: list      # list[ExtractedRequirement], stitched & document-level


def _finalize_requirement(
    doc: DocumentExtraction,
    requirement_id: str,
    start_off: int,
    end_off: int,
    status: str,
    end_resolved: bool,
    id_mismatch: bool,
    start_anchor: Optional[str],
    end_anchor: Optional[str],
    source_chunk_id: int = -1,
    source_chunk_ids: Optional[list] = None,
    end_inferred_from_next_start: bool = False,
    end_anchor_unresolved: bool = False,
    segment_offsets: Optional[list] = None,
    n_segments_expected: int = 1,
    segments_partial: bool = False,
    segment_anchor_pairs: Optional[list] = None,
    start_via_fallback: bool = False,
    ambiguous: bool = False,
) -> ExtractedRequirement:
    """Build a document-level requirement from resolved segment offsets.

    When ``segment_offsets`` is omitted, falls back to ``[(start_off, end_off)]``
    for backward-compatible single-span finalization.
    """
    if segment_offsets is None:
        if start_off >= 0 and end_off > start_off:
            segment_offsets = [ResolvedSegment(start_off, end_off, ROLE_REQUIREMENT)]
        elif start_off >= 0:
            segment_offsets = [ResolvedSegment(start_off, max(end_off, start_off), ROLE_REQUIREMENT)]
        else:
            segment_offsets = []

    segments, overlap_partial = _clip_overlapping_segments(list(segment_offsets))
    if overlap_partial:
        segments_partial = True

    role_texts = build_role_texts(doc.full_text, segments)
    requirement_text = role_texts[ROLE_REQUIREMENT]
    questionnaire_text = role_texts[ROLE_QUESTIONNAIRE]
    context_text = role_texts[ROLE_CONTEXT]
    original_text = requirement_text

    n_segments_resolved = len(segments)
    n_segments = max(n_segments_expected, n_segments_resolved, 1 if segments else 0)

    bound_start = segments[0].start if segments else start_off
    bound_end = segments[-1].end if segments else end_off

    page = bbox = None
    if bound_start >= 0:
        loc = doc.locate(bound_start)
        page = loc.get("page")
        bbox = loc.get("bbox")

    chunk_ids = source_chunk_ids if source_chunk_ids is not None else (
        [source_chunk_id] if source_chunk_id >= 0 else []
    )

    if n_segments == 1 and not segments_partial:
        verbatim_match = bool(end_resolved and bound_start >= 0 and original_text)
    else:
        verbatim_match = bool(
            not segments_partial
            and end_resolved
            and n_segments_resolved == n_segments_expected
            and n_segments_resolved > 0
            and original_text
        )

    return ExtractedRequirement(
        requirement_id=requirement_id,
        original_text=original_text,
        requirement_text=requirement_text,
        questionnaire_text=questionnaire_text,
        context_text=context_text,
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        status=status,
        verbatim_match=verbatim_match,
        segments=list(segments),
        n_segments=n_segments_expected,
        n_segments_resolved=n_segments_resolved,
        segments_partial=segments_partial,
        segment_anchor_pairs=list(segment_anchor_pairs or []),
        chunk_offset_start=-1,
        chunk_offset_end=-1,
        doc_offset_start=bound_start,
        doc_offset_end=bound_end,
        page=page,
        bbox=bbox,
        end_resolved=end_resolved,
        end_inferred_from_next_start=end_inferred_from_next_start,
        end_anchor_unresolved=end_anchor_unresolved,
        id_mismatch=id_mismatch,
        start_via_fallback=start_via_fallback,
        ambiguous=ambiguous,
        source_chunk_id=source_chunk_id,
        source_chunk_ids=chunk_ids,
    )


def _doc_segments_from_chunk_req(r: ExtractedRequirement, batch_offset_start: int) -> list:
    """Translate chunk-local segment offsets to document coordinates (with role)."""
    if r.segments:
        return [
            ResolvedSegment(
                batch_offset_start + seg.start,
                batch_offset_start + seg.end,
                seg.role,
            )
            for seg in (_coerce_resolved_segment(s) for s in r.segments)
        ]
    if r.doc_offset_start >= 0 and r.doc_offset_end > r.doc_offset_start:
        return [ResolvedSegment(r.doc_offset_start, r.doc_offset_end, ROLE_REQUIREMENT)]
    return []


def _finalize_from_chunk_requirement(
    doc: DocumentExtraction,
    r: ExtractedRequirement,
    batch_offset_start: int,
    chunk_id: int,
    source_chunk_ids: Optional[list] = None,
    end_resolved: Optional[bool] = None,
    end_inferred_from_next_start: bool = False,
    end_anchor_unresolved: Optional[bool] = None,
    id_mismatch: bool = False,
    status: Optional[str] = None,
) -> ExtractedRequirement:
    """Finalize a chunk-local requirement (single- or multi-span) at document level."""
    doc_segments = _doc_segments_from_chunk_req(r, batch_offset_start)
    if end_resolved is None:
        end_resolved = r.end_resolved
    if end_anchor_unresolved is None:
        end_anchor_unresolved = r.end_anchor_unresolved
    return _finalize_requirement(
        doc,
        r.requirement_id,
        r.doc_offset_start,
        r.doc_offset_end,
        status or r.status or "complete",
        end_resolved,
        id_mismatch,
        r.start_anchor,
        r.end_anchor,
        source_chunk_id=chunk_id,
        source_chunk_ids=source_chunk_ids or ([chunk_id] if chunk_id >= 0 else []),
        end_inferred_from_next_start=end_inferred_from_next_start,
        end_anchor_unresolved=end_anchor_unresolved,
        segment_offsets=doc_segments,
        n_segments_expected=r.n_segments,
        segments_partial=r.segments_partial,
        segment_anchor_pairs=r.segment_anchor_pairs,
        start_via_fallback=r.start_via_fallback,
        ambiguous=r.ambiguous,
    )


def _stitch_chunk(
    doc: DocumentExtraction,
    reqs: list,
    batch_offset_start: int,
    pending: Optional[PendingRequirement],
    final: list,
    range_end_offset: int,
    chunk_id: int = -1,
) -> Optional[PendingRequirement]:
    """Fold one chunk's chunk-local requirements into the linear ``final`` list,
    threading the single open ``pending`` requirement across chunk boundaries.

    ``chunk_id`` is the 1-based ordinal of the chunk being stitched (its index in
    ``RangeExtraction.results``); it is recorded on every finalized requirement
    for traceability.

    Boundary model: a requirement should provide ``end_anchor`` whenever its
    terminal text is visible. A null ``end_anchor`` may still be resolved at the
    next requirement's start when the model reports an immediate-adjacency case.
    That remains deterministic and ``end_resolved=True``, but is recorded as
    ``end_inferred_from_next_start=True`` for auditability.

    ``end_resolved=False`` is reserved for genuine uncertainty: a pending
    requirement interrupted by a new start, a requirement bounded at the range
    end because nothing follows it (the document ran out), or one whose start
    never resolved. Those are emitted (never dropped) and flagged for triage. A
    start-only item with no following requirement in its chunk is carried forward
    as ``pending`` rather than bounded at the range end, so it cannot swallow the
    rest of the document. Never silent, never a loop.
    """
    def closed_chunks(p: PendingRequirement) -> list:
        return sorted(set(p.chunk_ids + [chunk_id]))

    n = len(reqs)
    for j, r in enumerate(reqs):
        is_last = (j == n - 1)
        has_start = r.doc_offset_start >= 0
        has_end = r.doc_offset_end >= 0

        # --- Close, carry, or break an open pending requirement ---
        if pending is not None:
            is_continuation = (
                r.status == "truncated_at_start"
                or not has_start
                or r.requirement_id == pending.requirement_id
            )
            if is_continuation and has_end and r.doc_offset_end > pending.start:
                id_mismatch = bool(r.requirement_id) and r.requirement_id != pending.requirement_id
                if pending.n_segments > 1 or pending.segments:
                    merged = list(pending.segments)
                    merged.extend(_doc_segments_from_chunk_req(r, batch_offset_start))
                    merged, overlap_partial = _clip_overlapping_segments(merged)
                    partial = pending.segments_partial or overlap_partial or r.segments_partial
                    all_done = len(merged) >= pending.n_segments and not partial
                    final.append(_finalize_requirement(
                        doc, pending.requirement_id, pending.start, r.doc_offset_end,
                        "complete", all_done, id_mismatch, pending.start_anchor, r.end_anchor,
                        source_chunk_id=pending.start_chunk_id,
                        source_chunk_ids=closed_chunks(pending),
                        segment_offsets=merged,
                        n_segments_expected=pending.n_segments,
                        segments_partial=partial,
                        segment_anchor_pairs=pending.segment_anchor_pairs,
                        end_anchor_unresolved=pending.end_anchor_unresolved or r.end_anchor_unresolved,
                    ))
                else:
                    final.append(_finalize_requirement(
                        doc, pending.requirement_id, pending.start, r.doc_offset_end,
                        "complete", True, id_mismatch, pending.start_anchor, r.end_anchor,
                        source_chunk_id=pending.start_chunk_id,
                        source_chunk_ids=closed_chunks(pending),
                        end_anchor_unresolved=pending.end_anchor_unresolved,
                    ))
                pending = None
                continue
            if is_continuation and not has_end:
                if chunk_id not in pending.chunk_ids:
                    pending.chunk_ids.append(chunk_id)
                continue  # interior continuation: still no end, keep pending open
            # Pending interrupted by a NEW requirement: bounding at that next
            # start is deterministic containment, but not a confirmed terminal
            # end for the pending requirement.
            bound = r.doc_offset_start if has_start else range_end_offset
            if pending.n_segments > 1:
                # Multi-span pendings must NOT stretch the last segment to `bound`:
                # that re-swallows the intentional inter-segment gap (Suggested
                # Actions, extended About, etc.) when a later segment was left open.
                segs = list(pending.segments)
                if segs:
                    doc_start = _coerce_resolved_segment(segs[0]).start
                    doc_end = _coerce_resolved_segment(segs[-1]).end
                else:
                    doc_start = pending.start
                    doc_end = pending.start
                final.append(_finalize_requirement(
                    doc, pending.requirement_id, doc_start, doc_end,
                    "truncated_at_end", False, False, pending.start_anchor, pending.end_anchor,
                    source_chunk_id=pending.start_chunk_id,
                    source_chunk_ids=closed_chunks(pending),
                    end_anchor_unresolved=pending.end_anchor_unresolved,
                    segment_offsets=segs,
                    n_segments_expected=pending.n_segments,
                    segments_partial=True,
                    segment_anchor_pairs=pending.segment_anchor_pairs,
                ))
            elif pending.segments:
                segs = [_coerce_resolved_segment(s) for s in pending.segments]
                if segs and bound > segs[-1].start:
                    last = segs[-1]
                    segs[-1] = ResolvedSegment(last.start, bound, last.role)
                elif pending.start >= 0 and bound > pending.start:
                    segs = [ResolvedSegment(pending.start, bound, ROLE_REQUIREMENT)]
                final.append(_finalize_requirement(
                    doc, pending.requirement_id, pending.start, bound,
                    "truncated_at_end", False, False, pending.start_anchor, pending.end_anchor,
                    source_chunk_id=pending.start_chunk_id,
                    source_chunk_ids=closed_chunks(pending),
                    end_anchor_unresolved=pending.end_anchor_unresolved,
                    segment_offsets=segs,
                    n_segments_expected=pending.n_segments,
                    segments_partial=True,
                    segment_anchor_pairs=pending.segment_anchor_pairs,
                ))
            else:
                final.append(_finalize_requirement(
                    doc, pending.requirement_id, pending.start, bound,
                    "truncated_at_end", False, False, pending.start_anchor, None,
                    source_chunk_id=pending.start_chunk_id,
                    source_chunk_ids=closed_chunks(pending),
                    end_anchor_unresolved=pending.end_anchor_unresolved,
                ))
            pending = None

        # --- Open a new pending when the LAST item runs off the chunk end ---
        if is_last and r.status == "truncated_at_end" and has_start:
            doc_segs = _doc_segments_from_chunk_req(r, batch_offset_start)
            pending = PendingRequirement(
                requirement_id=r.requirement_id,
                start=r.doc_offset_start,
                start_anchor=r.start_anchor,
                start_chunk_id=chunk_id,
                chunk_ids=[chunk_id],
                end_anchor_unresolved=(bool(r.end_anchor) and not has_end) or r.end_anchor_unresolved,
                end_anchor=r.end_anchor,
                segments=doc_segs,
                segment_anchor_pairs=r.segment_anchor_pairs,
                n_segments=r.n_segments,
                segments_partial=r.segments_partial or r.n_segments_resolved < r.n_segments,
            )
            continue

        # --- Orphan continuation (chunk opened mid-requirement, no pending) ---
        if r.status == "truncated_at_start":
            start = batch_offset_start
            end = r.doc_offset_end if (has_end and r.doc_offset_end > start) else -1
            final.append(_finalize_requirement(
                doc, r.requirement_id, start, end,
                "truncated_at_start", bool(end >= 0), False, None, r.end_anchor,
                source_chunk_id=chunk_id,
            ))
            continue

        # --- Complete requirement fully inside this chunk (single- or multi-span) ---
        if has_start and has_end and r.doc_offset_end > r.doc_offset_start:
            if r.n_segments > 1:
                if r.n_segments_resolved == r.n_segments and not r.segments_partial:
                    final.append(_finalize_from_chunk_requirement(
                        doc, r, batch_offset_start, chunk_id,
                    ))
                    continue
                if not (is_last and r.status == "truncated_at_end"):
                    final.append(_finalize_from_chunk_requirement(
                        doc, r, batch_offset_start, chunk_id, end_resolved=False,
                    ))
                    continue
            elif r.segments and not r.segments_partial:
                final.append(_finalize_from_chunk_requirement(
                    doc, r, batch_offset_start, chunk_id,
                ))
                continue
            final.append(_finalize_requirement(
                doc, r.requirement_id, r.doc_offset_start, r.doc_offset_end,
                "complete", True, False, r.start_anchor, r.end_anchor,
                source_chunk_id=chunk_id,
                segment_offsets=[ResolvedSegment(
                    r.doc_offset_start, r.doc_offset_end, ROLE_REQUIREMENT,
                )],
                n_segments_expected=1,
            ))
            continue

        # --- Start resolved, no explicit end anchor. This is valid only for an
        # immediate-adjacency boundary; keep the deterministic next-start close
        # but record that the end was inferred rather than anchor-confirmed. ---
        if has_start:
            end_anchor_supplied_but_unresolved = bool(r.end_anchor) and not has_end
            next_start = None
            for k in range(j + 1, n):
                cand = reqs[k].doc_offset_start
                if cand >= 0 and cand > r.doc_offset_start:
                    next_start = cand
                    break
            if next_start is not None:
                doc_segs = _doc_segments_from_chunk_req(r, batch_offset_start)
                if doc_segs:
                    last = doc_segs[-1]
                    doc_segs[-1] = ResolvedSegment(last.start, next_start, last.role)
                elif r.doc_offset_start >= 0:
                    doc_segs = [ResolvedSegment(r.doc_offset_start, next_start, ROLE_REQUIREMENT)]
                if end_anchor_supplied_but_unresolved:
                    final.append(_finalize_requirement(
                        doc, r.requirement_id, r.doc_offset_start, next_start,
                        r.status or "complete", False, False, r.start_anchor, r.end_anchor,
                        source_chunk_id=chunk_id,
                        end_inferred_from_next_start=True,
                        end_anchor_unresolved=True,
                        segment_offsets=doc_segs,
                        n_segments_expected=r.n_segments,
                        segments_partial=True,
                        segment_anchor_pairs=r.segment_anchor_pairs,
                    ))
                else:
                    final.append(_finalize_requirement(
                        doc, r.requirement_id, r.doc_offset_start, next_start,
                        r.status or "complete", True, False, r.start_anchor, r.end_anchor,
                        source_chunk_id=chunk_id,
                        end_inferred_from_next_start=True,
                        end_anchor_unresolved=False,
                        segment_offsets=doc_segs,
                        n_segments_expected=r.n_segments,
                        segment_anchor_pairs=r.segment_anchor_pairs,
                    ))
                continue
            pending = PendingRequirement(
                requirement_id=r.requirement_id,
                start=r.doc_offset_start,
                start_anchor=r.start_anchor,
                start_chunk_id=chunk_id,
                chunk_ids=[chunk_id],
                end_anchor_unresolved=end_anchor_supplied_but_unresolved,
                end_anchor=r.end_anchor,
                segments=_doc_segments_from_chunk_req(r, batch_offset_start),
                segment_anchor_pairs=r.segment_anchor_pairs,
                n_segments=r.n_segments,
                segments_partial=r.segments_partial or end_anchor_supplied_but_unresolved,
            )
            continue

        # --- Neither bound resolved: emit flagged and empty for triage. ---
        final.append(_finalize_requirement(
            doc, r.requirement_id, r.doc_offset_start, r.doc_offset_end,
            r.status or "complete", False, False, r.start_anchor, r.end_anchor,
            source_chunk_id=chunk_id,
        ))
    return pending


# --------------------------------------------------------------------------- #
# Linear, forward-only orchestrator
# --------------------------------------------------------------------------- #
def extract_requirements_for_range(
    doc: DocumentExtraction,
    system_prompt: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    target_input_tokens: Optional[int] = None,   # None -> dynamic from context window
    expected_output_tokens: int = 8000,
    safety_margin: int = 8000,
    max_input_tokens_cap: Optional[int] = None,
    min_input_tokens: int = 600,
    shrink_factor: float = 0.5,
    max_consecutive_shrinks: int = 4,
    model: Optional[str] = None,
    max_output_tokens: Optional[int] = None,      # None -> expected_output_tokens
    temperature: float = 0.0,
    inter_call_pause_s: float = 0.5,
    max_iterations: int = 500,
    verbose: bool = True,
    requirement_boundary_pattern: Optional[str] = None,
) -> RangeExtraction:
    """Block-driven, dynamically-budgeted anchor extraction with linear,
    forward-only multi-chunk stitching.

    The cursor never moves backward. A requirement larger than one chunk is
    handled by a persisted ``PendingRequirement`` (its document start offset),
    closed when a later chunk supplies the end anchor. Output overflow
    (``stop_reason == "max_tokens"``) is the only reason a start block is
    reprocessed: the budget shrinks and the same start retries, then the cursor
    force-advances. A failed fallback chain on one batch becomes an
    ``api_error`` result and the cursor force-advances (no silent abort).

    Returns:
        RangeExtraction(results=per-chunk telemetry, requirements=stitched).
    """
    if not doc.blocks:
        return RangeExtraction(results=[], requirements=[])

    first_block_idx = 0 if start_page is None else _page_to_first_block_idx(doc, start_page)
    last_block_idx = (len(doc.blocks) - 1) if end_page is None else _page_to_last_block_idx(doc, end_page)
    if first_block_idx >= len(doc.blocks) or last_block_idx < 0 or first_block_idx > last_block_idx:
        raise ValueError(
            f"Requested page range ({start_page}-{end_page}) does not intersect "
            f"the extracted blocks (doc pages {doc.start_page}-{doc.end_page})."
        )

    boundary_blocks = _compute_boundary_blocks(
        doc, requirement_boundary_pattern, first_block_idx, last_block_idx,
    )

    dynamic = target_input_tokens is None
    if dynamic:
        target_input_tokens = compute_input_token_budget(
            system_prompt=system_prompt,
            model=model,
            expected_output_tokens=expected_output_tokens,
            safety_margin=safety_margin,
            max_input_tokens_cap=max_input_tokens_cap,
        )
    if max_output_tokens is None:
        max_output_tokens = expected_output_tokens

    cursor = first_block_idx
    initial_target = target_input_tokens
    current_target = target_input_tokens
    consecutive_shrinks = 0
    last_shrunk_cursor = -1
    range_end_offset = doc.blocks[last_block_idx].char_end
    pending: Optional[PendingRequirement] = None
    results: list = []
    requirements: list = []

    if verbose:
        ctx = _model_context_window(model)
        print(f"Chunk budget: {current_target} input tok "
              f"({'dynamic' if dynamic else 'override'}; ctx {ctx}, "
              f"prompt~{_estimate_tokens(system_prompt)}, out {max_output_tokens})")
        if boundary_blocks:
            print(f"Boundary-aware chunking: {len(boundary_blocks)} unit start(s) "
                  f"detected in blocks {first_block_idx}-{last_block_idx}")

    wall_start = time.time()
    iteration = 0

    while cursor <= last_block_idx and iteration < max_iterations:
        iteration += 1
        batch = _build_batch(
            doc, cursor, last_block_idx, current_target,
            boundary_blocks=boundary_blocks or None,
        )
        if batch is None:
            break
        first_idx, last_idx, batch_text, batch_offset_start = batch
        batch_offset_end = doc.blocks[last_idx].char_end

        if verbose:
            print(f"[iter {iteration:>3}] blocks {first_idx:>4}-{last_idx:<4} | "
                  f"pages {doc.blocks[first_idx].page:>3}-{doc.blocks[last_idx].page:<3} | "
                  f"est {_estimate_tokens(batch_text):>5} tok | target {current_target} | "
                  f"shrinks {consecutive_shrinks}", end=" ... ", flush=True)

        try:
            result = extract_requirements_from_chunk(
                chunk_text=batch_text,
                system_prompt=system_prompt,
                doc=doc,
                chunk_doc_offset_start=batch_offset_start,
                model=model,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            result = _failed_batch_result(exc, batch_text, first_idx, last_idx,
                                          batch_offset_start, batch_offset_end)
            results.append(result)
            if verbose:
                print(f"FAILED ({exc}); force-advancing past block {last_idx}")
            cursor = last_idx + 1
            current_target = initial_target
            consecutive_shrinks = 0
            last_shrunk_cursor = -1
            if inter_call_pause_s > 0:
                time.sleep(inter_call_pause_s)
            continue

        result.batch_first_block = first_idx
        result.batch_last_block = last_idx
        result.batch_doc_offset_start = batch_offset_start
        result.batch_doc_offset_end = batch_offset_end
        results.append(result)

        if verbose:
            ok = sum(1 for r in result.requirements if r.status == "complete")
            tr = sum(1 for r in result.requirements if r.status == "truncated_at_end")
            mf = len(result.malformed_items)
            print(f"{len(result.requirements)} reqs ({ok} ok, {tr} trunc) | "
                  f"stop={result.stop_reason} | {result.response_time_s}s"
                  + (f" | malformed={mf}" if mf else ""))

        # CASE 1: output overflow -> shrink batch and retry same starting block.
        # With boundary-aware chunking, shrinking cannot split a unit; after
        # max_consecutive_shrinks the cursor force-advances past the oversized unit.
        if result.stop_reason == "max_tokens":
            if last_shrunk_cursor != cursor:
                consecutive_shrinks = 0
                last_shrunk_cursor = cursor
            if consecutive_shrinks >= max_consecutive_shrinks or current_target <= min_input_tokens:
                if verbose:
                    print(f"  ! cannot shrink further; force-advancing past block {last_idx}")
                cursor = last_idx + 1
                current_target = initial_target
                consecutive_shrinks = 0
                last_shrunk_cursor = -1
                if inter_call_pause_s > 0:
                    time.sleep(inter_call_pause_s)
                continue
            current_target = max(int(current_target * shrink_factor), min_input_tokens)
            consecutive_shrinks += 1
            if verbose:
                print(f"  -> shrinking target -> {current_target} tok, retrying same start")
            if inter_call_pause_s > 0:
                time.sleep(inter_call_pause_s)
            continue

        # Successful call: reset shrink state and stitch.
        consecutive_shrinks = 0
        last_shrunk_cursor = -1
        current_target = initial_target

        before = len(requirements)
        # chunk_id == the 1-based ordinal of this result in `results` (the same
        # id used in extraction_run.json), so requirements link back to their call.
        pending = _stitch_chunk(doc, result.requirements, batch_offset_start,
                                pending, requirements, range_end_offset,
                                chunk_id=len(results))
        if verbose:
            state = "pending OPEN" if pending is not None else "no pending"
            print(f"  -> stitched {len(requirements) - before} requirement(s) | {state}")

        cursor = last_idx + 1
        if cursor > last_block_idx:
            break
        if inter_call_pause_s > 0:
            time.sleep(inter_call_pause_s)

    # Flush a requirement still open at range end (bounded + flagged).
    if pending is not None:
        if verbose:
            print(f"  ! range ended with {pending.requirement_id!r} still open; "
                  f"emitting bounded (end_resolved=False)")
        requirements.append(_finalize_requirement(
            doc, pending.requirement_id, pending.start, range_end_offset,
            "truncated_at_end", False, False, pending.start_anchor, pending.end_anchor,
            source_chunk_id=pending.start_chunk_id,
            source_chunk_ids=sorted(set(pending.chunk_ids)),
            end_anchor_unresolved=pending.end_anchor_unresolved,
            segment_offsets=(
                pending.segments
                if pending.segments
                else [ResolvedSegment(pending.start, range_end_offset, ROLE_REQUIREMENT)]
                if pending.start >= 0 else []
            ),
            n_segments_expected=pending.n_segments,
            segments_partial=True,
            segment_anchor_pairs=pending.segment_anchor_pairs,
        ))
        pending = None

    if iteration >= max_iterations:
        print(f"  ! safety cap hit at iteration {iteration} (max_iterations={max_iterations})")

    if verbose:
        total_calls = sum(len(r.attempts) for r in results)
        ok_calls = sum(1 for r in results for a in r.attempts if a.outcome == "ok")
        flagged = sum(1 for r in requirements if not r.end_resolved)
        print(f"\nDone. {len(results)} batch(es) in {time.time() - wall_start:.1f}s | "
              f"{ok_calls}/{total_calls} ok API call(s) | "
              f"{len(requirements)} requirement(s) stitched"
              + (f" ({flagged} flagged)" if flagged else "") + ".")

    return RangeExtraction(results=results, requirements=requirements)


# --------------------------------------------------------------------------- #
# Framework-level telemetry
# --------------------------------------------------------------------------- #
@dataclass
class FrameworkExtractionStats:
    """Aggregated counters across all chunks processed for one framework."""
    framework: str
    n_chunks_processed: int

    n_total_api_calls: int
    n_successful_calls: int
    n_failed_calls: int
    failures_by_outcome: dict
    calls_by_model: dict
    total_wall_time_s: float

    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_tokens: int
    total_cache_read_tokens: int

    # Output quality (counted from the stitched, document-level requirements)
    n_requirements_total: int
    n_flagged_requirements: int        # end not anchor-confirmed (end_resolved=False)
    n_id_mismatches: int

    # Per-batch explicit status counts (ok / empty / verbatim_failure / json_error / api_error)
    status_counts: dict = field(default_factory=dict)

    def print_summary(self) -> None:
        print(f"=== Framework: {self.framework or '<unset>'} ===")
        print(f"  chunks processed:        {self.n_chunks_processed}")
        print(f"  total API calls:         {self.n_total_api_calls}")
        print(f"    successful:            {self.n_successful_calls}")
        print(f"    failed:                {self.n_failed_calls}")
        if self.failures_by_outcome:
            print(f"    failures by outcome:   {dict(self.failures_by_outcome)}")
        print(f"  calls by model (ok):     {dict(self.calls_by_model)}")
        print(f"  batch status counts:     {dict(self.status_counts)}")
        print(f"  total wall time:         {round(self.total_wall_time_s, 2)}s")
        print(f"  tokens input/output:     {self.total_input_tokens} / {self.total_output_tokens}")
        print(f"  cache creation/read:     {self.total_cache_creation_tokens} / {self.total_cache_read_tokens}")
        print(f"  requirements (stitched): {self.n_requirements_total}")
        print(f"    flagged (no end):      {self.n_flagged_requirements}")
        print(f"    id mismatches:         {self.n_id_mismatches}")

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_extractions(extraction, framework: str = "") -> FrameworkExtractionStats:
    """Aggregate a ``RangeExtraction`` (or a bare list of ``ExtractionResult``)
    into framework-level stats. API/token counters come from the per-chunk
    results; requirement-quality counters come from the stitched output."""
    from collections import Counter

    if isinstance(extraction, RangeExtraction):
        results = extraction.results
        requirements = extraction.requirements
    else:
        results = list(extraction)
        requirements = [r for res in results for r in res.requirements]

    failures_by_outcome: Counter = Counter()
    calls_by_model: Counter = Counter()
    status_counts: Counter = Counter()
    total_wall = 0.0
    n_total = 0
    n_ok = 0

    for res in results:
        status_counts[res.batch_status] += 1
        for a in res.attempts:
            n_total += 1
            total_wall += a.duration_s
            if a.outcome == "ok":
                n_ok += 1
                calls_by_model[a.model] += 1
            else:
                failures_by_outcome[a.outcome] += 1

    return FrameworkExtractionStats(
        framework=framework,
        n_chunks_processed=len(results),
        n_total_api_calls=n_total,
        n_successful_calls=n_ok,
        n_failed_calls=n_total - n_ok,
        failures_by_outcome=dict(failures_by_outcome),
        calls_by_model=dict(calls_by_model),
        total_wall_time_s=round(total_wall, 3),
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        total_cache_creation_tokens=sum(r.cache_creation_tokens for r in results),
        total_cache_read_tokens=sum(r.cache_read_tokens for r in results),
        n_requirements_total=len(requirements),
        n_flagged_requirements=sum(1 for r in requirements if not r.end_resolved),
        n_id_mismatches=sum(1 for r in requirements if getattr(r, "id_mismatch", False)),
        status_counts=dict(status_counts),
    )





