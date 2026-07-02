"""anchor-extract: deterministic requirement extraction via verbatim boundary anchors."""

from .anchor_extraction import (
    ANCHOR_INPUT_SCHEMA,
    ANCHOR_SYSTEM_PROMPT_GENERIC,
    ANCHOR_TOOL_NAME,
    ExtractedRequirement,
    ExtractionResult,
    FrameworkExtractionStats,
    PendingRequirement,
    RangeExtraction,
    ResolvedSegment,
    SegmentSpec,
    build_anchor_system_prompt,
    build_role_texts,
    extract_requirements_for_range,
    extract_requirements_from_chunk,
    summarize_extractions,
)
from .pdf_extraction import DocumentExtraction, TextBlock, extract_pdf
from .pipeline import (
    extract_document,
    save_json,
    to_extraction_run_json,
    to_requirements_json,
)
from .trace_serialization import (
    build_extraction_run,
    build_requirements_doc,
    make_run_id,
)

__version__ = "0.1.0"

__all__ = [
    "ANCHOR_INPUT_SCHEMA",
    "ANCHOR_SYSTEM_PROMPT_GENERIC",
    "ANCHOR_TOOL_NAME",
    "DocumentExtraction",
    "ExtractedRequirement",
    "ExtractionResult",
    "FrameworkExtractionStats",
    "PendingRequirement",
    "RangeExtraction",
    "ResolvedSegment",
    "SegmentSpec",
    "TextBlock",
    "build_anchor_system_prompt",
    "build_extraction_run",
    "build_requirements_doc",
    "build_role_texts",
    "extract_document",
    "extract_pdf",
    "extract_requirements_for_range",
    "extract_requirements_from_chunk",
    "make_run_id",
    "save_json",
    "summarize_extractions",
    "to_extraction_run_json",
    "to_requirements_json",
]
