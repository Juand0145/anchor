"""anchor-extract: verifiable PDF span extraction via verbatim boundary anchors.

Core engine (anchors, not bodies) plus extraction profiles that define which
logical units to extract. The stitched deliverable is ``RangeExtraction.units``
(``ExtractedUnit`` objects); the v0 requirement-centric names are kept as
deprecated aliases.
"""

from .coverage import source_id_for_unit, source_ids
from .anchor_extraction import (
    ANCHOR_INPUT_SCHEMA,
    ANCHOR_SYSTEM_PROMPT_GENERIC,
    ANCHOR_TOOL_NAME,
    ExtractedRequirement,
    ExtractedUnit,
    ExtractionResult,
    FrameworkExtractionStats,
    PendingRequirement,
    PendingUnit,
    RangeExtraction,
    ResolvedSegment,
    SegmentSpec,
    build_anchor_system_prompt,
    build_role_texts,
    extract_requirements_for_range,
    extract_requirements_from_chunk,
    generate_anchor_id,
    generate_span_key,
    summarize_extractions,
)
from .inspect import (
    get_requirement,
    inspect_requirement,
    segment_metadatas,
    unit_metadata,
)
from .pdf_extraction import DocumentExtraction, TextBlock, extract_pdf
from .pipeline import (
    extract_document,
    save_json,
    to_extraction_run_json,
    to_requirements_json,
    to_units_json,
)
from .tabular import tabular_views
from .trace_serialization import (
    build_extraction_run,
    build_requirements_doc,
    build_units_doc,
    make_run_id,
)

__version__ = "0.1.0"

__all__ = [
    "ANCHOR_INPUT_SCHEMA",
    "ANCHOR_SYSTEM_PROMPT_GENERIC",
    "ANCHOR_TOOL_NAME",
    "DocumentExtraction",
    # ExtractedRequirement / PendingRequirement are deprecated aliases of
    # ExtractedUnit / PendingUnit.
    "ExtractedRequirement",
    "ExtractedUnit",
    "ExtractionResult",
    "FrameworkExtractionStats",
    "PendingRequirement",
    "PendingUnit",
    "RangeExtraction",
    "ResolvedSegment",
    "SegmentSpec",
    "TextBlock",
    "build_anchor_system_prompt",
    "build_extraction_run",
    # build_requirements_doc is the legacy-key form of build_units_doc.
    "build_requirements_doc",
    "build_role_texts",
    "build_units_doc",
    "extract_document",
    "extract_pdf",
    "extract_requirements_for_range",
    "extract_requirements_from_chunk",
    "get_requirement",
    "inspect_requirement",
    # generate_anchor_id is a deprecated alias of generate_span_key: since
    # schema 2.1 anchor_id is the reading-order sequence, not a positional key.
    "generate_anchor_id",
    "generate_span_key",
    "make_run_id",
    "save_json",
    "segment_metadatas",
    "source_id_for_unit",
    "source_ids",
    "summarize_extractions",
    "tabular_views",
    "to_extraction_run_json",
    # to_requirements_json is the legacy-key form of to_units_json.
    "to_requirements_json",
    "to_units_json",
    "unit_metadata",
]
