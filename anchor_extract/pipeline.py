"""Thin convenience wrapper around the anchor extraction orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .anchor_extraction import (
    RangeExtraction,
    build_anchor_system_prompt,
    extract_requirements_for_range,
    summarize_extractions,
)
from .pdf_extraction import DocumentExtraction, extract_pdf
from .settings import CLAUDE_CONFIG, PDF_PROCESSING
from .trace_serialization import (
    build_extraction_run,
    build_requirements_doc,
    make_run_id,
)


def extract_document(
    pdf_path: str,
    detection_prompt: str,
    *,
    doc: Optional[DocumentExtraction] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    model: Optional[str] = None,
    requirement_boundary_pattern: Optional[str] = None,
    target_input_tokens: Optional[int] = None,
    verbose: bool = True,
) -> RangeExtraction:
    """Extract requirements from a PDF using a framework detection prompt."""
    if doc is None:
        doc = extract_pdf(pdf_path, start_page=start_page, end_page=end_page)
    system_prompt = build_anchor_system_prompt(detection_prompt)
    if target_input_tokens is None:
        target_input_tokens = PDF_PROCESSING.get("target_input_tokens")
    return extract_requirements_for_range(
        doc=doc,
        system_prompt=system_prompt,
        start_page=start_page,
        end_page=end_page,
        target_input_tokens=target_input_tokens,
        requirement_boundary_pattern=requirement_boundary_pattern,
        model=model,
        verbose=verbose,
    )


def to_requirements_json(
    framework_name: str,
    doc: DocumentExtraction,
    extraction: RangeExtraction,
) -> dict:
    """Build the lean requirements.json document."""
    run_id = make_run_id(doc, extraction.results)
    return build_requirements_doc(framework_name, run_id, doc, extraction)


def to_extraction_run_json(
    framework_name: str,
    doc: DocumentExtraction,
    extraction: RangeExtraction,
    *,
    model: Optional[str] = None,
    target_input_tokens: Optional[int] = None,
) -> dict:
    """Build the full extraction_run.json trace document."""
    run_id = make_run_id(doc, extraction.results)
    stats = summarize_extractions(extraction, framework=framework_name)
    model_config = {
        "model": model or CLAUDE_CONFIG.get("model"),
        "fallback_models": CLAUDE_CONFIG.get("fallback_models", []) or [],
        "temperature": CLAUDE_CONFIG.get("temperature", 0.0),
        "max_output_tokens": CLAUDE_CONFIG.get("max_tokens", 8000),
    }
    if target_input_tokens is None:
        target_input_tokens = PDF_PROCESSING.get("target_input_tokens")
    return build_extraction_run(
        framework_name,
        run_id,
        doc,
        extraction,
        stats,
        model_config,
        target_input_tokens,
    )


def save_json(obj: dict, path: str | Path) -> None:
    """Write a JSON artifact to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
