"""Offline batch plans: same cursor loop as the orchestrator, no LLM.

When a boundary pattern is set, ``plan_batches`` reflects multi-unit packing
under ``target_input_tokens`` (whole units, never split).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Optional

from .anchor_extraction import (
    _build_batch,
    _compute_boundary_blocks,
    _estimate_tokens,
    _find_section_heading_ids,
)
from .pdf_extraction import DocumentExtraction


@dataclass
class BatchPlan:
    first_block_idx: int
    last_block_idx: int
    est_input_tokens: int
    n_units: int
    page_start: int
    page_end: int
    section_ids: list = field(default_factory=list)

    def contains_section_id(self, section_id: str) -> bool:
        """True if this batch's heading scan includes ``section_id`` (e.g. 160.103)."""
        needle = (section_id or "").replace("§", "").strip()
        return needle in self.section_ids

    def to_dict(self) -> dict:
        return asdict(self)


def plan_batches(
    doc: DocumentExtraction,
    first_block_idx: int,
    last_block_idx: int,
    target_input_tokens: int,
    boundary_pattern: Optional[str] = None,
) -> list:
    """Walk ``_build_batch`` from ``first_block_idx`` through ``last_block_idx``.

    Matches the orchestrator cursor (forward-only). When
    ``boundary_pattern`` is set, whole units are packed while they fit the
    token budget.
    """
    if first_block_idx > last_block_idx or not doc.blocks:
        return []
    last_block_idx = min(last_block_idx, len(doc.blocks) - 1)
    first_block_idx = max(first_block_idx, 0)
    bounds = _compute_boundary_blocks(
        doc, boundary_pattern, first_block_idx, last_block_idx,
    )
    boundary_arg = bounds or None
    plans: list = []
    cursor = first_block_idx
    while cursor <= last_block_idx:
        batch = _build_batch(
            doc, cursor, last_block_idx, target_input_tokens,
            boundary_blocks=boundary_arg,
        )
        if batch is None:
            break
        fi, li, text, _off = batch
        n_units = (
            sum(1 for b in bounds if fi <= b <= li) if bounds else 1
        )
        section_ids = (
            _find_section_heading_ids(text, boundary_pattern)
            if boundary_pattern else []
        )
        plans.append(BatchPlan(
            first_block_idx=fi,
            last_block_idx=li,
            est_input_tokens=_estimate_tokens(text),
            n_units=n_units,
            page_start=doc.blocks[fi].page,
            page_end=doc.blocks[li].page,
            section_ids=section_ids,
        ))
        cursor = li + 1
    return plans


def summarize_batch_plan(plans: list) -> dict:
    if not plans:
        return {
            "n_batches": 0,
            "max_est_tokens": 0,
            "min_est_tokens": 0,
            "median_est_tokens": 0,
            "max_units_per_batch": 0,
        }
    toks = [p.est_input_tokens for p in plans]
    return {
        "n_batches": len(plans),
        "max_est_tokens": max(toks),
        "min_est_tokens": min(toks),
        "median_est_tokens": int(median(toks)),
        "max_units_per_batch": max(p.n_units for p in plans),
    }
