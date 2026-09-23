"""Consumer-side coverage checks. The engine never reads these metadata keys."""

from __future__ import annotations

import re

# Metadata keys a profile may use for the source's own identifier.
SOURCE_ID_KEYS = ("req_id", "subcategory_id")


def _metadata_objects(unit) -> list:
    """Unit-level metadata, then each segment's metadata. Dicts only.

    JSON units (``requirements.json``) carry ``metadata`` plus ``segments[].metadata``.
    In-memory units carry ``segment_anchor_pairs``. Keys inside those dicts are
    not interpreted by the extraction engine.
    """
    if isinstance(unit, dict):
        metas = [unit.get("metadata")]
        metas += [s.get("metadata") for s in unit.get("segments") or []]
    else:
        metas = [
            getattr(spec, "metadata", None)
            for spec in getattr(unit, "segment_anchor_pairs", []) or []
        ]
    return [m for m in metas if isinstance(m, dict)]


def source_id_for_unit(unit):
    """First ``req_id`` or ``subcategory_id`` across the unit's metadata objects.

    Scans every segment (and, for JSON units, the unit-level ``metadata``)
    until one of ``SOURCE_ID_KEYS`` is a non-empty string. Returns ``None``
    when no segment carries one. This is not ``unit_metadata``: that helper
    returns the first non-empty metadata dict, even if it has no source id.
    """
    for meta in _metadata_objects(unit):
        value = next(
            (meta[k] for k in SOURCE_ID_KEYS
             if isinstance(meta.get(k), str) and meta[k].strip()),
            None,
        )
        if value:
            return value.strip()
    return None


def source_ids(requirements: list) -> set:
    """Source identifiers of exported units, read from their ``metadata``.

    Accepts ``requirements.json`` units (dicts) or ``ExtractedRequirement``
    objects. The id is whatever the profile asked the model to copy into
    segment metadata, never the exported ``requirement_id``.
    """
    ids = set()
    for unit in requirements or []:
        value = source_id_for_unit(unit)
        if value:
            ids.add(value)
    return ids


def source_ids_in_order(requirements) -> list:
    """``source_id_for_unit`` for each unit that has one, in list order."""
    ids = []
    for unit in requirements or []:
        value = source_id_for_unit(unit)
        if value:
            ids.append(value)
    return ids


def coverage_score(extracted_ids: set, expected: set) -> dict:
    """Compare source identifiers against an expected set."""
    extracted = {str(x).strip() for x in extracted_ids}
    expected = {str(x).strip() for x in expected}
    missing = sorted(expected - extracted)
    extra = sorted(extracted - expected)
    return {
        "missing": missing,
        "extra": extra,
        "missing_count": len(missing),
        "extra_count": len(extra),
    }


def missing_section_ids(text: str, units: list, pattern: str) -> list:
    """Section headings present in ``text`` but absent from the units' metadata.

    Replaces the engine-side audit removed in schema 2.2: heading detection is
    a check on the artifacts, not a step of the extraction pipeline.
    """
    if not pattern or not text:
        return []
    compiled = re.compile(pattern)
    emitted = source_ids(units)
    found: list = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        m = compiled.match(stripped)
        if m is None:
            continue
        ident = re.sub(r"\s+", " ", re.sub(r"^§\s*", "", m.group(
            1 if m.lastindex else 0).strip())).strip()
        if ident and ident not in emitted and ident not in found:
            found.append(ident)
    return found
