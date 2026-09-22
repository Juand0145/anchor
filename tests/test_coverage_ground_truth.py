"""Offline coverage helpers (no LLM).

Coverage is a consumer-side concern: the engine assigns only a reading-order
``anchor_id`` and a positional ``span_key``, so the source's own numbering must
be read from the exported ``metadata``. These helpers live here, never in the
pipeline.
"""

from __future__ import annotations

import re
import unittest

# Minimum § headings on HIPAA Administrative Simplification pages 11–17.
EXPECTED_HIPAA_11_17 = {
    "160.102",
    "160.103",
    "160.104",
    "160.105",
    "160.201",
}

# Metadata keys a profile may use for the source's own identifier.
SOURCE_ID_KEYS = ("req_id", "subcategory_id")


def source_ids(requirements: list) -> set:
    """Source identifiers of exported units, read from their ``metadata``.

    Accepts ``requirements.json`` units (dicts) or ``ExtractedRequirement``
    objects; in both cases the id is whatever the profile asked the model to
    copy into the segment metadata, never the exported ``requirement_id``.
    """
    ids = set()
    for unit in requirements or []:
        for meta in _metadata_objects(unit):
            value = next(
                (meta[k] for k in SOURCE_ID_KEYS
                 if isinstance(meta.get(k), str) and meta[k].strip()),
                None,
            )
            if value:
                ids.add(value.strip())
                break
    return ids


def _metadata_objects(unit) -> list:
    if isinstance(unit, dict):
        metas = [unit.get("metadata")]
        metas += [s.get("metadata") for s in unit.get("segments") or []]
    else:
        metas = [
            getattr(spec, "metadata", None)
            for spec in getattr(unit, "segment_anchor_pairs", []) or []
        ]
    return [m for m in metas if isinstance(m, dict)]


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


class TestCoverageScore(unittest.TestCase):
    def test_missing_160_103(self):
        got = coverage_score({"160.102", "160.104"}, EXPECTED_HIPAA_11_17)
        self.assertIn("160.103", got["missing"])
        self.assertGreaterEqual(got["missing_count"], 1)
        self.assertNotIn("160.102", got["missing"])

    def test_full_expected_no_missing(self):
        got = coverage_score(EXPECTED_HIPAA_11_17, EXPECTED_HIPAA_11_17)
        self.assertEqual(got["missing"], [])
        self.assertEqual(got["missing_count"], 0)
        self.assertEqual(got["extra_count"], 0)

    def test_extra_ids(self):
        got = coverage_score(EXPECTED_HIPAA_11_17 | {"999.999"}, EXPECTED_HIPAA_11_17)
        self.assertEqual(got["extra"], ["999.999"])
        self.assertEqual(got["extra_count"], 1)


class TestSourceIds(unittest.TestCase):
    def test_reads_unit_metadata_from_json_shape(self):
        units = [
            {"requirement_id": "1", "metadata": {"req_id": "160.102"}},
            {"requirement_id": "2", "metadata": {"subcategory_id": "GOVERN 1.1"}},
        ]
        self.assertEqual(source_ids(units), {"160.102", "GOVERN 1.1"})

    def test_falls_back_to_segment_metadata(self):
        units = [{"requirement_id": "1", "metadata": {},
                  "segments": [{"metadata": {"req_id": "160.103"}}]}]
        self.assertEqual(source_ids(units), {"160.103"})

    def test_ignores_display_ids(self):
        self.assertEqual(source_ids([{"requirement_id": "1", "metadata": {}}]), set())


class TestMissingSectionIds(unittest.TestCase):
    PATTERN = r"^\s*§\s*(\d+\.\d+)\s+[A-Z]"

    def test_missing_160_103(self):
        text = "§ 160.102 Applicability.\nbody\n§ 160.103 Definitions.\n"
        missing = missing_section_ids(
            text, [{"metadata": {"req_id": "160.102"}}], self.PATTERN,
        )
        self.assertEqual(missing, ["160.103"])

    def test_emitted_headings_not_missing(self):
        text = "§ 160.102 Applicability.\n§ 160.103 Definitions.\n"
        units = [{"metadata": {"req_id": "160.102"}}, {"metadata": {"req_id": "160.103"}}]
        self.assertEqual(missing_section_ids(text, units, self.PATTERN), [])

    def test_no_pattern_empty(self):
        self.assertEqual(missing_section_ids("§ 160.103 Definitions.", [], ""), [])


if __name__ == "__main__":
    unittest.main()
