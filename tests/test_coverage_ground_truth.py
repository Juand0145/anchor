"""Offline coverage helpers (no LLM).

Coverage is a consumer-side concern: the engine assigns only a reading-order
``anchor_id`` and a positional ``span_key``, so the source's own numbering must
be read from the exported ``metadata``. Helpers live in
``anchor_extract.coverage``, never in the extraction pipeline.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from anchor_extract.coverage import (
    coverage_score,
    missing_section_ids,
    source_id_for_unit,
    source_ids,
    source_ids_in_order,
)

# Minimum § headings on HIPAA Administrative Simplification pages 11–17.
EXPECTED_HIPAA_11_17 = {
    "160.102",
    "160.103",
    "160.104",
    "160.105",
    "160.201",
}


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

    def test_req_id_on_a_later_segment(self):
        """A title-only first segment must not hide a later source id."""
        unit = SimpleNamespace(segment_anchor_pairs=[
            SimpleNamespace(metadata={"title": "Applicability"}),
            SimpleNamespace(metadata={"req_id": "160.102"}),
        ])
        self.assertEqual(source_id_for_unit(unit), "160.102")
        self.assertEqual(source_ids([unit]), {"160.102"})
        self.assertEqual(source_ids_in_order([unit]), ["160.102"])


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
