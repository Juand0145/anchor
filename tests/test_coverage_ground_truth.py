"""Offline coverage helpers (no LLM)."""

from __future__ import annotations

import unittest

# Minimum § headings on HIPAA Administrative Simplification pages 11–17.
EXPECTED_HIPAA_11_17 = {
    "160.102",
    "160.103",
    "160.104",
    "160.105",
    "160.201",
}


def coverage_score(extracted_ids: set, expected: set) -> dict:
    """Compare extracted requirement/section ids to an expected set."""
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
