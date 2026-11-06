"""Shared paths for the tests that need the bundled sample PDF.

Not a ``test_*`` module, so unittest discovery ignores it. Tests that use
``HIPAA_PDF`` guard on ``HIPAA_PDF.is_file()`` so the suite stays green when
the PDF is absent.
"""

from __future__ import annotations

from pathlib import Path

HIPAA_PDF = Path(__file__).resolve().parents[1] / "pdf" / "hipaa-simplification-201303.pdf"
