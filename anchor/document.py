"""Public document handle returned by ``anchor.read`` (phase 1)."""

from __future__ import annotations

from anchor_extract.pdf_extraction import DocumentExtraction

from .serialization import blocks as _blocks_json


class Document:
    """A source file read into the positional block model.

    Wraps one ``DocumentExtraction``. The wrapped value stays reachable through
    ``extraction`` for engine-level callers such as
    ``extract_document(..., doc=document.extraction)``.
    """

    def __init__(self, extraction: DocumentExtraction):
        self._extraction = extraction

    @property
    def extraction(self) -> DocumentExtraction:
        """The engine-level positional extraction (blocks, pages, offsets)."""
        return self._extraction

    @property
    def full_text(self) -> str:
        return self._extraction.full_text

    @property
    def pdf_path(self) -> str:
        return self._extraction.pdf_path

    @property
    def pdf_hash(self) -> str:
        return self._extraction.pdf_hash

    @property
    def parser(self) -> str:
        return self._extraction.parser

    def blocks(self, *, include_full_text: bool = False) -> dict:
        """Build the JSON-serializable block view (phase 2).

        Args:
            include_full_text: also emit the full document text stream.

        Returns:
            dict with schema_version, source, summary and blocks.
        """
        return _blocks_json(self._extraction, include_full_text=include_full_text)

    def summary(self) -> dict:
        """Provenance and cleaning counters for the read document."""
        return self._extraction.summary()
