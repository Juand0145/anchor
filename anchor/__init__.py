"""anchor: phased public API over the anchor-extract core engine.

Phase 1 ``read`` ingests a PDF or plain-text file into the positional block
model; phase 2 ``blocks`` renders that document as JSON. Extraction (phase 3+)
stays in ``anchor_extract.extract_document``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from anchor_extract.ingest import extract_text
from anchor_extract.pdf_extraction import DocumentExtraction, TextBlock, extract_pdf

from .serialization import BLOCKS_SCHEMA_VERSION, blocks

__all__ = [
    "BLOCKS_SCHEMA_VERSION",
    "DocumentExtraction",
    "TextBlock",
    "blocks",
    "read",
]


def read(path: Union[str, Path],
         *,
         start_page: Optional[int] = None,
         end_page: Optional[int] = None,
         **pdf_kwargs) -> DocumentExtraction:
    """Read a source file into the positional block model.

    Args:
        path: ``.pdf`` or ``.txt`` file (suffix is case-insensitive).
        start_page: 1-based inclusive first page. PDF only; ignored for text.
        end_page: 1-based inclusive last page. PDF only; ignored for text.
        **pdf_kwargs: forwarded to ``extract_pdf``; ignored for text.

    Returns:
        DocumentExtraction usable as ``extract_document(..., doc=...)``.

    Raises:
        ValueError: the suffix is neither ``.pdf`` nor ``.txt``.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, start_page=start_page, end_page=end_page, **pdf_kwargs)
    if suffix == ".txt":
        return extract_text(path)
    raise ValueError(
        f"unsupported source '{path}': expected a .pdf or .txt file, got '{suffix or 'no suffix'}'"
    )
