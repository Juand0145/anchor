"""JSON-serializable views of a read document (phase 2)."""

from __future__ import annotations

from pathlib import Path

from anchor_extract.ingest import TEXT_PARSER
from anchor_extract.pdf_extraction import DocumentExtraction

BLOCKS_SCHEMA_VERSION = 1


def source_kind(document: DocumentExtraction) -> str:
    """"text" for plain-text ingest, "pdf" otherwise."""
    return "text" if document.parser == TEXT_PARSER else "pdf"


def blocks(document: DocumentExtraction, *, include_full_text: bool = False) -> dict:
    """Build the JSON-serializable block view of a document.

    Args:
        document: result of ``anchor.read``.
        include_full_text: also emit the full document text stream.

    Returns:
        dict with schema_version, source, summary and blocks.
    """
    if not isinstance(document, DocumentExtraction):
        raise TypeError(
            f"blocks() expects a DocumentExtraction, got {type(document).__name__}"
        )

    doc_json = {
        "schema_version": BLOCKS_SCHEMA_VERSION,
        "source": {
            "path": str(Path(document.pdf_path)),
            "kind": source_kind(document),
            "hash_prefix": document.pdf_hash[:12],
        },
        "summary": document.summary(),
        "blocks": [
            {
                "char_start": b.char_start,
                "char_end": b.char_end,
                "page": b.page,
                "bbox": list(b.bbox),
                "text": b.text,
                "block_no": b.block_no,
                "page_block_idx": b.page_block_idx,
            }
            for b in document.blocks
        ],
    }
    if include_full_text:
        doc_json["full_text"] = document.full_text
    return doc_json
