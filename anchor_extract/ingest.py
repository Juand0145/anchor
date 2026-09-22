"""Plain-text ingest into the positional block model.

Produces the same ``DocumentExtraction`` contract as ``pdf_extraction`` so any
downstream consumer (chunking, anchors, resolution) works unchanged.

Invariant: for every block ``b``, ``full_text[b.char_start:b.char_end] == b.text``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .pdf_extraction import (
    DocumentExtraction,
    PageInfo,
    TextBlock,
    _normalize_block_text,
)

TEXT_PARSER = "text"
TEXT_PARSER_VERSION = "1"

# Paragraph separator: one blank line (optionally holding whitespace).
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def extract_text(text_path) -> DocumentExtraction:
    """Extract a plain-text file preserving character offsets.

    The file is modelled as a single synthetic page. Blocks are the non-empty
    paragraphs separated by blank lines, normalized with the same rules as PDF
    blocks. Bounding boxes are ``(0, 0, 0, 0)`` because a text file carries no
    geometry.

    Args:
        text_path: path to the ``.txt`` file.

    Returns:
        DocumentExtraction with full_text, blocks (with char offsets) and one page.
    """
    text_path = str(text_path)

    # sha256 of the raw bytes - same provenance contract as the PDF path.
    raw_bytes = Path(text_path).read_bytes()
    text_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Hash the raw bytes, then normalize line endings so CRLF files split and
    # reflow exactly like LF files.
    raw_text = raw_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

    parts: list = []
    blocks: list = []
    cursor = 0  # running offset in full_text; equals len(''.join(parts)) at any moment
    n_seen = 0

    for paragraph in _PARAGRAPH_SPLIT.split(raw_text):
        n_seen += 1
        block_text = _normalize_block_text(paragraph)
        if not block_text:
            continue

        # Offsets are recorded before the '\n' separator is appended, so the
        # separator belongs to no block and the invariant holds.
        block_start = cursor
        parts.append(block_text)
        cursor += len(block_text)
        block_end = cursor

        blocks.append(TextBlock(
            char_start=block_start,
            char_end=block_end,
            page=1,
            bbox=(0.0, 0.0, 0.0, 0.0),
            text=block_text,
            block_no=len(blocks),
            page_block_idx=len(blocks),
        ))

        parts.append('\n')
        cursor += 1

    # Extra '\n' closing the page, matching the PDF page-boundary marker.
    parts.append('\n')
    cursor += 1

    pages = [PageInfo(
        page_num=1,
        width=0.0,
        height=0.0,
        char_start=0,
        char_end=cursor,
        n_blocks=len(blocks),
    )]

    text_cleaning = {
        "source": "text",
        "block_split": "paragraph",
        "n_blocks_removed": n_seen - len(blocks),
        "n_blocks_seen": n_seen,
    }

    return DocumentExtraction(
        pdf_path=text_path,
        pdf_hash=text_hash,
        full_text=''.join(parts),
        blocks=blocks,
        pages=pages,
        parser=TEXT_PARSER,
        parser_version=TEXT_PARSER_VERSION,
        start_page=1,
        end_page=1,
        total_pages_in_pdf=1,
        text_cleaning=text_cleaning,
    )
