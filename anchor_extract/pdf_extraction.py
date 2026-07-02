"""Positional PDF text extraction (PyMuPDF).

Ported from the anchor pipeline prototype (``anchor.ipynb``). Provides a
character-offset-addressable view of a PDF so that any text span can be mapped
back to its page and bounding box. This replaces the layout-lossy PyPDF2 path
of the legacy ``functions/chunk.py``.

Invariant: for every block ``b``, ``full_text[b.char_start:b.char_end] == b.text``.
"""

import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from bisect import bisect_right

import pymupdf


@dataclass
class TextBlock:
    """A contiguous text region in a PDF page, with character offsets in the full document text."""
    char_start: int
    char_end: int
    page: int
    bbox: tuple
    text: str
    block_no: int
    page_block_idx: int


@dataclass
class PageInfo:
    """Metadata for a single page in the extracted document."""
    page_num: int
    width: float
    height: float
    char_start: int
    char_end: int
    n_blocks: int


@dataclass
class DocumentExtraction:
    """Positional text extraction of a PDF.

    Invariant: for every block b, full_text[b.char_start:b.char_end] == b.text
    """
    pdf_path: str
    pdf_hash: str
    full_text: str
    blocks: list
    pages: list
    parser: str
    parser_version: str
    start_page: int
    end_page: int
    total_pages_in_pdf: int

    # Cache of block starts to enable O(log n) lookup by offset
    _block_starts: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._block_starts = [b.char_start for b in self.blocks]

    def text_in_range(self, char_start: int, char_end: int) -> str:
        return self.full_text[char_start:char_end]

    def block_at_offset(self, char_offset: int) -> Optional[TextBlock]:
        if not self.blocks:
            return None
        # bisect_right - 1 returns the block whose char_start is the greatest value <= offset.
        # We still check the upper bound because offsets in the inter-block separator ('\n')
        # belong to no block.
        idx = bisect_right(self._block_starts, char_offset) - 1
        if idx < 0:
            return None
        b = self.blocks[idx]
        if b.char_start <= char_offset < b.char_end:
            return b
        return None

    def locate(self, char_offset: int) -> dict:
        b = self.block_at_offset(char_offset)
        if b is None:
            return {"page": None, "bbox": None, "block_no": None}
        return {"page": b.page, "bbox": b.bbox, "block_no": b.block_no}

    def page_text(self, page_num: int) -> str:
        for p in self.pages:
            if p.page_num == page_num:
                return self.full_text[p.char_start:p.char_end]
        return ""

    def summary(self) -> dict:
        return {
            "pdf": Path(self.pdf_path).name,
            "pdf_hash": self.pdf_hash[:12],
            "pages_processed": f"{self.start_page}-{self.end_page} of {self.total_pages_in_pdf}",
            "n_pages": len(self.pages),
            "n_blocks": len(self.blocks),
            "n_chars": len(self.full_text),
            "parser": f"{self.parser} {self.parser_version}",
        }


def _repair_hyphenation(text: str) -> str:
    # Conservative: only join end-of-line hyphenation when both sides are lowercase letters.
    # This avoids breaking legitimate hyphenated terms like "end-of-life" or proper nouns.
    return re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)


def _normalize_block_text(text: str) -> str:
    # Reflow block text: hyphenation repair must run before collapsing newlines,
    # since the hyphen pattern requires the '\n' to still be present.
    text = _repair_hyphenation(text)
    text = text.replace('\n', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _reading_order_blocks(blocks: list, page_width: float, page_height: float,
                          col_tol: float = 50.0) -> list:
    """Reorder a page's text blocks into human reading order, column-aware.

    PyMuPDF's ``sort=True`` orders blocks by (y, x), which on a multi-column page
    reads ACROSS columns row-by-row and scrambles the text (a left-column
    paragraph interleaves with right-column paragraphs at the same height),
    breaking the contiguous-span contract a requirement relies on.

    This reorders as: header band (top margin) -> body columns left-to-right,
    each top-to-bottom -> footer band (bottom margin). Columns are detected by
    clustering block left edges (``x0``). A single-column page yields one
    cluster, so the result degrades to a plain top-to-bottom sort (no behaviour
    change for single-column docs).

    Args:
        blocks: list of PyMuPDF block tuples (x0, y0, x1, y1, text, no, type).
        page_width, page_height: page dimensions in PDF points.
        col_tol: max gap between sorted x0 values that still counts as the same
            column.
    """
    if not blocks:
        return blocks

    top = 0.07 * page_height
    bot = 0.93 * page_height
    header = [b for b in blocks if b[3] < top]
    footer = [b for b in blocks if b[1] > bot]
    body = [b for b in blocks if not (b[3] < top or b[1] > bot)]

    xs = sorted(b[0] for b in body)
    clusters: list = []
    for x in xs:
        if not clusters or x - clusters[-1][-1] > col_tol:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    centers = [sum(c) / len(c) for c in clusters] or [0.0]

    def col_of(b) -> int:
        # Full-width blocks (spanning section headings) anchor to the leftmost
        # column so they sort by vertical position at the start of the flow.
        if (b[2] - b[0]) > 0.6 * page_width:
            return 0
        return min(range(len(centers)), key=lambda i: abs(b[0] - centers[i]))

    body.sort(key=lambda b: (col_of(b), b[1], b[0]))
    header.sort(key=lambda b: (b[1], b[0]))
    footer.sort(key=lambda b: (b[1], b[0]))
    return header + body + footer


def extract_pdf(pdf_path,
                start_page: Optional[int] = None,
                end_page: Optional[int] = None,
                drop_empty_blocks: bool = True,
                detect_columns: bool = True) -> DocumentExtraction:
    """Extract text from a PDF preserving character offsets, page numbers, and bounding boxes.

    Args:
        pdf_path: path to the PDF file.
        start_page: 1-based inclusive starting page. None = first page.
        end_page: 1-based inclusive last page. None = last page.
        drop_empty_blocks: skip blocks whose normalized text is empty.

    Returns:
        DocumentExtraction with full_text, blocks (with bbox + char offsets) and pages.
    """
    pdf_path = str(pdf_path)

    # sha256 of the raw bytes - serves as a stable identifier for provenance and caching.
    # If the PDF changes (even by one byte) the hash changes; if the hash matches, the
    # extraction is reproducible.
    with open(pdf_path, 'rb') as f:
        pdf_hash = hashlib.sha256(f.read()).hexdigest()

    doc = pymupdf.open(pdf_path)
    total_pages = doc.page_count

    # Convert 1-based inclusive page range to 0-based [s, e) for indexing.
    s = max((start_page - 1) if start_page else 0, 0)
    e = min(end_page if end_page else total_pages, total_pages)

    parts: list = []
    blocks: list = []
    pages: list = []
    cursor = 0  # running offset in full_text; equals len(''.join(parts)) at any moment

    for page_idx in range(s, e):
        page = doc[page_idx]
        # block tuple format from PyMuPDF: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type: 0 = text, 1 = image. We keep only text blocks.
        if detect_columns:
            # Column-aware reading order: PyMuPDF's sort=True reads ACROSS columns
            # row-by-row and scrambles multi-column pages, which breaks the
            # contiguous-span contract anchors rely on. _reading_order_blocks
            # degrades to a plain top-to-bottom sort on single-column pages.
            page_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
            page_blocks = _reading_order_blocks(
                page_blocks, page.rect.width, page.rect.height
            )
        else:
            # sort=True orders blocks by (y, x); correct for single-column docs.
            page_blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0]

        page_char_start = cursor
        n_blocks_in_page = 0

        for pb_idx, b in enumerate(page_blocks):
            x0, y0, x1, y1, raw_text, block_no, _ = b
            text = _normalize_block_text(raw_text)
            if drop_empty_blocks and not text:
                continue

            # The invariant full_text[char_start:char_end] == text is preserved
            # because we append `text` first, then a single '\n' separator AFTER
            # recording the offsets. The '\n' belongs to no block.
            block_start = cursor
            parts.append(text)
            cursor += len(text)
            block_end = cursor

            blocks.append(TextBlock(
                char_start=block_start,
                char_end=block_end,
                page=page_idx + 1,
                bbox=(round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)),
                text=text,
                block_no=block_no,
                page_block_idx=pb_idx,
            ))

            parts.append('\n')
            cursor += 1
            n_blocks_in_page += 1

        # Extra '\n' between pages produces a "\n\n" page boundary marker in full_text,
        # which is a useful anchor for downstream consumers (e.g. detecting page breaks
        # without inspecting the pages list).
        parts.append('\n')
        cursor += 1

        pages.append(PageInfo(
            page_num=page_idx + 1,
            width=round(page.rect.width, 2),
            height=round(page.rect.height, 2),
            char_start=page_char_start,
            char_end=cursor,
            n_blocks=n_blocks_in_page,
        ))

    doc.close()

    return DocumentExtraction(
        pdf_path=pdf_path,
        pdf_hash=pdf_hash,
        full_text=''.join(parts),
        blocks=blocks,
        pages=pages,
        parser="pymupdf",
        parser_version=pymupdf.__version__,
        start_page=s + 1,
        end_page=e,
        total_pages_in_pdf=total_pages,
    )
