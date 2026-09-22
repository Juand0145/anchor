# Phased API

The `anchor` package exposes the pipeline as explicit phases. Each phase is
usable on its own; nothing is hidden behind a single call.

| Phase | Call | Output |
|---|---|---|
| 1 — read | `anchor.read(path, ...)` | `DocumentExtraction` (blocks + offsets + provenance) |
| 2 — blocks | `anchor.blocks(document)` | JSON-serializable dict |
| 3+ — extract | `anchor_extract.extract_document(...)` | anchors, spans, stitched units |

## Phase 1 — `read`

```python
import anchor

doc = anchor.read("pdf/hipaa-simplification-201303.pdf", start_page=11, end_page=12)
notes = anchor.read("notes.txt")  # paragraphs separated by blank lines
```

Suffix dispatch (case-insensitive):

- `.pdf` → `extract_pdf`; `start_page`, `end_page` and any extra keyword
  arguments are forwarded unchanged.
- `.txt` → plain-text ingest. `start_page`, `end_page` and PDF keyword
  arguments are accepted but **ignored**: a text file has no pages.
- anything else → `ValueError`.

Text ingest models the file as one synthetic page (`page_num=1`,
`total_pages_in_pdf=1`). Blocks are the non-empty paragraphs separated by blank
lines, normalized with the same rules as PDF blocks (hyphenation repair,
newline collapse, whitespace squeeze). `bbox` is `(0, 0, 0, 0)` because a text
file carries no geometry, and `block_no` equals `page_block_idx`. `pdf_hash` is
the sha256 of the raw file bytes, as for PDFs; `parser` is `text`.

Both sources satisfy the same invariant:

```
full_text[b.char_start:b.char_end] == b.text
```

The returned document is accepted directly by
`extract_document(..., doc=document)`.

## Phase 2 — `blocks`

```python
import json

print(json.dumps(anchor.blocks(doc), indent=2)[:500])
```

Shape:

| Key | Contents |
|---|---|
| `schema_version` | `1` |
| `source` | `{path, kind: "pdf" \| "text", hash_prefix}` (first 12 hex chars of the sha256) |
| `summary` | `DocumentExtraction.summary()` |
| `blocks` | list of `{char_start, char_end, page, bbox, text, block_no, page_block_idx}` |

`full_text` is omitted by default; pass `include_full_text=True` to add it.
`blocks()` raises `TypeError` for anything that is not a `DocumentExtraction`.

## Phase 3+ — extraction

Anchor extraction, chunking, resolution and stitching are unchanged. See
[core_pipeline.md](core_pipeline.md) for the design and the README quickstart
for `extract_document`.
