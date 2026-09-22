# Phased API

The `anchor` package exposes the pipeline as explicit phases. Each phase is
usable on its own; nothing is hidden behind a single call.

| Phase | Call | Output |
|---|---|---|
| 1 — read | `anchor.read(path, ...)` | `Document` (blocks + offsets + provenance) |
| 2 — blocks | `doc.blocks()` | JSON-serializable dict |
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

`read` returns a `Document`: a handle over the engine-level positional
extraction. The extraction itself stays reachable through `doc.extraction`
(blocks, pages, `full_text`, offsets), and `doc.summary()` forwards to it.
`full_text`, `pdf_path`, `pdf_hash` and `parser` are also exposed directly on
the handle.

Text ingest models the file as one synthetic page (`page_num=1`,
`total_pages_in_pdf=1`). Blocks are the non-empty paragraphs separated by blank
lines, normalized with the same rules as PDF blocks (hyphenation repair,
newline collapse, whitespace squeeze). `bbox` is `(0, 0, 0, 0)` because a text
file carries no geometry, and `block_no` equals `page_block_idx`. `pdf_hash` is
the sha256 of the raw file bytes, as for PDFs; `parser` is `text`.

Both sources satisfy the same invariant, for every block in
`doc.extraction.blocks`:

```
full_text[b.char_start:b.char_end] == b.text
```

The wrapped extraction is accepted directly by
`extract_document(..., doc=doc.extraction)`.

## Phase 2 — `blocks`

```python
import json

print(json.dumps(doc.blocks(), indent=2)[:500])
```

`anchor.blocks(doc)` is an equivalent module-level alias; it also accepts a
bare `DocumentExtraction` for callers that bypass phase 1.

Shape:

| Key | Contents |
|---|---|
| `schema_version` | `1` |
| `source` | `{path, kind: "pdf" \| "text", hash_prefix}` (first 12 hex chars of the sha256) |
| `summary` | `DocumentExtraction.summary()` |
| `blocks` | list of `{char_start, char_end, page, bbox, text, block_no, page_block_idx}` |

`full_text` is omitted by default; pass `include_full_text=True` to add it.
`anchor.blocks()` raises `TypeError` for anything that is neither a `Document`
nor a `DocumentExtraction`.

## Phase 3+ — extraction

Anchor extraction, chunking, resolution and stitching are unchanged. The
document read in phase 1 is passed through as `doc.extraction`:

```python
from pathlib import Path
from anchor_extract import extract_document
from anchor_extract.settings import EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN

detection_prompt = Path("anchor_extract/prompts/profiles/hipaa.txt").read_text(encoding="utf-8")
extraction = extract_document(
    "pdf/hipaa-simplification-201303.pdf",
    detection_prompt,
    doc=doc.extraction,
    start_page=11,
    end_page=17,
    requirement_boundary_pattern=EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN,
)
```

See [core_pipeline.md](core_pipeline.md) for the design, [profiles.md](profiles.md)
for the bundled profiles, and the README quickstart for the full HIPAA example.
