# anchor-extract

**Anchors, not bodies.** Core engine for verifiable extraction of **logical units**
from text-extractable PDFs. An LLM returns only verbatim boundary anchors — never
unit body text. Deterministic Python locates those anchors, slices byte-exact
spans, optionally groups segments by role, stitches multi-chunk units forward-only,
and validates provenance.

One logical unit can be several disjoint segments linked by a record id
(`requirement_id` in the v0 API).

## Core engine vs extraction profile

| Layer | Responsibility | Where |
|---|---|---|
| **Core engine** | PDF text + provenance, unit-aware chunking, tool-use anchor extraction, resolution, stitch, JSON artifacts | `anchor_extract/` and `anchor_extract/prompts/anchor.txt` |
| **Extraction profile** | What counts as a unit, how ids are formatted, optional unit-boundary regex, role conventions | `anchor_extract/prompts/profiles/` |

The engine is domain-agnostic. A profile specializes it for one document family.
Bundled examples include the NIST AI RMF Playbook and HIPAA Administrative
Simplification. Compliance is an **example domain**, not the product definition.

v0 public names stay requirement-centric (`extract_document`, `requirement_id`,
`requirements.json`). See the mapping table in [docs/core_pipeline.md](docs/core_pipeline.md).

## Install

```bash
cd <repo-root>
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e .
# or: pip install -r requirements.txt
cp .env.example .env       # set ANTHROPIC_API_KEY
```

## Quickstart

NIST AI RMF Playbook is the primary example profile:

```python
from pathlib import Path
from dotenv import load_dotenv
from anchor_extract import extract_pdf, extract_document, to_requirements_json, save_json
from anchor_extract.settings import EXAMPLE_AI_RMF_BOUNDARY_PATTERN

load_dotenv()
detection_prompt = Path("anchor_extract/prompts/profiles/ai_rmf_playbook.txt").read_text(encoding="utf-8")
pdf_path = Path("../pdf/AI_RMF_Playbook.pdf")

doc = extract_pdf(str(pdf_path), start_page=5, end_page=10)
extraction = extract_document(
    str(pdf_path), detection_prompt, doc=doc, start_page=5, end_page=10,
    requirement_boundary_pattern=EXAMPLE_AI_RMF_BOUNDARY_PATTERN,
)

for req in extraction.requirements:
    if not req.verbatim_match:
        continue
    print(req.requirement_id, req.requirement_text[:80].strip())
    if req.n_segments == 1 and req.doc_offset_start >= 0:
        ok = doc.full_text[req.doc_offset_start:req.doc_offset_end] == req.original_text
        print("  invariant:", "ok" if ok else "BAD")

save_json(to_requirements_json("demo", doc, extraction), "outputs/requirements.json")
```

Profile file: [`anchor_extract/prompts/profiles/ai_rmf_playbook.txt`](anchor_extract/prompts/profiles/ai_rmf_playbook.txt).

## How to write an extraction profile

You author **only** the profile block: what counts as a logical unit, how
`requirement_id` is formatted, and (optionally) segment roles and heading
boundaries. The generic anchor contract in `anchor_extract/prompts/anchor.txt` is
fixed and composed automatically via `build_anchor_system_prompt(detection_prompt)`.

See [docs/profiles.md](docs/profiles.md) for the checklist and bundled examples.

A good profile must respect these rules (summary of the generic contract):

- The model returns **anchors only** — never unit body text.
- `start_anchor` must be a **verbatim, unique** substring (~12–25 words), usually
  heading/id through the first complete sentence.
- Prefer **segments** when a unit is non-contiguous; tag each segment with a **role**.
- **Inclusive** `end_anchor`: anchor text is included in the slice.
- **Exclusive** `end_before_anchor`: slice stops immediately *before* the first
  occurrence of a section heading after the segment start.
- **Exclusive** `start_after_anchor`: slice starts immediately *after* a heading.
- Use `status`: `complete`, `truncated_at_end` (last item in chunk only), or
  `truncated_at_start` (continuation from a prior chunk).
- Optional `requirement_boundary_pattern` (Python regex; concept: unit-boundary
  regex) keeps whole units inside one chunk so segment boundaries are not split
  across API calls.

## Architecture

See [docs/core_pipeline.md](docs/core_pipeline.md) for the full design: PDF
extraction → unit-aware chunking → tool-use anchor extraction → global segment
resolution → forward-only stitching → role-grouped outputs → JSON trace artifacts.

Docs index: [docs/README.md](docs/README.md). The previous path
[docs/anchor_pipeline.md](docs/anchor_pipeline.md) redirects there.

## Limitations

- Output quality depends on the model following the anchor contract and copying
  text **verbatim**; small or local models degrade anchor quality quickly.
- Requires a **text-extractable PDF** (no OCR); scanned images are unsupported.
- Profile rules depend on document language and style. Bundled default/example
  profiles are tuned for **English** regulatory prose; that is a profile limit,
  not a core-engine limit.
- The stitcher supports **one pending unit** at a time (forward-only, no
  jumpback; v0 type: `PendingRequirement`).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

This project is provided as-is and is not affiliated with any standards body or
vendor. No warranty is implied.
