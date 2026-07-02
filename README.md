# anchor-extract

**Anchors, not bodies.** This library extracts regulatory and compliance
requirements from PDFs by asking an LLM only for verbatim boundary anchors — never
requirement text. Deterministic Python locates those anchors, slices byte-exact spans,
groups segments by role (`requirement` / `questionnaire` / `context`), stitches
multi-chunk requirements forward-only, and validates provenance. One logical unit can
be several disjoint segments linked by `requirement_id`.

## Install

```bash
cd Anchor
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -e .
# or: pip install -r requirements.txt
cp .env.example .env       # set ANTHROPIC_API_KEY
```

## Quickstart

```python
from pathlib import Path
from dotenv import load_dotenv
from anchor_extract import extract_pdf, extract_document, to_requirements_json, save_json
from anchor_extract.settings import EXAMPLE_AI_RMF_BOUNDARY_PATTERN

load_dotenv()
detection_prompt = Path("anchor_extract/prompts/examples/ai_rmf_playbook.txt").read_text(encoding="utf-8")
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

## How to write a detection prompt

You author **only** the framework-specific block: what counts as a requirement, how
`requirement_id` is formatted, and (optionally) segment roles and heading boundaries.
The generic anchor contract in `anchor_extract/prompts/anchor.txt` is fixed and
composed automatically via `build_anchor_system_prompt(detection_prompt)`.

See the example: [`anchor_extract/prompts/examples/ai_rmf_playbook.txt`](anchor_extract/prompts/examples/ai_rmf_playbook.txt).

A good detection prompt must respect these rules (summary of the generic contract):

- The model returns **anchors only** — never requirement body text.
- `start_anchor` must be a **verbatim, unique** substring (~12–25 words), usually
  heading/id through the first complete sentence.
- Prefer **segments** when a unit is non-contiguous; tag each segment with a **role**.
- **Inclusive** `end_anchor`: anchor text is included in the slice.
- **Exclusive** `end_before_anchor`: slice stops immediately *before* the first
  occurrence of a section heading after the segment start.
- **Exclusive** `start_after_anchor`: slice starts immediately *after* a heading
  (e.g. questionnaire begins after "Organizations can document the following").
- Use `status`: `complete`, `truncated_at_end` (last item in chunk only), or
  `truncated_at_start` (continuation from a prior chunk).
- Optional `requirement_boundary_pattern` (Python regex) keeps whole units inside
  one chunk so segment boundaries are not split across API calls.

## Architecture

See [docs/anchor_pipeline.md](docs/anchor_pipeline.md) for the full design: PDF
extraction → unit-aware chunking → tool-use anchor extraction → global segment
resolution → forward-only stitching → role-grouped outputs → JSON trace artifacts.

## Limitations

- Output quality depends on the model following the anchor contract and copying
  text **verbatim**; small or local models degrade anchor quality quickly.
- Requires a **text-extractable PDF** (no OCR); scanned images are unsupported.
- Anchor rules and prompts are tuned for **English** regulatory prose.
- The stitcher supports **one pending requirement** at a time (forward-only, no
  jumpback).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

This project is provided as-is and is not affiliated with any standards body or
vendor. No warranty is implied.
