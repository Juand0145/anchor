# Extraction profiles

An **extraction profile** tells the core engine *which* logical units to extract
from a PDF. The engine itself does not know a document type; it only enforces
the generic anchor contract.

A profile is:

1. A **detection prompt** (required) — what counts as a unit and how
   `requirement_id` is formatted.
2. An optional **unit-boundary regex** — v0 parameter
   `requirement_boundary_pattern`; keeps whole units inside one LLM chunk.
3. **Role conventions** — how to use the v0 role enum
   (`requirement` / `questionnaire` / `context`) for that document family.

## Where profiles live

Canonical directory: [`anchor_extract/prompts/profiles/`](../anchor_extract/prompts/profiles/).

| File | Domain (example) |
|---|---|
| [`ai_rmf_playbook.txt`](../anchor_extract/prompts/profiles/ai_rmf_playbook.txt) | NIST AI RMF Playbook subcategories |
| [`hipaa.txt`](../anchor_extract/prompts/profiles/hipaa.txt) | HIPAA Administrative Simplification (45 CFR 160/162/164) |

See [`profiles/README.md`](../anchor_extract/prompts/profiles/README.md) for
document type and boundary-pattern notes.

The generic contract stays at [`anchor_extract/prompts/anchor.txt`](../anchor_extract/prompts/anchor.txt)
(loaded at import; not hardcoded).

## Composition

You author **only** the profile block. Compose it with the generic contract:

```python
from pathlib import Path
from anchor_extract import build_anchor_system_prompt

detection_prompt = Path("anchor_extract/prompts/profiles/ai_rmf_playbook.txt").read_text(encoding="utf-8")
system_prompt = build_anchor_system_prompt(detection_prompt)
```

`build_anchor_system_prompt` prepends `anchor.txt` and instructs the model to
use the profile **only** to decide which spans are units and how ids are
formatted. Any leftover “return body text” instruction in a legacy prompt is
overridden. The composed string is what `extract_document` sends as the system
prompt.

Pass the same `detection_prompt` into `extract_document`. Optionally pass
`requirement_boundary_pattern` (unit-boundary regex) so `_build_batch` never
splits a unit across API calls.

## Unit-boundary regex

Concept: a Python regex that matches the **start of a new logical unit** at
(or near) the beginning of a text block.

v0 name: `requirement_boundary_pattern` on `extract_document` /
`extract_requirements_for_range`.

When set, chunking packs whole units under the token budget. When omitted, the
engine falls back to greedy block packing.

- AI RMF Playbook: `anchor_extract.settings.EXAMPLE_AI_RMF_BOUNDARY_PATTERN`
- HIPAA (CFR `§ NNN.NNN` title headings):
  `anchor_extract.settings.EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN`

For HIPAA, pass the bundled regex as `requirement_boundary_pattern`. Larger
page ranges need this so one call does not swallow many later sections with
§ 160.103.

## v0 roles

`ANCHOR_INPUT_SCHEMA` enumerates three roles: `requirement` (default),
`questionnaire`, `context`. That set is a **v0 profile convention**, not an
arbitrary user-defined taxonomy. Additional roles are future work; do not
emit values outside the enum.

A profile may use only `requirement` (single-span units) or all three
(non-contiguous units, as in the AI RMF Playbook). Different roles are never
concatenated. `original_text` is the `requirement`-role text only.

## Checklist

A good detection prompt:

- States what **is** a logical unit and what to skip (front matter, headings-only
  pages, references, noise).
- Defines `requirement_id` as a **verbatim** substring of the source; never
  invent ids.
- Does **not** ask the model to return body text, offsets, or page numbers.
- Uses `segments[]` when a unit is non-contiguous; tags each segment with a
  v0 `role`.
- Follows the generic contract: unique `start_anchor` (~12–25 words), inclusive
  `end_anchor` vs exclusive `end_before_anchor` / `start_after_anchor`, `status`
  (`complete` / `truncated_at_end` / `truncated_at_start`).
- Optionally documents a unit-boundary regex for `requirement_boundary_pattern`.

Full contract: [`anchor_extract/prompts/anchor.txt`](../anchor_extract/prompts/anchor.txt).
Pipeline internals: [core_pipeline.md](core_pipeline.md).

## Chunk-budget sweep

Offline batch plans (no LLM) and an optional live `target_input_tokens` sweep:
[testing.md](testing.md). Use this to compare `n_batches` vs missing § 160.103
when multi-unit packing is added.
