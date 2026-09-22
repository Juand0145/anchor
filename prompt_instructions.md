# Writing an extraction prompt for Anchor

How to author a document-specific extraction prompt (an **extraction profile**)
for the Anchor engine.

**The model proposes boundary anchors; the engine materializes spans.**

This file is the human-facing authoring guide. The non-negotiable generic
contract is [`anchor_extract/prompts/anchor.txt`](anchor_extract/prompts/anchor.txt),
loaded at import and prepended to your profile at run time. Pipeline design
lives in [docs/core_pipeline.md](docs/core_pipeline.md); the profile checklist and
bundled list live in [docs/profiles.md](docs/profiles.md).

## 1. Purpose: what the model does and what the engine does

| The model (LLM) | The engine (Python) |
|---|---|
| Decides what counts as a unit | Locates each anchor in the chunk text |
| Emits verbatim boundary anchors | Slices byte-exact spans, validates the offsets |
| Tags each segment with a role and metadata | Groups slices by role, stitches across chunks |
| Reports a status at chunk boundaries | Assigns `anchor_id` / `span_key` and writes trace artifacts |

The model never emits body text, never reports offsets or page numbers, and
never invents an identifier of its own. A profile that asks for body text is
overridden: the generic contract takes precedence.

## 2. What is an anchor

An anchor is a **verbatim substring of the chunk** — the cleaned PDF text
stream, not the visual page. The engine resolves it with a string search, so:

- Copy punctuation, casing, section symbols and spacing exactly.
- Never paraphrase, shorten, reflow or translate an anchor.
- Line breaks inside a PDF block are normalized to single spaces before the
  model sees the chunk; anchor the text as shown in the chunk.

## 3. Start anchor (inclusive)

`start_anchor` marks where a span begins and **is included** in the slice.

- Must be verbatim **and unique** within the chunk.
- Roughly 12–25 words: run from the heading or id line **through the first
  complete sentence** of the body.
- A bare heading is not acceptable. The same heading text recurs in tables of
  contents, indexes and cross-references, and the engine takes the first match,
  so a short heading resolves to the wrong place.
- Do not extend past the start of the next unit.
- `null` only for a `truncated_at_start` continuation.

## 4. Ending and starting a span: the three boundary fields

| Field | Semantics | Use when |
|---|---|---|
| `end_anchor` | **Inclusive** end — the anchor text is part of the slice | The unit's terminal sentence, clause or history bracket is visible |
| `end_before_anchor` | **Exclusive** end — the slice stops immediately *before* the first occurrence after the span start | The natural boundary is the next heading or labeled section |
| `start_after_anchor` | **Exclusive** start — the slice begins immediately *after* the anchor | The span should start right after a heading or lead-in phrase |

Precedence: `end_before_anchor` wins over `end_anchor`; `start_after_anchor`
wins over `start_anchor` for the slice start. `end_before_anchor` need not be
globally unique — it resolves to the first occurrence after the span start — but
it must still be verbatim.

## 5. Span vs engine unit

A **span** is one bounded interval defined by those fields. An **engine unit**
is the resolved pair of offsets the engine produces from a span.

One logical unit may be a single span or an ordered list of disjoint
`segments[]`. Text between segments is deliberately excluded. Segments must be
in document order and must not overlap; the engine clips overlaps and flags the
unit as partial.

## 6. `role`: grouping labels

`role` is a profile-defined label used to **group slices into separate
outputs**. Different roles are never concatenated.

- Roles are not business identifiers and not ids. They answer "what kind of text
  is this slice", not "which unit is this".
- Your profile declares the allowed role names for its document family.
- v0 conventions are `requirement`, `questionnaire` and `context`; v0 derives
  `original_text` from the `requirement` role only.
- Behavior note: the engine keeps any non-empty role label as emitted; a
  missing or unusable role falls back to `requirement`, since `original_text`
  depends on it.

Examples: HIPAA uses `requirement` alone, because the anchored span *is* the
regulatory text. The AI RMF Playbook uses all three, because a subcategory
interleaves a normative outcome, explanatory context and a questionnaire.

## 7. `metadata`: verbatim domain fields

`metadata` is an optional (recommended) JSON object **per segment** carrying the
domain fields your profile names — `req_id`, `subcategory_id`, `chapter`,
`source`, and so on.

- Every value is **copied verbatim from text visible in the chunk**.
- Never invent, normalize, translate or reformat a value. Omit the key instead
  of guessing.
- This is where the unit's identifier **as printed in the source** lives
  (`160.103`, `GOVERN 1.2`). The engine never reads it: metadata is copied into
  the artifacts untouched, for downstream tools and humans.
- Because nothing branches on it, a wrong or missing `req_id` cannot break
  extraction — and cannot be caught by the engine either. Reviewing metadata is
  the consumer's job.
- `metadata` is not `role` (grouping) and not an engine identifier. Do not ask
  the model to produce an `anchor_id`: engine ids belong to the engine.
- Metadata is exported: per unit (from the first segment that carries it) and
  per entry in `segments[]`.

## 8. `status`: chunk-boundary reporting

| Status | Meaning |
|---|---|
| `complete` | The start is in this chunk and the end is determinable here |
| `truncated_at_end` | **Last item of the chunk only**; body runs past the chunk edge, `end_anchor` null |
| `truncated_at_start` | Began in a previous chunk; `start_anchor` null |

Marking a middle item `truncated_at_end` is an error. When unsure about the
final item in a chunk, choose `truncated_at_end`.

## 9. How your profile composes with the generic contract

`build_anchor_system_prompt(detection_prompt)` prepends `anchor.txt` and a
separator stating that the profile decides only *which* spans are units and how
ids are formatted. Anything in your profile about output shape, `original_text`
or returning bodies is explicitly ignored.

```python
from pathlib import Path
from anchor_extract import build_anchor_system_prompt

detection_prompt = Path("anchor_extract/prompts/profiles/hipaa.txt").read_text(encoding="utf-8")
system_prompt = build_anchor_system_prompt(detection_prompt)
```

### Identifiers (schema 2.2)

The engine assigns two ids and reads none:

| Id | Owner | Source |
|---|---|---|
| `anchor_id` (exported as `requirement_id` too) | Engine | 1-based **reading order**: `"1"`, `"2"`, `"3"`, … Empty when the unit's start never resolved. The model must never emit one. |
| `span_key` | Engine | `{pdf_hash[:12]}:{doc_offset_start}` — positional key for dedup and re-run comparison, not for display. |
| `req_id`, `subcategory_id`, … | Profile | Verbatim inside segment `metadata`; exported as-is, never interpreted. |

The top-level `requirement_id` field is **deprecated and ignored** on the wire:
tell the model to omit it and put the source's identifier in `metadata`.

Cross-chunk stitching keys on **position and status** only, so a continuation
does not need to repeat any id: a unit split across chunks keeps the single
`anchor_id` assigned when its start first resolved.

## 10. Profile checklist

- [ ] States what **is** a unit and what to **skip** (front matter, tables of
      contents, references, page noise).
- [ ] Fixes **granularity**: which headings are one whole-section unit, and
      which split into separate units.
- [ ] States **completeness** rules: what must never be skipped between two
      emitted ids.
- [ ] Declares the **allowed role names** and what text each covers.
- [ ] Declares the **metadata keys**, which are required, and that values are
      verbatim.
- [ ] Ties `start_anchor` to heading **plus disambiguating body words**.
- [ ] Says which boundary field to use (`end_anchor` vs `end_before_anchor` vs
      `start_after_anchor`) for this document's layout.
- [ ] Documents the optional **unit-boundary regex**
      (`requirement_boundary_pattern`) that keeps whole units inside one chunk.
- [ ] Never asks for body text, offsets, page numbers or invented ids.

## 11. Common failures

| Failure | Consequence |
|---|---|
| Short heading-only `start_anchor` | Resolves to a table-of-contents or cross-reference hit; wrong span |
| Paraphrased or reflowed anchor | Anchor not found; the unit is rejected |
| Wrong granularity | A list item like "A health plan." becomes a standalone unit, or a whole section collapses into one |
| Missing last section of a chunk | The model stops after the first unit; the tail is silently lost |
| Middle item marked `truncated_at_end` | Stitcher waits for a continuation that never comes |
| Invented id or metadata value | Unverifiable output; breaks provenance |
| Fabricating units in a noise chunk | An empty `requirements` array is the correct answer for many chunks |

Bundled examples:
[`hipaa.txt`](anchor_extract/prompts/profiles/hipaa.txt) (single-role,
whole-section granularity) and
[`ai_rmf_playbook.txt`](anchor_extract/prompts/profiles/ai_rmf_playbook.txt)
(three roles, heading-bounded segments). Directory notes:
[`profiles/README.md`](anchor_extract/prompts/profiles/README.md).
