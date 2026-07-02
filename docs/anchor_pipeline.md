# Anchor Pipeline — Technical Design

Reference for the anchor-based requirement extraction pipeline. It extracts
regulatory/compliance requirements from a PDF as discrete, fully traceable text
spans. A requirement may be a single contiguous span or an ordered set of
**disjoint segments** (multi-span), and each segment carries a **role**
(`requirement` / `questionnaire` / `context`) so one logical unit can be split
into separate, linked outputs.

**Implementation (source of truth):**

| Concern | File |
|---|---|
| Pipeline (PDF → blocks → chunks → anchors → resolution → stitch → validate) | `anchor_extract/anchor_extraction.py` |
| PDF text extraction + reading order | `anchor_extract/pdf_extraction.py` |
| LLM client + model fallback chain | `anchor_extract/llm_client.py` |
| Settings (model defaults, PDF budget) | `anchor_extract/settings.py` |
| Generic anchor contract (prompt) | `anchor_extract/prompts/anchor.txt` (loaded at import; **not** hardcoded) |
| Example detection prompt (AI RMF Playbook) | `anchor_extract/prompts/examples/ai_rmf_playbook.txt` |
| Convenience wrapper + JSON helpers | `anchor_extract/pipeline.py` |
| Trace artifacts (`extraction_run.json`, `requirements.json`) | `anchor_extract/trace_serialization.py` |

`notebooks/anchor_demo.ipynb` is a walkthrough; the package modules are authoritative.

---

## 1. System Overview

### Problem

Compliance frameworks (HIPAA, ISO 27002, PCI, NIST, NIST AI RMF, …) ship as PDFs
whose requirements must be extracted as individual units, each linked back to its
exact source location. Two failure modes dominate naive LLM extraction:

- **Boundary errors.** Fixed page/token windows cut requirements in half and
  overlap-based stitching produces divergent duplicates.
- **Loss of provenance and fidelity.** When the model emits the requirement
  body, it paraphrases, drops clauses, and cannot reliably report character
  offsets, so the output is not verifiable against the source.

A third structural need appears in richer documents (e.g. the AI RMF Playbook): a
single logical unit interleaves the **normative outcome**, **explanatory
context** ("About", "Suggested Actions"), and a **questionnaire** ("Transparency
& Documentation" questions). These parts are *non-contiguous* — the questionnaire
sits after context we intentionally exclude from the requirement — yet they must
stay linked by `requirement_id`.

### Approach: anchors, not bodies

The LLM never returns requirement text. For each requirement it returns only
**anchors** — short verbatim substrings marking segment boundaries — plus an id,
a status, and (optionally) a per-segment `role`. Deterministic Python locates
those anchors in the source text, slices each segment, groups slices by role, and
emits typed outputs. The model does semantic work (what is a requirement, where
does each part begin/end, what role each part plays); Python does everything
mechanical (offsets, slicing, validation, provenance, stitching).

### Division of responsibility

| Concern | Owner |
|---|---|
| Identify requirements, assign ids, choose segment anchors + roles, classify boundary status | **LLM** |
| Locate anchors, compute offsets, slice text, group by role, resolve page/bbox, validate invariants, stitch across chunks | **Python** |

### Invariants

1. `full_text[block.char_start:block.char_end] == block.text` for every block.
2. For every cleanly resolved **single-span** requirement,
   `full_text[doc_offset_start:doc_offset_end] == original_text`.
3. For a **multi-span** requirement the contiguous invariant does NOT hold;
   instead each resolved segment satisfies `full_text[seg.start:seg.end]` and the
   role text is `SEGMENT_SEPARATOR.join(segment slices of that role)`.
   `doc_offset_start`/`doc_offset_end` are the *bounding* span (first segment
   start → last segment end), not a clean slice.
4. The LLM never produces an offset, page number, or body text.
5. Extraction is deterministic given the same `full_text` and anchors
   (`temperature=0`; anchor resolution is pure string search).

---

## 2. Architecture

```
PDF
 |
 v
[1] Text extraction with provenance      extract_pdf -> DocumentExtraction
 |      (PyMuPDF blocks; char offsets, page, bbox; sha256)
 v
[2] Block normalization                   _normalize_block_text, _reading_order_blocks
 |      (hyphenation repair, reflow, column-aware reading order)
 v
[3] Unit-aware dynamic chunking           _compute_boundary_blocks, _build_batch
 |      (block-aligned batches sized to budget; never split a requirement unit)
 v
[4] Anchor extraction (LLM)               extract_requirements_from_chunk
 |      (tool-use; per-requirement segments[] with role + inclusive/exclusive anchors)
 v
[5] Segment resolution                     _resolve_chunk_anchors (global ordering)
 |      (normalize ws + punctuation; exact match w/ forward cursor;
 |       prefix/suffix fallback; inclusive + exclusive boundaries -> doc offsets)
 v
[6] Requirement reconstruction             extract_requirements_for_range, _stitch_chunk
 |      (linear forward-only stitch; multi-span pending carry-forward)
 v
[7] Validation, roles & provenance         _finalize_requirement, build_role_texts
 |      (slice per segment, group by role, verify invariant, resolve page/bbox, flag)
 v
RangeExtraction { results (telemetry), requirements (deliverable) }
 |
 v
[8] Serialization                          build_extraction_run, build_requirements_doc
        (extraction_run.json + requirements.json via pipeline helpers)
```

**[1] Text extraction.** `extract_pdf` reads the PDF with PyMuPDF, keeps text
blocks only, and concatenates their normalized text into `full_text` while
recording each block's character range, page, and bounding box. The PDF's
sha256 is stored for provenance.

**[2] Block normalization.** Each block is reflowed: end-of-line hyphenation is
repaired conservatively, wrap newlines collapse to spaces, runs of whitespace
collapse to one. Blocks are emitted in **column-aware reading order** so a
multi-column page does not interleave text from different columns.

**[3] Unit-aware dynamic chunking.** A token budget bounds the chunk text. When a
framework declares a `requirement_boundary_pattern`, `_compute_boundary_blocks`
marks the block indices where a new unit begins, and `_build_batch` packs *whole
units* — a single requirement unit is never split across chunks. Without a
pattern it falls back to greedy block packing. See §5.

**[4] Anchor extraction.** Each batch is sent to the LLM with tool-use forcing
the anchor schema. Per requirement the model returns
`{requirement_id, status}` plus either legacy top-level `start_anchor`/`end_anchor`
(single span) or an ordered `segments[]` array, each segment carrying
`start_anchor`, an inclusive `end_anchor` or exclusive `end_before_anchor`, an
optional exclusive `start_after_anchor`, and a `role`.

**[5] Segment resolution.** All segments of all requirements in a chunk are
resolved together (`_resolve_chunk_anchors`) under one global forward ordering
cursor, so repeated anchor text is disambiguated by document order. Matching
tolerates whitespace and typographic punctuation differences; inclusive and
exclusive boundaries are handled distinctly. See §4.

**[6] Reconstruction.** The orchestrator walks chunks forward only, stitching
requirements whose segments fall in different chunks via a single `pending`
state that carries resolved segments plus the still-open segment spec.

**[7] Validation, roles & provenance.** Each resolved segment is sliced from
`full_text`; slices are grouped by role into `requirement_text` /
`questionnaire_text` / `context_text` (`build_role_texts`). `original_text` is the
`requirement`-role text only. The offset invariant is checked for single-span
requirements; page/bbox of the bounding start are attached. Anything that does
not resolve cleanly is emitted but flagged.

**[8] Serialization.** `requirements.json` (lean deliverable, intact per-role
text + resolved segments) and `extraction_run.json` (full per-call / per-anchor
trace) are built by `anchor_extract.trace_serialization` and written via
`pipeline.to_requirements_json`, `pipeline.to_extraction_run_json`, and
`pipeline.save_json`. See §7.

---

## 3. Core Data Model

### `TextBlock`
A contiguous text region of one page.

| Field | Purpose |
|---|---|
| `char_start`, `char_end` | Range in `full_text`; `full_text[start:end] == text` |
| `page` | 1-based page number |
| `bbox` | `(x0, y0, x1, y1)` in PyMuPDF coordinates |
| `text` | Normalized block text |
| `block_no`, `page_block_idx` | PyMuPDF block index / sequential index of kept blocks |

*Invariant:* the char range maps back to `text` exactly.

### `PageInfo`
Per-page metadata: `page_num`, `width`, `height`, `char_start`, `char_end`,
`n_blocks`. The page owns the offset range `[char_start, char_end)` in
`full_text`.

### `DocumentExtraction`
The single source of truth for everything downstream.

| Field | Purpose |
|---|---|
| `full_text` | All normalized blocks joined by `\n` (`\n\n` between pages) |
| `blocks`, `pages` | Ordered `TextBlock` / `PageInfo` lists |
| `pdf_path`, `pdf_hash` | Source path and sha256 of raw bytes |
| `parser`, `parser_version`, `start_page`, `end_page`, `total_pages_in_pdf` | Provenance |

Methods: `text_in_range`, `block_at_offset` (O(log n) via a cached
`_block_starts` + bisect), `locate` (→ `{page, bbox, block_no}`), `page_text`,
`summary`.

*Invariant:* block char ranges are contiguous and non-overlapping; the
inter-block `\n` separators belong to no block.

### `SegmentSpec`
One segment boundary spec as the model expressed it (before resolution). Carried
into the trace and the pending state.

| Field | Purpose |
|---|---|
| `start_anchor` | Verbatim substring where the segment begins (inclusive). Null for a `truncated_at_start` continuation |
| `end_anchor` | Verbatim substring where the segment ends, **inclusive** (anchor text kept) |
| `end_before_anchor` | Exclusive end: segment stops immediately BEFORE the first occurrence of this anchor after the segment start; the anchor text is NOT included. Takes precedence over `end_anchor` |
| `start_after_anchor` | Exclusive start: segment begins immediately AFTER this anchor; the anchor text is NOT included. Wins over `start_anchor` for the slice start |
| `role` | `requirement` (default) / `questionnaire` / `context` |

### `ResolvedSegment`
One resolved span, `(start, end, role)`, in chunk-local coordinates before
stitching and document coordinates after. The provenance unit for multi-span
requirements.

### `ExtractedRequirement`
One requirement, after resolution. Used both for per-chunk detections and for the
final stitched output.

| Field | Purpose |
|---|---|
| `requirement_id` | Id as it appears verbatim in the source |
| `start_anchor`, `end_anchor` | First segment start / last segment end anchor (for display/trace); either may be null |
| `status` | `complete` / `truncated_at_end` / `truncated_at_start` |
| `original_text` | The `requirement`-role text only. **Never** the model's words. Empty when not resolvable |
| `requirement_text`, `questionnaire_text`, `context_text` | Per-role joined slices (`SEGMENT_SEPARATOR` between multiple slices of a role) |
| `segments` | Resolved `ResolvedSegment` list (doc coords after stitching). Provenance source of truth for multi-span |
| `n_segments` | Number of segments the model specified |
| `n_segments_resolved` | Number of segments that resolved cleanly |
| `segments_partial` | True ⇒ at least one specified segment did not fully resolve |
| `segment_anchor_pairs` | The full `SegmentSpec` list (incl. unresolved), for trace + pending carry-forward |
| `doc_offset_start`, `doc_offset_end` | **Bounding** span (first seg start → last seg end); `-1` when unresolved. Not a clean slice when `n_segments > 1` |
| `chunk_offset_start`, `chunk_offset_end` | Bounding span within the chunk (`-1` after stitching) |
| `page`, `bbox` | Physical location of the bounding start |
| `verbatim_match` | True iff the boundaries required by `status`/segments resolved and a non-empty slice exists |
| `end_resolved` | False ⇒ the end is only a containment bound, not a confirmed terminal end |
| `end_inferred_from_next_start` | True ⇒ end anchor was null and the stitcher closed at the next resolved start (valid for immediate adjacency; visible for audit) |
| `end_anchor_unresolved` | True ⇒ the model supplied an `end_anchor` but it did not resolve after the start (end is not anchor-confirmed) |
| `id_mismatch` | True ⇒ a continuation closed a pending requirement whose id differed |
| `start_via_fallback` | True ⇒ a start resolved via the strict prefix fallback (offset identical to exact match; text unaffected) |
| `ambiguous` | True ⇒ a start could not be resolved deterministically (text recurs within the ordering window); left unresolved on purpose |
| `source_chunk_id`, `source_chunk_ids` | Start chunk ordinal, and every chunk that contributed text (`len > 1` ⇒ stitched across chunks) |

*Invariant:* for a single-span requirement with `verbatim_match=True`,
`full_text[doc_offset_start:doc_offset_end] == original_text`. For multi-span,
per-segment slices hold instead.

### `ExtractionResult`
Output of one LLM call: `requirements` (chunk-local) plus provenance/telemetry —
`model_used`, token counts (input/output/cache), `response_time_s`,
`stop_reason`, `chunk_hash`, `timestamp`, `batch_status`
(`ok`/`empty`/`overflow`/`verbatim_failure`/`json_error`/`api_error`),
`raw_payload`, `malformed_items`, batch position (`batch_first_block` …
`batch_doc_offset_end`), and `attempts: list[CallAttempt]`. `stop_reason` is the
signal that drives the output-overflow shrink.

### `CallAttempt`
One record per API attempt (success, retry, or fallback): `model`,
`attempt_no`, `duration_s`, `outcome` (`ok`/`not_found`/`api_error`/
`connection_error`), `error_message`. Enables exact "how many calls did this
run cost?" accounting.

### `PendingRequirement`
A transient record the orchestrator holds while a requirement spans chunks.
Offsets are the source of truth: `start` (document offset) and, for multi-span,
`segments` (resolved doc segments so far) + `segment_anchor_pairs` (full spec) +
`n_segments` + `segments_partial`. Also carries `start_anchor`, `end_anchor`,
`end_anchor_unresolved`, `start_chunk_id`, `chunk_ids`. At most one pending
requirement exists at a time.

### `RangeExtraction`
The orchestrator's return value: `results` (per-chunk `ExtractionResult`s for
telemetry) and `requirements` (the stitched, document-level deliverable).

---

## 4. Anchor-Based Extraction Logic

### Roles and multi-span

A requirement is an ordered list of segments (a single-span requirement is just
one segment). Each segment has a `role`; segments are grouped by role and joined
with `SEGMENT_SEPARATOR` (`"\n\n"`) into `requirement_text`, `questionnaire_text`,
and `context_text` (`build_role_texts`). Different roles are NEVER concatenated
together. Text **between** segments is intentionally excluded — that is the whole
point of multi-span: skip "Suggested Actions"/extended context while still
capturing the questionnaire that follows it.

`original_text` is always the `requirement`-role text only (the primary
normative deliverable for that unit).

### Anchor kinds

| Anchor | Meaning | Slice effect |
|---|---|---|
| `start_anchor` | Inclusive start | slice starts at the anchor's first char |
| `end_anchor` | Inclusive end | slice ends after the anchor's last char (anchor kept) |
| `start_after_anchor` | Exclusive start | slice starts right AFTER the anchor (anchor dropped) |
| `end_before_anchor` | Exclusive end | slice ends right BEFORE the anchor (anchor dropped) |

Precedence: if both start forms are given, `start_after_anchor` wins for the slice
start; if both end forms are given, `end_before_anchor` wins. A `start_anchor`
spans the heading/id line through the first complete sentence (~12–25 words) so it
is unique — a bare heading is rejected because it recurs in tables of contents and
cross-references and resolution is first-match. An inclusive `end_anchor` is the
complete final sentence/clause/regulatory-history bracket through terminal
punctuation. Exclusive `end_before_anchor` is used when the natural boundary is
the *start of the next labeled section/heading* (e.g. stop before "Transparency &
Documentation"); it need not be globally unique — it resolves to the first
occurrence after the segment start within the ordering window.

### Resolution (`_resolve_chunk_anchors`)

Anchor text from the LLM rarely matches the PDF byte-for-byte. Resolution applies
layered, **deterministic** tolerance — never fuzzy matching, because a false
positive (wrong location) is worse than a failed extraction. All segments across
all requirements in the chunk are flattened and resolved together in three passes
sharing one global forward cursor, so `previous_start < this_start < next_start`
holds globally.

**Normalization (`_normalize_ws_with_map` + `_fold_confusables`).** A normalized
copy of the chunk is built with an index map back to raw offsets. Two classes of
difference are absorbed, both 1:1 / length-preserving so the map stays exact:

- **Whitespace.** A PDF newline inside what the model sees as one phrase
  (`"§ 160.103\nDefinitions"` vs `"§ 160.103 Definitions"`); runs of whitespace
  collapse to one space.
- **Typographic punctuation.** Curly quotes/apostrophes and en/em dashes
  (`business’s`, `“…”`, `data–processing`) fold to canonical ASCII on **both**
  sides of the match (and in the anchor). Recovered text is still sliced from the
  raw source, so the **output preserves the original PDF characters**.

**Pass 1 — starts, exact match, forward cursor.** For each segment in flat order:
if it uses `start_after_anchor`, the anchor is found exactly from the cursor and
the slice start is set to the anchor's *raw end* (exclusive); otherwise the
`start_anchor` is found exactly and the slice start is its *raw start*. The cursor
advances just past each resolved start. Long, context-rich anchors are essentially
unique, so this resolves the majority.

**Pass 2 — ordering-bounded strict prefix fallback** (`_find_anchor_unique_prefix`).
For a `start_anchor` that did not match exactly (and does not use
`start_after_anchor`), the anchor is shortened word by word from the END and
searched within `[nearest resolved previous start, nearest resolved next start)`.
A candidate is accepted **only if exactly one occurrence exists in that window**;
if the longest matching prefix is ambiguous (≥2 in-window matches) the segment is
left **unresolved and marked `ambiguous`** rather than guessed. A floor
(`min_words=4` / `min_chars=24`) blocks degenerate short matches. Prefix fallback
applies to **starts only** (a prefix shares the anchor's start offset, so recovery
is loss-free).

**Pass 3 — ends.** For each segment, the search window is `[segment start, next
resolved start)`. If `end_before_anchor` is set (exclusive), it is found exactly
in the window and the slice end is the anchor's *raw start* (upper bound uses the
next segment's `start_norm_end` so a shared heading is reachable). Otherwise the
`end_anchor` (inclusive) is found exactly, then via the strict **suffix** fallback
(`_find_anchor_unique_suffix`, shortens from the FRONT, ends only, loss-free); the
slice end is the anchor's *raw end*.

Starts and ends resolve **independently**: a missing or unmatched anchor yields a
partial segment for that side only. Resolved segments are sorted and
overlap-clipped (`_clip_overlapping_segments`); any clip sets `segments_partial`.

> `_resolve_anchors` (single-requirement, no ordering context) is retained as a
> primitive; the production path uses `_resolve_chunk_anchors`.

### Why offsets are the source of truth

Anchor text is only a *locator*, valid within the chunk that produced it. Once
resolved, the requirement is defined by its segment document offsets. The pending
state persists resolved **offsets** (and the open segment spec), never re-scanning
prior chunks and never navigating backward.

### Multi-chunk requirements

If a requirement's segments span chunks, the start chunk yields resolved
segment(s) + an open spec (status `truncated_at_end`) and a later chunk supplies
the remaining segment ends (status `truncated_at_start` or matching id). The
orchestrator merges the resolved segments and finalizes. See §6.

---

## 5. Chunk Processing Strategy

### Budget

```
budget = context_window
       - system_prompt_tokens
       - schema_overhead_tokens
       - expected_output_tokens
       - safety_margin
budget = min(budget, max_input_tokens_cap)   # optional practical ceiling
```

`compute_input_token_budget` computes this when `target_input_tokens` is `None`
(dynamic). In practice `anchor_extract.settings.PDF_PROCESSING` supplies a
`target_input_tokens` override (default `12_000`), so the budget is usually a
fixed value, not derived from the context window. The system prompt is sent under
`cache_control` and counted separately, so it is not double-subtracted from the
usable batch.

### Token estimate

`_estimate_tokens(text) = len(text)//4 + 1` — a deliberately cheap heuristic
that avoids a tokenizer round-trip per iteration. It is slightly conservative;
`safety_margin` and `max_input_tokens_cap` guard against real-token overflow.

### Unit-aware batch construction (`_build_batch` + `_compute_boundary_blocks`)

The key change over naive greedy chunking: a requirement **unit** must never be
split across chunks, because a segment's `end_before_anchor` (or a later role
segment) may sit a few blocks below its start — if the cut lands between them the
start resolves in one chunk and the end in another, and the segment is dropped in
the cross-chunk merge.

- `_compute_boundary_blocks(doc, pattern, first, last)` returns the block indices
  whose text *starts a new unit*, i.e. matches the framework's
  `requirement_boundary_pattern` at (or within `_BOUNDARY_PREFIX_TOLERANCE = 8`
  chars of) its beginning. Empty when no pattern is configured.
- With boundaries, `_build_batch` packs **whole units**: it extends from the
  cursor to just before the next boundary, then keeps adding whole units while the
  estimate stays under budget, and never ends a batch mid-unit. Always includes at
  least one unit (an oversized single unit is sent whole rather than dropped).
- Without a pattern, `_build_batch` falls back to greedy consecutive-block
  packing under the budget (at least one block always included).

Example pattern (NIST AI RMF Playbook, `anchor_extract/settings.py` —
`EXAMPLE_AI_RMF_BOUNDARY_PATTERN`):

```
r"^\s*(?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+\b"
```

This keeps each subcategory (e.g. `GOVERN 4.2` with its About / Suggested Actions
/ Transparency & Documentation) inside one chunk.

### Linear processing

The cursor moves **forward only**. Each block is read at most once; the only
reason a starting block is reprocessed is output overflow. Because multi-chunk
requirements are handled by persisted state rather than re-reading, the full
budget can be used per chunk — it only shrinks when the *output* would overflow.

---

## 6. Truncation Recovery

A single `pending` slot threads an open requirement across chunk boundaries. It
carries the document start offset and, for multi-span, the resolved segments so
far plus the full segment spec. The orchestrator is forward-only; there is no
jumpback.

### State machine

```
        no pending
            |
            |  last item of chunk is truncated_at_end (start resolved)
            v
        PENDING (start offset + resolved segments + open spec)
            |                              \
            | continuation with resolved    \  new unrelated requirement starts,
            | end (segments completed)        \ or range ends, before a clean end
            v                                   v
        CLOSED (complete)                 FLUSHED (bounded, end_resolved=False)
```

### Cases

| Case | Signal | Action |
|---|---|---|
| **Complete in one chunk (single-span)** | start + explicit end resolved | Emit `full_text[start:end]`, `end_resolved=True` |
| **Complete in one chunk (multi-span)** | all segments resolved, not partial | Emit per-role text from segments, `end_resolved=True` |
| **Immediate adjacency in one chunk** | start resolved, end null, next start resolves | Emit bounded at `next_start`, `end_resolved=True`, `end_inferred_from_next_start=True` |
| **Truncated at chunk end** | last item, `truncated_at_end`, start resolved | Open `pending` (carry resolved segments + spec); advance |
| **Continues / ends in next chunk** | continuation with a resolved end after `pending.start` | Merge segments; finalize; clear `pending` |
| **Ends at document end** | range exhausted with `pending` open | Flush: emit resolved segments bounded at range end, `end_resolved=False` |
| **Interrupted (single-span)** | new unrelated requirement starts before `pending` closes | Bound at the new start, `end_resolved=False`; process the new one |
| **Interrupted (multi-span)** | same, but pending has multiple segments | **Emit only the already-resolved segments** (do NOT stretch the last segment to the interrupt point), `segments_partial=True`, `end_resolved=False` |
| **Mid-chunk start-only** | non-final item, start but no usable end | Bound at next detected start (or range end), `end_resolved=False` |
| **Output overflow** | `stop_reason == "max_tokens"` | Shrink budget by `shrink_factor` and retry the same start, up to `max_consecutive_shrinks`; then force-advance |

**Multi-span safety net (why the interrupted case is special).** If a later
segment of a multi-span requirement is left open and a new requirement then
interrupts the pending, stretching the last resolved segment to the interrupt
point would re-swallow the intentional inter-segment gap (Suggested Actions,
extended About, References). So the stitcher emits only the segments that actually
resolved, marks `segments_partial=True` / `end_resolved=False`, and moves on.

The failure policy is uniform: anything that cannot be closed by a confirmed end
is still emitted, bounded deterministically, and flagged for triage — never
silently dropped and never looped on. `max_iterations` is a final runaway guard.

---

## 7. Validation, Roles, and Provenance

`_finalize_requirement` builds every document-level requirement from resolved
segment offsets (falling back to `[(start, end)]` for single-span):

1. **Clip.** Segments are sorted and overlap-clipped; a clip flags
   `segments_partial`.
2. **Role text.** `build_role_texts` slices each segment from `full_text` and
   joins by role → `requirement_text` / `questionnaire_text` / `context_text`.
   `original_text = requirement_text`.
3. **Bounding span.** `doc_offset_start` = first segment start,
   `doc_offset_end` = last segment end. For single-span this is a clean slice; for
   multi-span it only bounds the unit.
4. **`verbatim_match`.** Single-span: end resolved + start present + non-empty
   text. Multi-span: not partial + end resolved + `n_segments_resolved ==
   n_segments_expected` + non-empty text.
5. **Page / bbox.** `doc.locate(bounding start)` maps to the containing block's
   page and bounding box.
6. **Flags.** `end_resolved`, `end_inferred_from_next_start`,
   `end_anchor_unresolved`, `id_mismatch`, `start_via_fallback`, `ambiguous`,
   `n_segments` / `n_segments_resolved` / `segments_partial` make every non-ideal
   or implicit outcome observable.

Run-level telemetry is aggregated by `summarize_extractions` into
`FrameworkExtractionStats`: API calls (total/ok/failed, by outcome, by model),
wall time, token totals, per-batch `status_counts`, and requirement quality
(total, flagged `end_resolved=False`, id mismatches).

### Output artifacts

This package writes **JSON only**. Use `anchor_extract.pipeline`:

- `to_requirements_json(framework_name, doc, extraction)` → lean deliverable
- `to_extraction_run_json(framework_name, doc, extraction, ...)` → full trace
- `save_json(obj, path)` → write either artifact to disk

Default output location in the demo notebook is `outputs/requirements.json`.

- **`requirements.json`** — lean deliverable: per requirement `uid`,
  `requirement_id`, `title`, `extracted_text` (= `requirement_text`), the three
  role texts (`requirement_text`, `questionnaire_text`, `context_text`), `source`
  (doc offset + page range), `anchors` (incl. `segment_anchor_pairs`),
  `segments[]` (each with `doc_offset`, `page_range`, `role`, `resolved`),
  `trace` (chunk ids, `spans_chunks`), and `validation` (`status`,
  `end_resolved`, `end_inferred_from_next_start`, `end_anchor_unresolved`,
  `verbatim_match`, `n_segments`, `n_segments_resolved`, `segments_partial`,
  `warnings`).
- **`extraction_run.json`** — full trace: `document`, `model_config`, `totals`,
  and per-chunk `chunks[]` with the call result and per-anchor `resolution`
  (`start_found`, `end_found`, `chunk_offset`, `doc_offset`, resolved `segments`
  with role, `n_segments`, `n_segments_resolved`, `segments_partial`,
  `verbatim_match`, `resolved_via_prefix_fallback`, `ambiguous`).

Both JSON files carry a controlled-vocabulary `warnings` list for triage:
`start_unresolved`, `end_unresolved`, `id_mismatch`, `empty_text`,
`ambiguous_anchor`, `resolved_via_prefix_fallback`, `segments_partial`,
`multi_span`, `spans_multiple_chunks`.

**Out of scope (reference implementation in the source application).** The parent
preprocessor repo additionally emits Excel workbooks (`requirements.xlsx`,
`questionnaire.xlsx`, `context.xlsx`, `blocks.xlsx`) and feeds an atomizer
stage. Those writers are not part of this package.

---

## 8. Error Handling and Edge Cases

| Case | Behavior |
|---|---|
| **Missing anchor** | The unresolved side yields a partial segment; the requirement is recovered by a deterministic bound where possible and flagged `end_resolved=False`, or emitted empty if nothing resolves. Never dropped. |
| **Null end with immediate adjacency** | If the next requirement start resolves in the same chunk, the stitcher closes at `next_start` and records `end_inferred_from_next_start=True`. |
| **`end_anchor` supplied but unresolved** | `end_anchor_unresolved=True`; the end is treated as unconfirmed (bounded by next-start inference or carried as pending), never trusted as a terminal end. |
| **Ambiguous / duplicate anchor** | Global forward cursor + the `previous_start < current < next_start` window disambiguate most repeats. If still not unique, the segment is left unresolved and marked `ambiguous` — never guessed. |
| **Exclusive-boundary segments** | `start_after_anchor` / `end_before_anchor` drop the anchor text from the slice; `end_before_anchor` need not be globally unique (first occurrence after the start within the window). |
| **Non-contiguous roles** | Multi-span with per-segment `role` captures a questionnaire that follows excluded context, linked by `requirement_id` (roles never concatenated). |
| **Whitespace / punctuation differences** | Absorbed by whitespace + confusable-punctuation folding (curly quotes/apostrophes, en/em dashes). |
| **Substituted / dropped word in an anchor** | Exact match fails; prefix (starts) or suffix (ends) fallback recovers it **iff** the surviving prefix/suffix is unique in the window, else flagged `ambiguous`. Never a crash. |
| **Scrambled PDF reading order** | `_reading_order_blocks` clusters blocks into columns by left edge and reads each top-to-bottom, header/footer bands pulled out. Single-column docs are unaffected. |
| **Large requirement (> 1 chunk)** | Stitched across chunks via `pending` (§6). Unit-aware chunking (§5) keeps a whole unit in one chunk when a `requirement_boundary_pattern` is set. |
| **Multi-span pending interrupted** | Only resolved segments are emitted; the last segment is NOT stretched to the interrupt point (safety net, §6). Flagged `segments_partial` / `end_resolved=False`. |
| **Output overflow** | Input batch is shrunk and retried; on repeated failure the cursor force-advances with a warning. |
| **Malformed LLM output** | `requirements` returned as a JSON string is parsed and recovered; non-dict items and missing anchors are routed to `malformed_items`; the raw payload is preserved. The pipeline does not crash. |
| **API errors / model not provisioned** | Per-model retries with backoff (transient 4xx/5xx only), then fallback down `CLAUDE_CONFIG["fallback_models"]`. Every attempt recorded in `attempts`; an exhausted chain becomes an `api_error` batch and the cursor force-advances. |
| **Framework-specific rules** | The anchor contract is framework-agnostic and lives in `anchor_extract/prompts/anchor.txt` (loaded at import via `_load_anchor_system_prompt`; raises if missing/empty). A per-framework detection prompt decides only *what* is a requirement and how ids are formatted; `build_anchor_system_prompt` composes the two and overrides any legacy "return body text" instruction. |

---

## 9. Implementation Map

Production code lives in `anchor_extract/`. `notebooks/anchor_demo.ipynb` mirrors
it for validation but is not authoritative.

| Stage | Symbols | Location |
|---|---|---|
| Data model | `TextBlock`, `PageInfo`, `DocumentExtraction`, `SegmentSpec`, `ResolvedSegment`, `ExtractedRequirement`, `PendingRequirement`, `RangeExtraction` | `anchor_extract/anchor_extraction.py` |
| Roles | `ROLE_REQUIREMENT`/`ROLE_QUESTIONNAIRE`/`ROLE_CONTEXT`, `SEGMENT_SEPARATOR`, `_normalize_role`, `build_role_texts`, `_join_role_slices` | `anchor_extract/anchor_extraction.py` |
| Extraction + normalization + reading order | `extract_pdf`, `_normalize_block_text`, `_reading_order_blocks` | `anchor_extract/pdf_extraction.py` |
| LLM client / model fallback chain | `get_anthropic_client`, `_build_model_chain` | `anchor_extract/llm_client.py` |
| Settings | `CLAUDE_CONFIG`, `PDF_PROCESSING`, `EXAMPLE_AI_RMF_BOUNDARY_PATTERN` | `anchor_extract/settings.py` |
| Generic anchor contract + schema | `anchor.txt`, `_load_anchor_system_prompt`, `ANCHOR_INPUT_SCHEMA`, `build_anchor_system_prompt` | `anchor_extract/prompts/anchor.txt`, `anchor_extract/anchor_extraction.py` |
| Normalization + folding | `_normalize_ws_with_map`, `_fold_confusables`, `_CONFUSABLE_CHARS` | `anchor_extract/anchor_extraction.py` |
| Anchor primitives | `_find_anchor_raw`, `_find_anchor_unique_prefix`, `_find_anchor_unique_suffix` | `anchor_extract/anchor_extraction.py` |
| Segment resolution | `_normalize_model_segments`, `_SegmentResolution`, `_resolve_chunk_anchors`, `_build_chunk_requirement_from_segments`, `_clip_overlapping_segments`, `_resolve_anchors` | `anchor_extract/anchor_extraction.py` |
| Chunk extraction | `extract_requirements_from_chunk`, `_derive_batch_status` | `anchor_extract/anchor_extraction.py` |
| Unit-aware chunking + budget | `_estimate_tokens`, `compute_input_token_budget`, `_compute_boundary_blocks`, `_build_batch` | `anchor_extract/anchor_extraction.py` |
| Orchestrator + stitch + finalize | `extract_requirements_for_range`, `_stitch_chunk`, `_finalize_requirement`, `_finalize_from_chunk_requirement`, `_doc_segments_from_chunk_req` | `anchor_extract/anchor_extraction.py` |
| Telemetry | `summarize_extractions`, `FrameworkExtractionStats` | `anchor_extract/anchor_extraction.py` |
| Pipeline wrapper + JSON helpers | `extract_document`, `to_requirements_json`, `to_extraction_run_json`, `save_json` | `anchor_extract/pipeline.py` |
| Trace serialization | `build_extraction_run`, `build_requirements_doc`, `_requirement_warnings`, `_serialize_segment_specs`, `_serialize_resolved_segments` | `anchor_extract/trace_serialization.py` |

### Dependencies
`pymupdf >= 1.27`, `anthropic >= 0.100`, `python-dotenv`, `pandas`.
`ANTHROPIC_API_KEY` must be set in the environment.

### Determinism notes
`temperature=0`; anchor resolution is pure string search; offsets, slicing, and
role grouping are computed in Python. Given the same `full_text` and the same
anchors, the output is reproducible. The only non-deterministic element is the
LLM's choice of anchors and roles; every downstream step is deterministic and
verifiable.
