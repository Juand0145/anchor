# Docs

| Doc | Contents |
|---|---|
| [phases.md](phases.md) | Phased public API: `anchor.read` (PDF/text ingest) and `Document.blocks` (JSON) |
| [README — Inspecting a unit by anchor_id](../README.md#inspecting-a-unit-by-anchor_id) | `get_requirement`, `inspect_requirement`, `unit_metadata`, `segment_metadatas` |
| [README — Tabular views](../README.md#tabular-views-notebook--pandas) | `tabular_views` / `RangeExtraction.tabular`: `segments`, `units`, `anchors`, `calls` rows |
| [core_pipeline.md](core_pipeline.md) | Core engine: pipeline, data model, resolution, stitch, JSON artifacts, `units` vs legacy `requirements` naming |
| [profiles.md](profiles.md) | Extraction profiles: how to author a detection prompt, checklist, bundled examples |
| [testing.md](testing.md) | Unittest discover and chunk-budget sweep (offline / optional live) |
| [anchor_pipeline.md](anchor_pipeline.md) | Stub redirect (legacy path) |

Bundled profile files live in [`anchor_extract/prompts/profiles/`](../anchor_extract/prompts/profiles/README.md).
