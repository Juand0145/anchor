# Bundled extraction profiles

Detection prompts for the core engine. Compose with
`anchor_extract/prompts/anchor.txt` via `build_anchor_system_prompt`.
Authoring guide: `../../../prompt_instructions.md` (anchors, roles, metadata);
bundled-profile notes: `docs/profiles.md`.

| Profile | File | Intended document | Unit-boundary regex |
|---|---|---|---|
| HIPAA Administrative Simplification | `hipaa.txt` | 45 CFR Parts 160, 162, 164 (text PDF) | `anchor_extract.settings.EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN` |
| NIST AI RMF Playbook | `ai_rmf_playbook.txt` | Playbook subcategory pages (GOVERN/MAP/MEASURE/MANAGE n.n) | `anchor_extract.settings.EXAMPLE_AI_RMF_BOUNDARY_PATTERN` |

Pass that constant as `requirement_boundary_pattern` on `extract_document` so
each `§ NNN.NNN` title heading is one chunking unit (Definitions § 160.103 stays
in one batch; later sections are not packed into the same call merely because
the remainder fits).

Role labels are profile-defined; each profile declares the set it allows. The v0
conventions are `requirement` / `questionnaire` / `context`, and a segment with
no role is treated as `requirement` (v0 derives `original_text` from that role).
AI RMF uses all three; HIPAA uses single-role `requirement` units.

Each profile also names its own segment `metadata` keys — `req_id` for HIPAA,
`subcategory_id` for AI RMF — whose values are copied verbatim from the chunk.
