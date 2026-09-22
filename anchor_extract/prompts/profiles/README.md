# Bundled extraction profiles

Detection prompts for the core engine. Compose with
`anchor_extract/prompts/anchor.txt` via `build_anchor_system_prompt`.
Authoring guide: `docs/profiles.md`.

| Profile | File | Intended document | Unit-boundary regex |
|---|---|---|---|
| HIPAA Administrative Simplification | `hipaa.txt` | 45 CFR Parts 160, 162, 164 (text PDF) | `anchor_extract.settings.EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN` |
| NIST AI RMF Playbook | `ai_rmf_playbook.txt` | Playbook subcategory pages (GOVERN/MAP/MEASURE/MANAGE n.n) | `anchor_extract.settings.EXAMPLE_AI_RMF_BOUNDARY_PATTERN` |

Pass that constant as `requirement_boundary_pattern` on `extract_document` so
each `§ NNN.NNN` title heading is one chunking unit (Definitions § 160.103 stays
in one batch; later sections are not packed into the same call merely because
the remainder fits).

The v0 role enum (`requirement` / `questionnaire` / `context`) is a profile
convention. AI RMF uses all three; HIPAA typically uses single-span
`requirement` units.
