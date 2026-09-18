# Bundled extraction profiles

Detection prompts for the core engine. Compose with
`anchor_extract/prompts/anchor.txt` via `build_anchor_system_prompt`.
Authoring guide: `docs/profiles.md`.

| Profile | File | Intended document | Unit-boundary regex |
|---|---|---|---|
| NIST AI RMF Playbook | `ai_rmf_playbook.txt` | Playbook subcategory pages (GOVERN/MAP/MEASURE/MANAGE n.n) | `anchor_extract.settings.EXAMPLE_AI_RMF_BOUNDARY_PATTERN` |
| HIPAA Administrative Simplification | `hipaa.txt` | 45 CFR Parts 160, 162, 164 (text PDF) | none bundled |

The v0 role enum (`requirement` / `questionnaire` / `context`) is a profile
convention. AI RMF uses all three; HIPAA typically uses single-span
`requirement` units.
