# Testing

Offline (no API keys):

```bash
python -m unittest discover -s tests -q
python scripts/sweep_chunk_budget.py \
  --pdf pdf/hipaa-simplification-201303.pdf \
  --start-page 11 --end-page 25 \
  --budgets 3000,4000,6000,12000
```

Writes `outputs/batch_sweep_offline.json`. Multi-unit packing under
`target_input_tokens`: `n_batches` **changes with budget**. HIPAA 11–25
(`total_est_user_tok` ~14307):

| budget | expected `n_batches` | notes |
|---|---|---|
| 3000 | more batches | § 160.103 (~5797 tok) is sent whole, over budget |
| 4000–6000 | mid | 160.103 isolated; not packed with 160.104 |
| 12000 | **~2–5** | `max_units_per_batch` > 1 |

A 12000-token plan must not still be 24 one-section batches.

Live sweep (skipped if Azure/Anthropic env is missing; exit 0):

```bash
python scripts/sweep_chunk_budget.py \
  --pdf pdf/hipaa-simplification-201303.pdf \
  --start-page 11 --end-page 25 \
  --live --budgets 6000,12000
```

Compare `n_calls` / `n_batches` vs `has_160_103` and `missing_section_ids` in
`outputs/batch_sweep_live.json`. Multi-unit batches can still drop later
sections inside one call; that is an LLM-completeness issue, not a packing bug.
