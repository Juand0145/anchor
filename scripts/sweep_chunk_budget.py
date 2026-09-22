#!/usr/bin/env python3
"""Sweep target_input_tokens: offline batch plans, optional live extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _llm_ready() -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    try:
        from anchor_extract.llm_client import resolve_llm_provider
        resolve_llm_provider()
        return True
    except Exception:
        return False


def _parse_budgets(raw: str) -> list:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit("no budgets given")
    return out


def _print_table(rows: list) -> None:
    headers = [
        "budget", "n_batches", "max_batch_tok", "min_batch_tok",
        "median_tok", "total_est_user_tok", "max_units",
    ]
    print("  ".join(f"{h:>18}" for h in headers))
    for r in rows:
        print("  ".join(f"{r[h]:>18}" for h in headers))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", required=True)
    p.add_argument("--start-page", type=int, default=11)
    p.add_argument("--end-page", type=int, default=25)
    p.add_argument(
        "--profile",
        default=str(ROOT / "anchor_extract/prompts/profiles/hipaa.txt"),
    )
    p.add_argument("--boundary-pattern", default=None)
    p.add_argument("--budgets", default="3000,4000,6000,8000,12000")
    p.add_argument("--live", action="store_true")
    p.add_argument(
        "--out",
        default=None,
        help="JSON output path (default under outputs/)",
    )
    args = p.parse_args(argv)

    pdf = Path(args.pdf)
    if not pdf.is_file():
        print(f"PDF not found: {pdf}", file=sys.stderr)
        return 1

    from anchor_extract import extract_pdf
    from anchor_extract.anchor_extraction import (
        _page_to_first_block_idx,
        _page_to_last_block_idx,
    )
    from anchor_extract.batch_planning import plan_batches, summarize_batch_plan
    from anchor_extract.settings import EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN

    pattern = args.boundary_pattern or EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN
    budgets = _parse_budgets(args.budgets)
    doc = extract_pdf(str(pdf), start_page=args.start_page, end_page=args.end_page)
    first = _page_to_first_block_idx(doc, args.start_page)
    last = _page_to_last_block_idx(doc, args.end_page)

    if args.live:
        if not _llm_ready():
            print("skip: no LLM provider configured (Azure/Anthropic env)")
            return 0
        return _run_live(args, doc, pattern, budgets)

    rows = []
    payload = {"mode": "offline", "pdf": str(pdf), "pages": [args.start_page, args.end_page], "runs": []}
    for budget in budgets:
        plans = plan_batches(doc, first, last, budget, pattern)
        summary = summarize_batch_plan(plans)
        total_est = sum(pl.est_input_tokens for pl in plans)
        row = {
            "budget": budget,
            "n_batches": summary["n_batches"],
            "max_batch_tok": summary["max_est_tokens"],
            "min_batch_tok": summary["min_est_tokens"],
            "median_tok": summary["median_est_tokens"],
            "total_est_user_tok": total_est,
            "max_units": summary["max_units_per_batch"],
        }
        rows.append(row)
        payload["runs"].append({
            **row,
            "plans": [pl.to_dict() for pl in plans],
        })
    _print_table(rows)
    out = Path(args.out or ROOT / "outputs" / "batch_sweep_offline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _source_ids(requirements) -> list:
    """Identifiers as printed in the source, read from segment metadata.

    Consumer-side read for reporting only: the engine never interprets these
    keys, so coverage checks live here rather than in the pipeline.
    """
    ids = []
    for r in requirements:
        for spec in getattr(r, "segment_anchor_pairs", []) or []:
            meta = getattr(spec, "metadata", None)
            if not isinstance(meta, dict):
                continue
            value = meta.get("req_id") or meta.get("subcategory_id")
            if isinstance(value, str) and value.strip():
                ids.append(value.strip())
                break
    return ids


def _run_live(args, doc, pattern, budgets) -> int:
    from anchor_extract import extract_document

    profile = Path(args.profile).read_text(encoding="utf-8")
    payload = {
        "mode": "live",
        "pdf": args.pdf,
        "pages": [args.start_page, args.end_page],
        "runs": [],
    }
    for budget in budgets:
        extraction = extract_document(
            args.pdf,
            profile,
            doc=doc,
            start_page=args.start_page,
            end_page=args.end_page,
            requirement_boundary_pattern=pattern,
            target_input_tokens=budget,
            verbose=False,
        )
        ids = _source_ids(extraction.requirements)
        n_in = sum(res.input_tokens for res in extraction.results)
        n_out = sum(res.output_tokens for res in extraction.results)
        run = {
            "budget": budget,
            "n_calls": len(extraction.results),
            "sum_input_tokens": n_in,
            "sum_output_tokens": n_out,
            "source_ids": ids,
            "has_160_103": any(i == "160.103" or str(i).startswith("160.103") for i in ids),
        }
        payload["runs"].append(run)
        print(
            f"budget={budget} n_calls={run['n_calls']} "
            f"in={n_in} out={n_out} has_160.103={run['has_160_103']}"
        )
    out = Path(args.out or ROOT / "outputs" / "batch_sweep_live.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
