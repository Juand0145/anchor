"""Environment-driven defaults for anchor-extract."""

import os

CLAUDE_CONFIG = {
    "model": os.environ.get("ANCHOR_MODEL", "claude-haiku-4-5-20251001"),
    "max_tokens": int(os.environ.get("ANCHOR_MAX_TOKENS", "8000")),
    "temperature": float(os.environ.get("ANCHOR_TEMPERATURE", "0")),
    "max_retries": int(os.environ.get("ANCHOR_MAX_RETRIES", "5")),
    "retry_wait": int(os.environ.get("ANCHOR_RETRY_WAIT", "20")),
    "fallback_models": [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
    ],
}

PDF_PROCESSING = {
    "target_input_tokens": int(os.environ.get("ANCHOR_TARGET_INPUT_TOKENS", "12000")),
}

# anthropic | azure | auto (auto: Azure if endpoint+key+deployment set, else Anthropic)
LLM_PROVIDER = (os.environ.get("ANCHOR_LLM_PROVIDER") or "auto").strip().lower()

_azure_max_raw = os.environ.get("ANCHOR_AZURE_MAX_TOKENS")
AZURE_OPENAI_CONFIG = {
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", "") or "",
    "api_key": os.environ.get("AZURE_OPENAI_API_KEY", "") or "",
    "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "") or "",
    "deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "") or "",
    "max_tokens": int(
        _azure_max_raw if _azure_max_raw else os.environ.get("ANCHOR_MAX_TOKENS", "8000")
    ),
}

# Example unit-boundary regex (NIST AI RMF Playbook subcategory ids).
EXAMPLE_AI_RMF_BOUNDARY_PATTERN = (
    r"^\s*(?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+\b"
)

# HIPAA CFR section heading at block start (after lstrip, prefix tolerance 8).
# Matches "§ 160.103 Definitions." — requires a title word so wrapped
# cross-references like "§ 164.501 of this subchapter" are not boundaries.
# Capture group 1 is the section number (160.103).
# HIPAA CFR section heading at block start (after lstrip, prefix tolerance 8).
# Matches "§ 160.103 Definitions." — requires a title word so wrapped
# cross-references like "§ 164.501 of this subchapter" are not boundaries.
# Capture group 1 is the section number (160.103).
EXAMPLE_HIPAA_SECTION_BOUNDARY_PATTERN = (
    r"^\s*§\s*(\d+\.\d+)\s+[A-Z]"
)

# Standalone page-number blocks (normalized text).
GENERIC_DROP_BLOCK_PATTERNS = [
    r"^\d{1,4}$",
]

# Running header/date as separate PyMuPDF blocks (also duplicated in body).
EXAMPLE_HIPAA_DROP_BLOCK_PATTERNS = [
    r"^HIPAA Administrative Simplification Regulation Text(?:\s+March 2013)?$",
    r"^March 2013$",
]

_drop_margins_raw = (os.environ.get("ANCHOR_PDF_DROP_MARGINS") or "1").strip().lower()
PDF_TEXT_CLEANING = {
    "drop_margin_blocks": _drop_margins_raw not in {"0", "false", "no"},
    "generic_drop_block_patterns": list(GENERIC_DROP_BLOCK_PATTERNS),
    "drop_block_patterns": (
        list(GENERIC_DROP_BLOCK_PATTERNS) + list(EXAMPLE_HIPAA_DROP_BLOCK_PATTERNS)
    ),
}
