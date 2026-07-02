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

# Example unit-boundary regex (NIST AI RMF Playbook subcategory ids).
EXAMPLE_AI_RMF_BOUNDARY_PATTERN = (
    r"^\s*(?:GOVERN|MAP|MEASURE|MANAGE)\s+\d+\.\d+\b"
)
