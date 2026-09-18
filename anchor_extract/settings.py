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
