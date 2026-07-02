"""Minimal Anthropic client helpers for anchor extraction."""

import os
from typing import List

import anthropic

from .settings import CLAUDE_CONFIG

_client = None


def _build_model_chain(primary: str = None) -> List[str]:
    """Return the ordered list of models to try: primary first, then fallbacks."""
    primary = primary or CLAUDE_CONFIG.get("model")
    chain = [primary] if primary else []
    for m in CLAUDE_CONFIG.get("fallback_models", []) or []:
        if m and m not in chain:
            chain.append(m)
    return chain


def get_anthropic_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    _client = anthropic.Anthropic(api_key=api_key)
    return _client
