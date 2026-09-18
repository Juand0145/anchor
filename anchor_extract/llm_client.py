"""LLM client helpers for anchor extraction (Anthropic and Azure OpenAI)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import anthropic
import openai
from openai import AzureOpenAI

from .settings import CLAUDE_CONFIG

_client = None
_azure_client = None

_AZURE_AUTO_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT",
)
_AZURE_REQUIRED_VARS = _AZURE_AUTO_VARS + ("AZURE_OPENAI_API_VERSION",)

_DEFAULT_TOOL_DESCRIPTION = (
    "Emit the boundary anchors of the regulatory requirements found in the provided chunk."
)


@dataclass
class LlmCallAttempt:
    model: Optional[str]
    attempt_no: int
    duration_s: float
    outcome: str
    error_message: str = ""


@dataclass
class AnchorToolResult:
    payload: dict
    model_used: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    stop_reason: str = ""
    attempts: list = field(default_factory=list)
    response_time_s: float = 0.0


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _build_model_chain(primary: str = None) -> List[str]:
    """Return the ordered list of models to try: primary first, then fallbacks."""
    primary = primary or CLAUDE_CONFIG.get("model")
    chain = [primary] if primary else []
    for m in CLAUDE_CONFIG.get("fallback_models", []) or []:
        if m and m not in chain:
            chain.append(m)
    return chain


def _azure_auto_ready() -> bool:
    return all(_env(k) for k in _AZURE_AUTO_VARS)


def _missing_azure_vars() -> List[str]:
    return [k for k in _AZURE_REQUIRED_VARS if not _env(k)]


def resolve_llm_provider() -> str:
    """Return ``azure`` or ``anthropic`` from env (``ANCHOR_LLM_PROVIDER``)."""
    raw = (_env("ANCHOR_LLM_PROVIDER") or "auto").lower()
    if raw not in {"anthropic", "azure", "auto"}:
        raise ValueError(
            "ANCHOR_LLM_PROVIDER must be anthropic, azure, or auto; "
            f"got {raw!r}."
        )
    anthropic_ready = bool(_env("ANTHROPIC_API_KEY"))
    if raw == "azure":
        missing = _missing_azure_vars()
        if missing:
            raise ValueError(
                "Azure OpenAI is not fully configured. Missing: "
                + ", ".join(missing)
            )
        return "azure"
    if raw == "anthropic":
        if not anthropic_ready:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
        return "anthropic"
    if _azure_auto_ready():
        missing = _missing_azure_vars()
        if missing:
            raise ValueError(
                "Azure OpenAI is not fully configured. Missing: "
                + ", ".join(missing)
            )
        return "azure"
    if anthropic_ready:
        return "anthropic"
    raise ValueError(
        "No LLM provider configured. Set Azure "
        "(AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
        "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT) "
        "or ANTHROPIC_API_KEY. "
        "ANTHROPIC_API_KEY environment variable not set."
    )


def get_anthropic_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    _client = anthropic.Anthropic(api_key=api_key)
    return _client


def get_azure_openai_client():
    global _azure_client
    if _azure_client is not None:
        return _azure_client
    missing = _missing_azure_vars()
    if missing:
        raise ValueError(
            "Azure OpenAI is not fully configured. Missing: "
            + ", ".join(missing)
        )
    _azure_client = AzureOpenAI(
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_API_KEY"),
        api_version=_env("AZURE_OPENAI_API_VERSION"),
    )
    return _azure_client


def _is_retryable_http(is_connection: bool, status_code) -> bool:
    return is_connection or status_code in {408, 409, 429, 500, 502, 503, 504, 529}


def _parse_tool_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Model returned invalid tool arguments JSON."
            ) from e
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError(
        "Model returned no tool_use block despite tool_choice forcing it."
    )


def _azure_chat_create(client, **kwargs):
    try:
        return client.chat.completions.create(**kwargs)
    except openai.BadRequestError as e:
        msg = str(e).lower()
        if "max_tokens" in kwargs and "max_completion_tokens" in msg:
            kwargs = dict(kwargs)
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            return client.chat.completions.create(**kwargs)
        raise


def _map_azure_stop_reason(finish_reason: Optional[str], has_tool_calls: bool) -> str:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls" or has_tool_calls:
        return "tool_use"
    return finish_reason or ""


def _resolve_tool_spec(tool_name, tool_description, input_schema):
    if tool_name is None or input_schema is None:
        from .anchor_extraction import ANCHOR_INPUT_SCHEMA, ANCHOR_TOOL_NAME
        tool_name = tool_name or ANCHOR_TOOL_NAME
        input_schema = input_schema or ANCHOR_INPUT_SCHEMA
    if tool_description is None:
        tool_description = _DEFAULT_TOOL_DESCRIPTION
    return tool_name, tool_description, input_schema


def _invoke_anthropic(
    system_prompt: str,
    chunk_text: str,
    *,
    tool_name: str,
    tool_description: str,
    input_schema: dict,
    model: Optional[str],
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    retry_wait_s: int,
) -> AnchorToolResult:
    tools = [{
        "name": tool_name,
        "description": tool_description,
        "input_schema": input_schema,
    }]
    system_blocks = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]
    messages = [{"role": "user", "content": chunk_text}]

    client = get_anthropic_client()
    last_exception = None
    response = None
    used_model = None
    response_time = 0.0
    attempts: list = []

    for current_model in _build_model_chain(model):
        for attempt_no in range(1, max_retries + 1):
            t0 = time.time()
            try:
                response = client.messages.create(
                    model=current_model,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    system=system_blocks,
                    tools=tools,
                    tool_choice={"type": "tool", "name": tool_name},
                    messages=messages,
                )
                duration = time.time() - t0
                response_time = duration
                used_model = current_model
                attempts.append(LlmCallAttempt(
                    current_model, attempt_no, round(duration, 3), "ok",
                ))
                break
            except anthropic.NotFoundError as e:
                duration = time.time() - t0
                attempts.append(LlmCallAttempt(
                    current_model, attempt_no, round(duration, 3),
                    "not_found", str(e)[:200],
                ))
                last_exception = e
                break
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                duration = time.time() - t0
                is_connection = isinstance(e, anthropic.APIConnectionError)
                status_code = getattr(e, "status_code", None)
                retryable = _is_retryable_http(is_connection, status_code)
                outcome = "connection_error" if is_connection else "api_error"
                attempts.append(LlmCallAttempt(
                    current_model, attempt_no, round(duration, 3),
                    outcome, str(e)[:200],
                ))
                last_exception = e
                if retryable and attempt_no < max_retries:
                    time.sleep(retry_wait_s)
                else:
                    break
        if response is not None:
            break

    if response is None:
        raise RuntimeError(f"All models exhausted. Last error: {last_exception}")

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        raise RuntimeError(
            "Model returned no tool_use block despite tool_choice forcing it."
        )

    usage = response.usage
    return AnchorToolResult(
        payload=tool_block.input if isinstance(tool_block.input, dict) else {},
        model_used=used_model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        stop_reason=response.stop_reason or "",
        attempts=attempts,
        response_time_s=round(response_time, 3),
    )


def _invoke_azure(
    system_prompt: str,
    chunk_text: str,
    *,
    tool_name: str,
    tool_description: str,
    input_schema: dict,
    model: Optional[str],
    max_output_tokens: int,
    temperature: float,
    max_retries: int,
    retry_wait_s: int,
) -> AnchorToolResult:
    deployment = (model or _env("AZURE_OPENAI_DEPLOYMENT")).strip()
    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT is not set.")

    tools = [{
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": input_schema,
        },
    }]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk_text},
    ]

    client = get_azure_openai_client()
    last_exception = None
    response = None
    response_time = 0.0
    attempts: list = []

    for attempt_no in range(1, max_retries + 1):
        t0 = time.time()
        try:
            response = _azure_chat_create(
                client,
                model=deployment,
                messages=messages,
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": tool_name},
                },
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            duration = time.time() - t0
            response_time = duration
            attempts.append(LlmCallAttempt(
                deployment, attempt_no, round(duration, 3), "ok",
            ))
            break
        except openai.NotFoundError as e:
            duration = time.time() - t0
            attempts.append(LlmCallAttempt(
                deployment, attempt_no, round(duration, 3),
                "not_found", str(e)[:200],
            ))
            last_exception = e
            break
        except (openai.APIError, openai.APIConnectionError) as e:
            duration = time.time() - t0
            is_connection = isinstance(e, openai.APIConnectionError)
            status_code = getattr(e, "status_code", None)
            retryable = _is_retryable_http(is_connection, status_code)
            outcome = "connection_error" if is_connection else "api_error"
            attempts.append(LlmCallAttempt(
                deployment, attempt_no, round(duration, 3),
                outcome, str(e)[:200],
            ))
            last_exception = e
            if retryable and attempt_no < max_retries:
                time.sleep(retry_wait_s)
            else:
                break

    if response is None:
        raise RuntimeError(f"All models exhausted. Last error: {last_exception}")

    choice = response.choices[0] if response.choices else None
    message = choice.message if choice is not None else None
    tool_calls = getattr(message, "tool_calls", None) if message is not None else None
    if not tool_calls:
        raise RuntimeError(
            "Model returned no tool_use block despite tool_choice forcing it."
        )

    payload = _parse_tool_arguments(tool_calls[0].function.arguments)
    usage = response.usage
    finish_reason = getattr(choice, "finish_reason", None)
    return AnchorToolResult(
        payload=payload,
        model_used=deployment,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cache_creation_tokens=0,
        cache_read_tokens=0,
        stop_reason=_map_azure_stop_reason(finish_reason, True),
        attempts=attempts,
        response_time_s=round(response_time, 3),
    )


def invoke_anchor_tool(
    system_prompt: str,
    chunk_text: str,
    *,
    model: Optional[str] = None,
    max_output_tokens: int = 4096,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_wait_s: int = 10,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    input_schema: Optional[dict] = None,
) -> AnchorToolResult:
    """Call the configured LLM with forced emit-anchors tool use."""
    tool_name, tool_description, input_schema = _resolve_tool_spec(
        tool_name, tool_description, input_schema,
    )
    provider = resolve_llm_provider()
    kwargs = dict(
        system_prompt=system_prompt,
        chunk_text=chunk_text,
        tool_name=tool_name,
        tool_description=tool_description,
        input_schema=input_schema,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        max_retries=max_retries,
        retry_wait_s=retry_wait_s,
    )
    if provider == "azure":
        return _invoke_azure(**kwargs)
    return _invoke_anthropic(**kwargs)
