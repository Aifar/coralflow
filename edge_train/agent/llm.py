"""OpenAI-compatible LLM API client."""

from __future__ import annotations

import json
import os
import secrets
import sys
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import requests


class PromptFn(Protocol):
    def __call__(self, label: str, *, default: str = "") -> str: ...


@dataclass
class LLMConfig:
    endpoint: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        import os

        return cls(
            endpoint=os.environ.get(
                "CORALFLOW_LLM_ENDPOINT", "https://api.openai.com/v1"
            ),
            api_key=os.environ.get("CORALFLOW_LLM_API_KEY", ""),
            model=os.environ.get("CORALFLOW_LLM_MODEL", "gpt-4o"),
        )

    def is_valid(self) -> bool:
        return bool(self.api_key.strip())


def _mask_api_key(api_key: str) -> str:
    key = api_key.strip()
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def prompt_llm_config_interactive(config: LLMConfig, prompt_fn: PromptFn) -> LLMConfig:
    """Prompt for api_key, endpoint, and model one parameter at a time."""
    api_key = prompt_fn(
        f"LLM API key (CORALFLOW_LLM_API_KEY) [{_mask_api_key(config.api_key)}]",
        default="",
    ).strip()
    if api_key.lower() == "skip":
        return LLMConfig(
            endpoint=config.endpoint,
            api_key="",
            model=config.model,
        )
    if api_key:
        config.api_key = api_key

    endpoint = prompt_fn(
        f"LLM endpoint (CORALFLOW_LLM_ENDPOINT) [{config.endpoint}]",
        default=config.endpoint,
    ).strip()
    if endpoint:
        config.endpoint = endpoint

    model = prompt_fn(
        f"LLM model (CORALFLOW_LLM_MODEL) [{config.model}]",
        default=config.model,
    ).strip()
    if model:
        config.model = model

    return config


def persist_llm_config(config: LLMConfig) -> None:
    """Save LLM settings to .env (no duplicate keys) and os.environ."""
    from edge_train.config import persist_env_values

    updates: dict[str, str] = {}
    if config.api_key.strip():
        updates["CORALFLOW_LLM_API_KEY"] = config.api_key.strip()
    if config.endpoint.strip():
        updates["CORALFLOW_LLM_ENDPOINT"] = config.endpoint.strip()
    if config.model.strip():
        updates["CORALFLOW_LLM_MODEL"] = config.model.strip()
    persist_env_values(updates)


def ensure_llm_client(
    config: LLMConfig,
    prompt_fn: PromptFn,
    *,
    echo: Callable[[str], None] | None = None,
    is_tty: bool | None = None,
) -> LLMClient:
    """Resolve LLM config interactively. Exits if the LLM cannot be configured."""
    if is_tty is None:
        is_tty = sys.stdout.isatty() and sys.stdin.isatty()

    def _exit_unavailable(message: str) -> None:
        if echo:
            echo(message)
        sys.exit(1)

    llm = LLMClient(config)
    if config.is_valid():
        ok, err = llm.verify_connection()
        if ok:
            persist_llm_config(config)
            return llm
        if not is_tty:
            _exit_unavailable(f"LLM connection failed.\n{err}")
        if echo:
            echo(f"LLM connection failed.\n{err}\n")
    elif not is_tty:
        _exit_unavailable("LLM API key is not set.")
    elif echo:
        echo("LLM API key is not set.\n")

    while True:
        if echo:
            echo(format_llm_setup_hint(config))
            echo(
                "\nEnter LLM settings below (press Enter to keep the shown default).\n"
            )

        config = prompt_llm_config_interactive(config, prompt_fn)
        llm = LLMClient(config)

        if not config.is_valid():
            _exit_unavailable(
                "LLM API key is required. Set CORALFLOW_LLM_API_KEY or pass --api-key."
            )

        ok, err = llm.verify_connection()
        if ok:
            persist_llm_config(config)
            return llm

        if echo:
            echo(f"\nLLM connection still failed.\n{err}\n")


def format_llm_setup_hint(config: LLMConfig | None = None) -> str:
    """Instructions for configuring LLM via environment or CLI flags."""
    model = config.model if config else "gpt-4o"
    endpoint = config.endpoint if config else "https://api.openai.com/v1"
    return (
        "Configure the LLM via environment or .env:\n"
        "  export CORALFLOW_LLM_API_KEY=sk-...\n"
        f"  export CORALFLOW_LLM_ENDPOINT={endpoint}  # optional\n"
        f"  export CORALFLOW_LLM_MODEL={model}  # optional\n"
        "\n"
        "Or pass flags when starting the agent:\n"
        "  coralflow agent --api-key sk-... "
        f'--endpoint "{endpoint}" --model {model}'
    )


def is_llm_error_response(resp: "LLMResponse") -> bool:
    """True when chat() returned a transport/API failure message."""
    return bool(resp.content and resp.content.startswith("Error:"))


GEMINI_SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def is_gemini_compatible(config: LLMConfig) -> bool:
    """True for Google Gemini OpenAI-compatible endpoints."""
    combined = f"{config.endpoint} {config.model}".lower()
    return (
        "generativelanguage.googleapis.com" in combined
        or "generativelanguage" in combined
        or "gemini" in combined
    )


def _thought_signature_present(tool_call: dict) -> bool:
    extra = tool_call.get("extra_content") or {}
    google = extra.get("google") or {}
    return bool(google.get("thought_signature"))


def ensure_gemini_tool_signatures(
    messages: list[dict], config: LLMConfig
) -> list[dict]:
    """Ensure Gemini 3 tool-call history includes required thought signatures."""
    if not is_gemini_compatible(config):
        return messages

    prepared: list[dict] = []
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            prepared.append(msg)
            continue

        tool_calls: list[dict] = []
        for i, tc in enumerate(msg["tool_calls"]):
            tc_copy = dict(tc)
            if i == 0 and not _thought_signature_present(tc_copy):
                extra = dict(tc_copy.get("extra_content") or {})
                google = dict(extra.get("google") or {})
                google["thought_signature"] = GEMINI_SKIP_THOUGHT_SIGNATURE
                extra["google"] = google
                tc_copy["extra_content"] = extra
            tool_calls.append(tc_copy)

        msg_copy = dict(msg)
        msg_copy["tool_calls"] = tool_calls
        prepared.append(msg_copy)
    return prepared


def tool_call_to_dict(tc: "ToolCall") -> dict:
    """Serialize a ToolCall for the chat/completions messages array."""
    args = tc.arguments
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    else:
        args_str = args or "{}"

    out: dict[str, Any] = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": args_str},
    }
    if tc.extra_content:
        out["extra_content"] = tc.extra_content
    return out


def build_assistant_tool_calls_message(resp: "LLMResponse") -> dict:
    """Build an assistant message preserving provider-specific fields."""
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": resp.content or "",
        "tool_calls": [tool_call_to_dict(tc) for tc in resp.tool_calls or []],
    }
    if resp.reasoning_content:
        msg["reasoning_content"] = resp.reasoning_content
    if resp.extra_content:
        msg["extra_content"] = resp.extra_content
    return msg


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    extra_content: dict | None = None


@dataclass
class LLMResponse:
    content: str | None = None
    reasoning_content: str | None = None
    extra_content: dict | None = None
    tool_calls: list[ToolCall] | None = None


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def verify_connection(self) -> tuple[bool, str]:
        """Probe the LLM API with a minimal request. Returns (ok, error_message)."""
        if not self.config.is_valid():
            return False, "CORALFLOW_LLM_API_KEY is not set."

        resp = self.chat([{"role": "user", "content": "ping"}], tools=None)
        if is_llm_error_response(resp):
            return False, resp.content or "LLM request failed."
        if resp.content or resp.tool_calls:
            return True, ""
        return False, "No response from LLM."

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        """Send a chat completion request. Returns LLMResponse with content and/or tool_calls."""
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": ensure_gemini_tool_signatures(messages, self.config),
        }

        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            # Some providers reject tools; do not strip tools for Gemini signature errors.
            if resp.status_code == 400 and tools:
                detail = resp.text.lower()
                if "thought_signature" not in detail:
                    payload.pop("tools", None)
                    payload.pop("tool_choice", None)
                    resp = requests.post(
                        url, headers=headers, json=payload, timeout=120
                    )
            resp.raise_for_status()
            body = resp.json()
        except requests.Timeout:
            return LLMResponse(content="Error: LLM request timed out after 120s.")
        except requests.RequestException as e:
            detail = ""
            try:
                detail = resp.text[:500]  # type: ignore[unbound]
            except Exception:
                pass
            msg = f"Error: LLM request failed: {e}"
            if detail:
                msg += f"\nResponse: {detail}"
            return LLMResponse(content=msg)

        choices = body.get("choices") or []
        if not choices:
            return LLMResponse(
                content=body.get("error", {}).get("message", "No response from LLM.")
            )
        choice = choices[0]
        message = choice.get("message", {})

        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        extra_content = message.get("extra_content")
        raw_tool_calls = message.get("tool_calls") or []

        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, dict):
                    args = raw_args
                else:
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        args = {}
                extra = tc.get("extra_content")
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id") or f"call_{secrets.token_hex(12)}",
                        name=func.get("name", ""),
                        arguments=args,
                        extra_content=extra if isinstance(extra, dict) else None,
                    )
                )

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            extra_content=extra_content if isinstance(extra_content, dict) else None,
            tool_calls=tool_calls,
        )
