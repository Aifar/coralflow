"""OpenAI-compatible LLM API client."""

import json
import secrets
from dataclasses import dataclass, field
from typing import Any

import requests


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


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    reasoning_content: str | None = None
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
            "messages": messages,
        }

        if tools:
            payload["tools"] = tools

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            # Retry once without tool_choice / tools if the API rejects them
            if resp.status_code == 400 and tools:
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
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
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id") or f"call_{secrets.token_hex(12)}",
                        name=func.get("name", ""),
                        arguments=args,
                    )
                )

        return LLMResponse(
            content=content, reasoning_content=reasoning_content, tool_calls=tool_calls
        )
