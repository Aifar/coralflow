"""OpenAI-compatible LLM API client."""

import json
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
        return bool(self.api_key)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config

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
            payload["tool_choice"] = "auto"

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            body = resp.json()
        except requests.Timeout:
            return LLMResponse(content="Error: LLM request timed out after 120s.")
        except requests.RequestException as e:
            return LLMResponse(content=f"Error: LLM request failed: {e}")

        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content")
        raw_tool_calls = message.get("tool_calls", [])

        tool_calls = None
        if raw_tool_calls:
            tool_calls = []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""), name=func.get("name", ""), arguments=args
                    )
                )

        return LLMResponse(content=content, tool_calls=tool_calls)
