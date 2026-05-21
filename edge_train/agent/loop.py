"""REPL loop — prompt → LLM → tools → response → repeat."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

import click

from edge_train.agent.tools import TOOLS, execute_tool

if TYPE_CHECKING:
    from edge_train.agent.llm import LLMClient


SYSTEM_PROMPT = """You are a CoralFlow agent — an AI assistant that helps users train, validate, deploy, and monitor tiny ML models on edge devices.

Capabilities:
- Assess local hardware (CPU, RAM, disk) to determine training eligibility
- Discover datasets on disk and recommend public datasets from the web
- Validate dataset quality: column structure, class balance, data types
- Train text classifiers locally with TensorFlow (no GPU needed)
- Convert models to TFLite and validate size/latency for edge deployment
- Deploy models to edge devices via HTTP transport
- Run predictions and monitor performance via Arize Phoenix
- Check prediction accuracy and trigger retraining when needed

Guidelines:
- Before any training, assess local resources and explain the recommendation
- If local resources are sufficient, offer both local and cloud options
- If local resources are insufficient, recommend cloud with clear rationale
- When the user asks to train, first scan for datasets and present options
- If no local dataset matches, recommend public datasets from the web
- Always validate a dataset before training — flag quality issues early
- After training, suggest validation and deployment
- Show key results (accuracy, model size, latency) after each step
- If Phoenix is configured, suggest monitoring after deployment
- Be concise — users are in the terminal"""


def run_agent_loop(llm: "LLMClient", state, scan_result: str = ""):
    """Main REPL: prompt → LLM → tools → response → repeat.

    Slash commands are parsed into tool_calls, executed, and the result
    is sent to the LLM for a response — keeping conversation context coherent.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Inject startup context
    if state.conversation_summary:
        messages.append(
            {
                "role": "system",
                "content": f"Previous session summary: {state.conversation_summary}",
            }
        )
    if scan_result:
        messages.append({"role": "system", "content": f"Startup scan:\n{scan_result}"})

    click.echo()
    click.echo("Type /help for commands, /exit to quit.")
    click.echo()

    while True:
        try:
            user_input = click.prompt("coralflow", prompt_suffix="> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nExiting...")
            _save_and_exit(state, messages)
            return

        if not user_input:
            continue

        # ── Slash commands ──────────────────────────────────────────
        if user_input.startswith("/"):
            if user_input in ("/exit", "/quit"):
                _save_and_exit(state, messages)
                return

            if user_input == "/help":
                _print_help()
                continue

            tool_call_msg, result_text = _handle_slash_command(user_input, llm)
            if tool_call_msg is None:
                continue  # unhandled

            messages.append({"role": "user", "content": f"/{user_input[1:]}"})
            messages.append(tool_call_msg)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_msg["tool_call_id"],
                    "content": result_text,
                }
            )

            # Send to LLM for response
            resp = llm.chat(messages, TOOLS)
            if resp.content:
                messages.append({"role": "assistant", "content": resp.content})
                click.echo()
                click.echo(resp.content)
            continue

        # ── Normal text ──────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        resp = llm.chat(messages, TOOLS)

        # Tool call loop
        while resp.tool_calls:
            for tc in resp.tool_calls:
                click.echo(f"  ⚙ {tc.name}...")
                result = execute_tool(tc.name, tc.arguments, llm=llm)

                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(
                                        tc.arguments, ensure_ascii=False
                                    ),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

            resp = llm.chat(messages, TOOLS)

        if resp.content:
            messages.append({"role": "assistant", "content": resp.content})
            click.echo()
            click.echo(resp.content)

        # Keep conversation bounded
        if len(messages) > 40:
            _trim_history(messages)
            messages.insert(
                1,
                {
                    "role": "system",
                    "content": f"Previous session summary: {state.conversation_summary}",
                },
            )


def _handle_slash_command(user_input: str, llm: "LLMClient") -> tuple[dict | None, str]:
    """Parse a slash command into a tool_call message + execute it. Returns (tool_call_msg, result)."""
    # Strip the leading /
    rest = user_input[1:].strip()

    # Split into command name and arguments
    parts = shlex.split(rest)
    if not parts:
        return None, ""

    cmd = parts[0].lower()
    args_list = parts[1:]

    # Map slash commands to tool calls
    tool_name = None
    tool_args: dict = {}

    if cmd == "datasets":
        tool_name = "scan_datasets"
    elif cmd == "models":
        tool_name = "scan_models"
    elif cmd == "status":
        tool_name = "get_status"
    elif cmd in ("train", "predict", "deploy", "monitor", "validate", "cost", "init"):
        # Rebuild the coralflow command
        tool_name = "run_shell"
        tool_args["command"] = f"{cmd} {' '.join(args_list)}"
    else:
        click.echo(f"  Unknown command: /{cmd}. Type /help for available commands.")
        return None, ""

    if tool_name == "scan_models":
        from edge_train.agent import scan_models

        models = scan_models()
        if not models:
            result = "No trained models found."
        else:
            lines = [f"Found {len(models)} model(s):"]
            for m in models:
                classes_str = ", ".join(m.get("classes", [])[:5])
                lines.append(
                    f"  • {m['name']} — {classes_str} ({m.get('created_at', '?')})"
                )
            result = "\n".join(lines)
    elif tool_name in ("scan_datasets", "get_status"):
        result = execute_tool(tool_name, {})
    else:
        result = execute_tool(tool_name, tool_args)

    tool_call_id = f"call_{cmd}_{len(result)}"
    tool_call_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args, ensure_ascii=False),
                },
            }
        ],
    }

    return tool_call_msg, result


def _print_help():
    click.echo("""
Available slash commands:
  /train <args>    — Train a model (e.g. /train -d data.csv -o ./out)
  /predict <args>  — Run predictions
  /deploy <args>   — Deploy a model to an edge device
  /validate <args> — Validate a model (TFLite conversion)
  /monitor <args>  — Check Phoenix monitoring status
  /cost <args>     — Estimate cloud training cost
  /init <args>     — Download built-in datasets
  /datasets        — List discovered datasets
  /models          — List trained models
  /status          — Show agent state summary
  /help            — Show this help
  /exit, /quit     — Save state and exit

You can also just describe what you want to do in natural language
and the agent will figure out the right commands to run.
""")


def _save_and_exit(state, messages: list[dict]):
    """Summarize conversation, save state, and exit."""
    # Summarize the last few exchanges
    recent = [
        m
        for m in messages[-10:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    summary = " | ".join(m["content"][:100] for m in recent[-5:]) if recent else ""
    if summary:
        state.conversation_summary = summary[:500]
        state.last_step = "agent_session"
    state.save()
    click.echo("State saved. Goodbye!")


def _trim_history(messages: list[dict]):
    """Keep system messages + last N exchanges to stay within context limits."""
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    messages.clear()
    messages.extend(system_msgs)
    messages.extend(other_msgs[-30:])
