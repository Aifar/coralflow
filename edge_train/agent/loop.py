"""REPL loop — prompt → LLM → tools → response → repeat."""

from __future__ import annotations

import json
import secrets
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from edge_train.agent.tools import TOOLS, execute_tool
from edge_train.agent.ui import CoralFlowUI

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
- **Training flow is mandatory: assess → choose → train**
- When the user asks to train, you MUST first call `assess_resources` with the dataset path
- After assessment, you MUST present the options and ask the user to choose (1 or 2)
- Wait for the user's choice before calling `train_model` or suggesting cloud
- If local resources are insufficient, explain why and recommend cloud
- When the user asks to train, first scan for datasets and present options
- If no local dataset matches, recommend public datasets from the web
- Always validate a dataset before training — flag quality issues early
- After training, suggest validation and deployment
- Show key results (accuracy, model size, latency) after each step
- If Phoenix is configured, suggest monitoring after deployment
- **Retrain simulation flow:** after training + predicting, use `label_predictions` (action='list') to show unlabeled predictions → ask the user which ones were wrong → call `label_predictions` (action='label') with the corrections → call `check_retrain` to trigger retraining if accuracy dropped
- Be concise — users are in the terminal
- **ALWAYS respond in English** — all responses, explanations, and tool outputs must be in English, regardless of the user's language

Format all responses in Markdown:
- Use **bold** for key values (accuracy, model names, sizes)
- Use `backticks` for commands, file paths, and code
- Use bullet lists for options and recommendations
- Use ### headers for sections when appropriate
- Keep it concise — terminal users don't need essays"""


def run_agent_loop(
    llm: "LLMClient", state, scan_result: str = "", ctx_summary: str = ""
):
    """Main REPL: prompt → LLM → tools → response → repeat.

    Slash commands are parsed into tool_calls, executed, and the result
    is sent to the LLM for a response — keeping conversation context coherent.
    """
    ui = CoralFlowUI()

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

    # Banner
    banner = (
        "TinyML continuous training — LLM-powered\n\n"
        + (f"{ctx_summary}\n\n" if ctx_summary else "")
        + "Type `/help` for commands, `/exit` to quit."
    )
    ui.panel(banner, title="CoralFlow Agent")

    while True:
        try:
            user_input = ui.prompt("coralflow")
        except (EOFError, KeyboardInterrupt):
            ui.info("\nExiting...")
            _save_and_exit(state, messages, ui)
            return

        if not user_input:
            continue

        # ── Slash commands ──────────────────────────────────────────
        if user_input.startswith("/"):
            if user_input in ("/exit", "/quit"):
                _save_and_exit(state, messages, ui)
                return

            if user_input == "/help":
                _print_help(ui)
                continue

            tool_call_msg, result_text = _handle_slash_command(user_input, llm, ui)
            if tool_call_msg is None:
                continue  # unhandled

            messages.append({"role": "user", "content": f"/{user_input[1:]}"})
            messages.append(tool_call_msg)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_msg["tool_calls"][0]["id"],
                    "content": result_text,
                }
            )

            ui.separator()
            ui.markdown(result_text)

            # Send to LLM for response
            resp = llm.chat(messages, TOOLS)
            if resp.content:
                messages.append({"role": "assistant", "content": resp.content})
                ui.markdown(resp.content)
                ui.separator()
            continue

        # ── Normal text ──────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        resp = llm.chat(messages, TOOLS)

        # Tool call loop
        while resp.tool_calls:
            # One assistant message with ALL tool_calls from this response
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
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
                        for tc in resp.tool_calls
                    ],
                }
            )

            # Execute each tool and append results
            for tc in resp.tool_calls:
                ui.tool_start(tc.name)
                result = execute_tool(tc.name, tc.arguments, llm=llm)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result) if result else "",
                    }
                )

                ui.separator()
                ui.markdown(result)
                ui.separator()

            resp = llm.chat(messages, TOOLS)

        if resp.content:
            messages.append({"role": "assistant", "content": resp.content})
            ui.markdown(resp.content)
            ui.separator()

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


def _handle_slash_command(
    user_input: str, llm: "LLMClient", ui: CoralFlowUI
) -> tuple[dict | None, str]:
    """Parse a slash command into a tool_call message + execute it. Returns (tool_call_msg, result)."""
    rest = user_input[1:].strip()

    parts = shlex.split(rest)
    if not parts:
        return None, ""

    cmd = parts[0].lower()
    args_list = parts[1:]

    tool_name = None
    tool_args: dict = {}

    if cmd == "datasets":
        tool_name = "scan_datasets"
    elif cmd == "models":
        tool_name = "scan_models"
    elif cmd == "status":
        tool_name = "get_status"
    elif cmd in ("train", "predict", "deploy", "monitor", "validate", "cost", "init"):
        tool_name = "run_shell"
        tool_args["command"] = f"{cmd} {' '.join(args_list)}"
    else:
        ui.error(f"Unknown command: /{cmd}. Type /help for available commands.")
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

    tool_call_id = f"call_{secrets.token_hex(12)}"
    tool_call_msg = {
        "role": "assistant",
        "content": "",
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


def _print_help(ui: CoralFlowUI):
    ui.panel(
        """Available slash commands:
- `/train <args>` — Train a model (e.g. `/train -d data.csv -o ./out`)
- `/predict <args>` — Run predictions
- `/deploy <args>` — Deploy a model to an edge device
- `/validate <args>` — Validate a model (TFLite conversion)
- `/monitor <args>` — Check Phoenix monitoring status
- `/cost <args>` — Estimate cloud training cost
- `/init <args>` — Download built-in datasets
- `/datasets` — List discovered datasets
- `/models` — List trained models
- `/status` — Show agent state summary
- `/help` — Show this help
- `/exit`, `/quit` — Save state and exit

You can also just describe what you want to do in natural language
and the agent will figure out the right commands to run.""",
        title="CoralFlow Commands",
    )


def _save_and_exit(state, messages: list[dict], ui: CoralFlowUI):
    """Summarize conversation, save state, and exit."""
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
    ui.success("State saved. Goodbye!")


def _trim_history(messages: list[dict]):
    """Keep system messages + last N exchanges to stay within context limits."""
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    messages.clear()
    messages.extend(system_msgs)
    messages.extend(other_msgs[-30:])
