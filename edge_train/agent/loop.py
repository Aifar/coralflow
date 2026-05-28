"""REPL loop — prompt → LLM → tools → response → repeat."""

from __future__ import annotations

import json
import secrets
import shlex
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
- When the user replies with only `1` or `2` after training options, call `train_model` for option 1; for option 2 use `run_shell` with `train --cloud ...` (auto-routed: Gemini SFT for text, AutoML Tabular/Image/Video otherwise). Always pass `--purpose <project_name>` when the user names the project (e.g. neu_cls_defect_classifier_v3).
- **After cloud AutoML training succeeds**, present post-training options:
  1. Export & deploy to edge (local TFLite path when available)
  2. **Run predictions on the test set** — call `run_predictions` (NOT ad-hoc Python). Always show the inference environment (Vertex cloud vs local).
  3. Validate model size & latency for edge deployment
- For option 2 (test-set predictions): use `run_predictions` with `endpoint`, `modality`, and `test_jsonl` (e.g. `gs://coralflow/neu-cls/test.jsonl`). Equivalent CLI: `coralflow evaluate --endpoint ... --modality image --dataset ...`
- **NEVER** use `run_shell` with inline Python (`google.cloud.aiplatform`, heredocs). Use `run_predictions`, `predict`, or CLI subcommands (`evaluate`, `simulate`, `deploy --cloud`) instead.
- AutoML Image models deploy with **automatic_resources** on Vertex — do not pass machine_type to deploy.
- For cloud **text** training (Gemini Fine-Tuning), always state which **publisher base model** will be fine-tuned (from `GCP_FINETUNE_MODEL`, default `gemini-2.0-flash-001`). Mention `coralflow models list` to see alternatives and `--base-model` / `GCP_FINETUNE_MODEL` to change it
- Wait for the user's choice before calling `train_model` or suggesting cloud
- Use `run_shell` for coralflow CLI subcommands (`train`, `predict`, `deploy`, `validate`, `monitor`, `cost`, `init`, `simulate`, `evaluate`) when the user asks to run CLI-style commands
- Prefer dedicated tools (`train_model`, `run_predictions`, `predict`, etc.) for the guided agent workflow; use `run_shell` when the user explicitly wants the CLI or passes CLI flags
- If local resources are insufficient, explain why and recommend cloud
- When the user asks to train, first scan for datasets and present options
- If no local dataset matches, recommend public datasets from the web
- Always validate a dataset before training — flag quality issues early
- After training, suggest validation and deployment
- Show key results (accuracy, model size, latency) after each step
- If Phoenix is configured, suggest monitoring after deployment; `check_monitoring` verifies Phoenix is reachable before predict/monitor
- **Retrain simulation flow:** after training + predicting, use `label_predictions` (action='list') to show unlabeled predictions → ask the user which ones were wrong → call `label_predictions` (action='label') with the corrections → call `check_retrain` to trigger retraining if accuracy dropped
- Be concise — users are in the terminal
- **ALWAYS respond in English** — all responses, explanations, and tool outputs must be in English, regardless of the user's language

Format all responses in Markdown:
- Use **bold** for key values (accuracy, model names, sizes)
- Use `backticks` for commands, file paths, and code
- Use bullet lists for options and recommendations
- Use ### headers for sections when appropriate
- Keep it concise — terminal users don't need essays"""


def _assistant_msg(resp) -> dict:
    """Build an assistant message dict, including reasoning_content if present (DeepSeek)."""
    msg: dict = {"role": "assistant", "content": resp.content or ""}
    if resp.reasoning_content:
        msg["reasoning_content"] = resp.reasoning_content
    return msg


def _prepare_messages_for_llm(messages: list[dict]) -> list[dict]:
    """Ensure tool messages follow assistant tool_calls (OpenAI / DeepSeek requirement)."""
    prepared: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")

        if role == "tool":
            i += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            expected_ids = [tc.get("id") for tc in tool_calls if tc.get("id")]
            i += 1
            tool_by_id: dict[str, dict] = {}
            while i < len(messages) and messages[i].get("role") == "tool":
                tid = messages[i].get("tool_call_id")
                if tid in expected_ids:
                    tool_by_id[tid] = messages[i]
                i += 1

            if expected_ids and len(tool_by_id) == len(expected_ids):
                prepared.append(msg)
                for tid in expected_ids:
                    prepared.append(tool_by_id[tid])
            else:
                stripped = {k: v for k, v in msg.items() if k != "tool_calls"}
                content = stripped.get("content") or ""
                if not content:
                    stripped["content"] = "(tool call omitted — incomplete history)"
                prepared.append(stripped)
            continue

        prepared.append(msg)
        i += 1

    return prepared


def _run_llm_turn(messages: list[dict], llm: "LLMClient", ui: CoralFlowUI) -> None:
    """Call LLM until no more tool calls; append all messages and print responses."""
    resp = llm.chat(_prepare_messages_for_llm(messages), TOOLS)

    while resp.tool_calls:
        tc_msg: dict = {
            "role": "assistant",
            "content": resp.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ],
        }
        if resp.reasoning_content:
            tc_msg["reasoning_content"] = resp.reasoning_content
        messages.append(tc_msg)

        for tc in resp.tool_calls:
            cmd = None
            if tc.name == "run_shell":
                cmd = (tc.arguments.get("command") or "").strip() or "(no command)"
                ui.tool_start(tc.name, cmd)
                sub = cmd.split()[0] if cmd.split() else ""
                if sub in ("train", "validate"):
                    ui.info(
                        "Running CLI (may take several minutes) — live output below:"
                    )
            else:
                ui.tool_start(tc.name)
            result = execute_tool(tc.name, tc.arguments, llm=llm, ui=ui)
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

        resp = llm.chat(_prepare_messages_for_llm(messages), TOOLS)

    if resp.content:
        messages.append(_assistant_msg(resp))
        ui.markdown(resp.content)
        ui.separator()


def run_agent_loop(
    llm: "LLMClient", state, scan_result: str = "", ctx_summary: str = ""
):
    """Main REPL: prompt → LLM → tools → response → repeat.

    Slash commands (except `/help`, `/exit`) run locally first; the result is
    sent to the LLM for a follow-up reply. All other input goes to the LLM,
    which decides which tools to call next.
    """
    ui = CoralFlowUI()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Inject startup context
    from edge_train.agent.context import format_project_context

    project_ctx = format_project_context(state)
    if project_ctx:
        messages.append(
            {
                "role": "system",
                "content": f"Saved project context (resume from here):\n{project_ctx}",
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

            tool_call_msg, result_text = _handle_slash_command(user_input, ui)
            if tool_call_msg is None:
                continue

            messages.append({"role": "user", "content": user_input})
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

            _run_llm_turn(messages, llm, ui)
            continue

        # ── Normal text → LLM decides tools ─────────────────────────
        messages.append({"role": "user", "content": user_input})
        _run_llm_turn(messages, llm, ui)

        # Keep conversation bounded
        if len(messages) > 40:
            _trim_history(messages)
            from edge_train.agent.context import format_project_context

            project_ctx = format_project_context(state)
            if project_ctx:
                messages.insert(
                    1,
                    {
                        "role": "system",
                        "content": f"Saved project context:\n{project_ctx}",
                    },
                )


def _handle_slash_command(user_input: str, ui: CoralFlowUI) -> tuple[dict | None, str]:
    """Run a slash command locally, return (synthetic tool_call message, result)."""
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
        result = execute_tool(tool_name, {}, ui=ui)
    else:
        shell_cmd = tool_args.get("command", "")
        ui.tool_start(tool_name, shell_cmd)
        if shell_cmd.split()[0] in ("train", "validate"):
            ui.info("Running CLI (may take several minutes) — live output below:")
        result = execute_tool(tool_name, tool_args, ui=ui)

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
        """Slash commands (run locally, then summarized by the LLM):
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

Natural language (e.g. "train on urgent", or `1` after training options) is
sent to the LLM, which chooses tools (`train_model`, `run_shell`, etc.).""",
        title="CoralFlow Commands",
    )


def _save_and_exit(state, messages: list[dict], ui: CoralFlowUI):
    """Persist focused project context and exit."""
    from edge_train.agent.context import sync_agent_context

    sync_agent_context(state)
    ui.success("Project context saved. Goodbye!")


def _trim_history(messages: list[dict], max_messages: int = 30):
    """Keep system messages + recent history without splitting tool/tool_calls pairs."""
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    blocks: list[list[dict]] = []
    i = 0
    while i < len(other_msgs):
        m = other_msgs[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            block = [m]
            i += 1
            while i < len(other_msgs) and other_msgs[i].get("role") == "tool":
                block.append(other_msgs[i])
                i += 1
            blocks.append(block)
        else:
            blocks.append([m])
            i += 1

    total = sum(len(b) for b in blocks)
    start = 0
    while start < len(blocks) and total > max_messages:
        total -= len(blocks[start])
        start += 1

    trimmed = [msg for block in blocks[start:] for msg in block]
    messages.clear()
    messages.extend(system_msgs)
    messages.extend(trimmed)
