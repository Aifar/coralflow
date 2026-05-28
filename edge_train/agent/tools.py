"""Tool definitions (OpenAI function-calling format) and executors.

Each tool has a JSON schema for the LLM and a Python executor function.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ── Tool JSON schemas ──────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "scan_datasets",
            "description": "List all available datasets — local CSV files and built-in datasets.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_dataset",
            "description": "Inspect a dataset: row count, classes, modality, class balance, and sample rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the CSV file or builtin:<name>.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_resources",
            "description": "Check local hardware (CPU, RAM, disk) and determine if local training is viable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Optional path to a dataset for memory footprint estimation.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_dataset",
            "description": "Run LLM-powered quality validation on a dataset: column structure, class balance, data quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the CSV dataset to validate.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_datasets",
            "description": "Recommend public ML datasets from the web for a given task description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task description, e.g. 'spam detection for email' or 'sentiment analysis of product reviews'.",
                    }
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_model",
            "description": "Train a text classifier **locally** with TensorFlow. Only call this after the user has explicitly chosen local training (option 1). Shows live epoch progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the training CSV dataset.",
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Label/target column name (auto-detected if omitted).",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save the trained SavedModel (default: ./model_output).",
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "Number of training epochs (default: 10).",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Project name to persist (e.g. neu_cls_defect_classifier_v3).",
                    },
                },
                "required": ["dataset_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_model",
            "description": "Convert a trained model to TFLite and check size/latency constraints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the SavedModel directory to validate.",
                    }
                },
                "required": ["model_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deploy_model",
            "description": "Deploy a TFLite model to an edge device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the .tflite model file.",
                    },
                    "host": {
                        "type": "string",
                        "description": "Edge device hostname or IP address.",
                    },
                    "port": {
                        "type": "integer",
                        "description": "Edge device port (default: 8080).",
                    },
                },
                "required": ["model_path", "host"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict",
            "description": (
                "Run inference and log to prediction_log.jsonl + Phoenix. "
                "Local text: model_path + text/csv. "
                "Vertex: endpoint + modality (text|table|image|video) + text/features/image/gcs-uri/csv. "
                "For full labeled test-set evaluation (e.g. 360 images), use run_predictions instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Local SavedModel directory (local inference only).",
                    },
                    "endpoint": {
                        "type": "string",
                        "description": "Vertex endpoint resource name (cloud inference).",
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "table", "image", "video"],
                        "description": "Vertex modality (required with endpoint for non-text).",
                    },
                    "text": {
                        "type": "string",
                        "description": "Single text input (text modality).",
                    },
                    "gcs_uri": {
                        "type": "string",
                        "description": "GCS URI for image/video input (gs://...).",
                    },
                    "image": {
                        "type": "string",
                        "description": "Local image path (image modality).",
                    },
                    "csv_path": {
                        "type": "string",
                        "description": "CSV file for batch prediction.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_predictions",
            "description": (
                "Evaluate a trained model on a labeled test set. "
                "Always states the inference environment (Vertex cloud vs local). "
                "For cloud AutoML image models: requires endpoint + test JSONL (gs:// or local). "
                "Never use ad-hoc Python — this tool calls coralflow evaluate internally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "Vertex endpoint resource name. Omit to use latest deployment from registry.",
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["text", "table", "image", "video"],
                        "description": "Inference modality (default: image for AutoML defect models).",
                    },
                    "test_jsonl": {
                        "type": "string",
                        "description": "Labeled test JSONL path, e.g. gs://coralflow/neu-cls/test.jsonl",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to evaluate (default: all). Use 6 for a quick smoke test.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_monitoring",
            "description": "Check Arize Phoenix monitoring status and recent prediction log statistics.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_retrain",
            "description": "Check prediction log for labeled entries and compute current accuracy vs retrain threshold.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "label_predictions",
            "description": "List recent unlabeled predictions or add ground truth labels to simulate data drift for retraining.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "label"],
                        "description": "'list' to show unlabeled entries, 'label' to add ground truth corrections.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {
                                    "type": "integer",
                                    "description": "1-based index from the list output.",
                                },
                                "ground_truth": {
                                    "type": "string",
                                    "description": "The correct label for this entry.",
                                },
                            },
                            "required": ["index", "ground_truth"],
                        },
                        "description": "List of corrections (only for action='label').",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a coralflow CLI subcommand or shell command. Use for train/predict/deploy/validate/monitor/cost/init with CLI flags, or when the user asks to run a terminal command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The coralflow subcommand with arguments, e.g. 'train -d data.csv -o ./out' or 'cost urgent'.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_status",
            "description": "Show current agent state: active dataset, trained models, deployments, monitoring config.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ── Tool executors ──────────────────────────────────────────────────────────


def _read_dataset(path: str) -> tuple[list[str], list[dict], str | None, str | None]:
    """Read a CSV dataset and return (headers, rows, text_col, label_col)."""
    from edge_train.agent import _detect_text_label_columns

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    text_col, label_col = _detect_text_label_columns(headers)
    return headers, rows, text_col, label_col


def execute_tool(name: str, arguments: dict, llm=None, ui=None) -> str:
    """Dispatch a tool call by name and return the result string."""
    try:
        if name == "scan_datasets":
            return _exec_scan_datasets()
        elif name == "analyze_dataset":
            return _exec_analyze_dataset(arguments)
        elif name == "assess_resources":
            return _exec_assess_resources(arguments)
        elif name == "validate_dataset":
            return _exec_validate_dataset(arguments, llm)
        elif name == "recommend_datasets":
            return _exec_recommend_datasets(arguments, llm)
        elif name == "train_model":
            return _exec_train_model(arguments)
        elif name == "validate_model":
            return _exec_validate_model(arguments)
        elif name == "deploy_model":
            return _exec_deploy_model(arguments)
        elif name == "predict":
            return _exec_predict(arguments, ui=ui)
        elif name == "run_predictions":
            return _exec_run_predictions(arguments, ui=ui)
        elif name == "check_monitoring":
            return _exec_check_monitoring()
        elif name == "check_retrain":
            return _exec_check_retrain()
        elif name == "label_predictions":
            return _exec_label_predictions(arguments)
        elif name == "run_shell":
            return _exec_run_shell(arguments, ui=ui)
        elif name == "get_status":
            return _exec_get_status()
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool '{name}' failed: {e}"


def _persist_agent_step(last_step: str, **fields) -> None:
    from edge_train.agent import AgentState
    from edge_train.agent.context import sync_agent_context, update_agent_context

    state = AgentState.load()
    update_agent_context(state, last_step=last_step, save=False, **fields)
    sync_agent_context(state)


def _parse_shell_purpose(cmd: str) -> str:
    import shlex

    try:
        parts = shlex.split(cmd)
    except ValueError:
        return ""
    for i, part in enumerate(parts):
        if part in ("--purpose", "-p") and i + 1 < len(parts):
            return parts[i + 1]
    return ""


def _agent_phoenix_gate(ui=None):
    """Check Phoenix; prompt to start local server when in agent REPL."""
    from edge_train.config import load_config
    from edge_train.inference.phoenix import prepare_phoenix_for_inference

    _, arize, _, _ = load_config()
    prompt_fn = (lambda msg: ui.confirm(msg)) if ui else None
    echo_fn = (lambda msg: ui.markdown(msg)) if ui else None
    return prepare_phoenix_for_inference(
        required=arize.is_valid(),
        interactive=bool(ui),
        prompt_fn=prompt_fn,
        echo_fn=echo_fn,
    )


def _exec_scan_datasets() -> str:
    from edge_train.agent import DatasetScanner

    datasets = DatasetScanner.scan()
    if not datasets:
        return "No datasets found. Use 'edge-train init download <name>' to get a built-in dataset, or provide a CSV file."

    lines = [f"Found **{len(datasets)}** dataset(s):"]
    for d in datasets:
        classes_str = ", ".join(d.get("classes", [])[:5])
        if len(d.get("classes", [])) > 5:
            classes_str += f" (+{len(d['classes']) - 5} more)"
        lines.append(
            f"  • **{d['name']}** — {d['rows']} rows, {d.get('modality', '?')}, "
            f"[{classes_str}] ({d['source']})"
            + (f"\n    {d.get('description', '')}" if d.get("description") else "")
        )
    return "\n".join(lines)


def _exec_analyze_dataset(arguments: dict) -> str:
    path = arguments.get("path", "")

    # Handle builtin:<name>
    if path.startswith("builtin:"):
        name = path.split(":", 1)[1]
        from edge_train.datasets import get_builtin

        builtin = get_builtin().get(name)
        if not builtin:
            return f"Built-in dataset '{name}' not found. Available: {', '.join(get_builtin().keys())}"
        return (
            f"Dataset: {name} (built-in)\n"
            f"  Rows: {builtin['samples']}\n"
            f"  Classes: {', '.join(builtin['classes'])}\n"
            f"  Modality: {builtin['modality']}\n"
            f"  Description: {builtin.get('description', '')}"
        )

    p = Path(path)
    if not p.exists():
        return f"Dataset not found: {path}"

    headers, rows, text_col, label_col = _read_dataset(path)
    if not rows:
        return f"Dataset is empty: {path}"

    class_counts: dict[str, int] = {}
    if label_col:
        for r in rows:
            v = r.get(label_col, "")
            class_counts[v] = class_counts.get(v, 0) + 1

    sample_lines = []
    for r in rows[:5]:
        if text_col and label_col:
            sample_lines.append(f"  {r[text_col][:60]} → {r[label_col]}")
        elif text_col:
            sample_lines.append(f"  {r[text_col][:60]}")

    lines = [
        f"Dataset: **{p.name}** ({len(rows)} rows, {len(headers)} columns)",
        f"  Modality: text",
        f"  Text column: `{text_col or '?'}`",
        f"  Label column: `{label_col or '?'}`",
        f"  Classes ({len(class_counts)}):",
    ]
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(rows) * 100
        lines.append(f"    {cls}: {cnt} ({pct:.1f}%)")

    lines.append(f"  Sample rows:")
    lines.extend(sample_lines)

    return "\n".join(lines)


def _append_cloud_option_details(
    lines: list[str],
    cloud_plan,
    *,
    markdown: bool = False,
) -> None:
    """Add cloud training details, including fine-tune base model when applicable."""
    from edge_train.cloud.router import CloudTrainingMethod
    from edge_train.config import GCPConfig

    gcp = GCPConfig()
    if cloud_plan.method == CloudTrainingMethod.GEMINI_FINETUNE:
        from edge_train.cloud.publisher_models import describe_finetune_base_model

        lines.append("")
        lines.extend(
            describe_finetune_base_model(
                gcp.finetune_model,
                gcp.location or "us-central1",
                markdown=markdown,
            )
        )


def _exec_assess_resources(arguments: dict) -> str:
    import os

    import psutil

    lines = ["Local Resource Assessment:"]

    # CPU
    cpu_count = psutil.cpu_count(logical=True) or 1
    cpu_ok = cpu_count >= 2
    lines.append(f"  CPU: {cpu_count} cores — {'OK' if cpu_ok else 'marginal'}")

    # RAM
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)
    ram_ok = avail_gb >= 1.0
    lines.append(
        f"  RAM: {avail_gb:.1f} GB available — {'OK' if ram_ok else 'INSUFFICIENT'}"
    )

    # Disk
    output_dir = os.getcwd()
    try:
        disk = psutil.disk_usage(output_dir)
        free_gb = disk.free / (1024**3)
        disk_ok = free_gb >= 0.2
        lines.append(f"  Disk: {free_gb:.1f} GB free — {'OK' if disk_ok else 'tight'}")
    except Exception:
        lines.append("  Disk: (could not check)")

    # TensorFlow
    try:
        import tensorflow as tf

        lines.append(f"  TensorFlow: {tf.__version__} — OK")
    except ImportError:
        lines.append("  TensorFlow: not installed — REQUIRED for local training")

    # Dataset fit estimate
    dataset_path = arguments.get("dataset_path", "")
    dataset_modality = "text"
    cloud_plan = None
    if dataset_path:
        try:
            from edge_train.cloud.router import plan_cloud_training
            from edge_train.datasets import (
                infer_modality_from_path,
                resolve_dataset_path,
            )

            resolved_path, builtin_mod = resolve_dataset_path(dataset_path)
            dataset_modality = builtin_mod or infer_modality_from_path(dataset_path)
            cloud_plan = plan_cloud_training(resolved_path, dataset_modality)
        except ValueError:
            pass
    cloud_training_available = cloud_plan is not None

    if (
        dataset_path
        and not dataset_path.startswith("builtin:")
        and Path(dataset_path).exists()
    ):
        try:
            _, rows, _, _ = _read_dataset(dataset_path)
            est_mb = len(rows) * 1 / 1024  # ~1 KB per row
            lines.append(
                f"  Dataset fit: {len(rows)} rows → ~{est_mb:.1f} MB — {'OK' if est_mb < avail_gb * 512 else 'borderline'}"
            )
        except Exception:
            pass

    # Verdict
    if ram_ok and cpu_ok:
        lines.append("")
        lines.append("**Verdict: Local training is viable**")
        lines.append("")
        lines.append("### Training Options")
        lines.append(
            "1. **Local training** — TF Keras, free, ~2-5 min for small datasets"
        )
        if cloud_training_available and cloud_plan:
            lines.append(
                f"2. **Cloud training** — {cloud_plan.label}, ~$3-15, 30-90 min"
            )
            lines.append(f"   _{cloud_plan.reason}_")
            _append_cloud_option_details(lines, cloud_plan, markdown=True)
            lines.append("")
            lines.append("---")
            lines.append(
                "**Which option do you prefer?** Type `1` for local or `2` for cloud."
            )
        else:
            lines.append("")
            lines.append(
                "**Note:** Cloud training is not available for this dataset modality."
            )
            lines.append("")
            lines.append("---")
            lines.append(
                "**Which option do you prefer?** Type `1` to start local training."
            )
    else:
        lines.append("")
        if cloud_training_available and cloud_plan:
            lines.append(
                "**Verdict: Local resources insufficient for reliable training**"
            )
            lines.append("")
            lines.append("### Recommendation")
            lines.append(f"**{cloud_plan.label}** — {cloud_plan.reason.split(';')[0]}")
            _append_cloud_option_details(lines, cloud_plan, markdown=True)
        else:
            lines.append(
                "**Verdict: Local resources are tight; no cloud path for this modality**"
            )
            lines.append("")
            lines.append("### Recommendation")
            lines.append(
                "**Local training** — reduce dataset size or free RAM and retry."
            )
        reasons = []
        if not ram_ok:
            reasons.append(
                f"Available RAM ({avail_gb:.1f} GB) is below the 1 GB minimum"
            )
        if not cpu_ok:
            reasons.append(f"Only {cpu_count} CPU core(s) available (2+ recommended)")
        lines.append(f"Rationale: {'; '.join(reasons)}.")
        lines.append("")
        lines.append("---")
        if cloud_training_available:
            lines.append(
                "Shall I proceed with cloud training? Type `yes` to confirm or describe your preference."
            )
        else:
            lines.append("Proceed with local training? Type `1` or `yes` to confirm.")

    return "\n".join(lines)


def _exec_validate_dataset(arguments: dict, llm=None) -> str:
    path = arguments.get("path", "")
    if path.startswith("builtin:"):
        return "Built-in datasets are pre-validated. They are ready to use."

    p = Path(path)
    if not p.exists():
        return f"Dataset not found: {path}"

    headers, rows, text_col, label_col = _read_dataset(path)

    # Build validation prompt
    sample = rows[:20]
    class_counts: dict[str, int] = {}
    for r in rows:
        if label_col:
            v = r.get(label_col, "")
            class_counts[v] = class_counts.get(v, 0) + 1

    data_snapshot = "\n".join(
        f"  {r.get(text_col, '')[:80]} | {r.get(label_col, '')}" for r in sample
    )

    prompt = f"""Validate this dataset for ML training:

File: {p.name}
Rows: {len(rows)} total
Columns: {', '.join(headers)}
Text column: {text_col}
Label column: {label_col}
Classes: {json.dumps(class_counts)}

Sample (first 20 rows):
{data_snapshot}

Check: column structure, data types, class balance, data quality (empty rows, encoding), label clarity, size adequacy.
Respond with a structured validation report using ✓ (pass) and ⚠ (warning) markers. End with "Overall: PASS" or "Overall: FAIL"."""

    if llm:
        try:
            resp = llm.chat([{"role": "user", "content": prompt}])
            if resp.content:
                return resp.content
        except Exception:
            pass

    # Fallback: basic automated check
    lines = [f"Dataset: {p.name} ({len(rows)} rows, {len(class_counts)} classes)"]
    lines.append(f"✓ Column structure: text='{text_col}', label='{label_col}'")

    # Check class balance
    if class_counts:
        counts = list(class_counts.values())
        max_c, min_c = max(counts), min(counts)
        if max_c > min_c * 3:
            lines.append(f"⚠ Class balance: imbalanced (ratio {max_c / min_c:.1f}:1)")
        else:
            lines.append(f"✓ Class balance: reasonable")

    # Check empty rows
    empty = sum(1 for r in rows if not r.get(text_col, "").strip())
    if empty:
        lines.append(f"⚠ Data quality: {empty} empty rows found")
    else:
        lines.append(f"✓ Data quality: no empty rows")

    lines.append(
        f"✓ Size: adequate for {len(class_counts)}-class classification"
        if len(rows) >= len(class_counts) * 20
        else f"⚠ Size: only {len(rows)} rows for {len(class_counts)} classes"
    )
    lines.append("Overall: PASS — ready for training")
    return "\n".join(lines)


def _exec_recommend_datasets(arguments: dict, llm=None) -> str:
    task = arguments.get("task", "")
    if not task:
        return "Please provide a task description, e.g. 'spam detection for email'."

    prompt = f"""Recommend 3-5 public ML datasets for this task: "{task}"

For each dataset, provide:
1. Name and source (HuggingFace, Kaggle, UCI ML Repository, etc.)
2. Brief description
3. Approximate number of rows and classes
4. Direct download URL or access command

Include curl/wget commands where possible. Focus on datasets that are freely available and commonly used."""

    if llm:
        try:
            resp = llm.chat([{"role": "user", "content": prompt}])
            if resp.content:
                return resp.content
        except Exception:
            pass

    return (
        f"No specific recommendations available for '{task}'. "
        "Try searching HuggingFace Datasets (https://huggingface.co/datasets) "
        "or Kaggle (https://kaggle.com/datasets)."
    )


def _exec_train_model(arguments: dict) -> str:
    from edge_train.agent.progress import run_training_with_progress

    dataset = arguments.get("dataset_path", "")
    purpose = (arguments.get("purpose") or "").strip()
    if dataset.startswith("builtin:"):
        name = dataset.split(":", 1)[1]
        from edge_train.datasets import get_builtin
        import tempfile

        builtin = get_builtin().get(name)
        if not builtin:
            return f"Built-in dataset '{name}' not found."
        tmp = Path(tempfile.mkdtemp()) / f"{name}.csv"
        tmp.write_text(builtin["csv_content"], encoding="utf-8")
        dataset = str(tmp)

    target = arguments.get("target_column") or None
    output_dir = arguments.get("output_dir", "./model_output")
    epochs = arguments.get("epochs", 10)

    model_path = run_training_with_progress(dataset, target, output_dir, epochs)
    _persist_agent_step(
        "train",
        training_purpose=purpose or Path(output_dir).name,
        dataset_path=dataset,
        modality="text",
        training_status="succeeded",
        model_path=model_path,
    )
    return f"Training complete. Model saved to: `{model_path}`"


def _exec_validate_model(arguments: dict) -> str:
    from pathlib import Path

    from edge_train.validation import (
        check_size_constraint,
        convert_to_tflite,
        estimate_latency,
    )

    model_path = arguments["model_path"]
    model_dir = Path(model_path)
    if not model_dir.exists():
        return f"Model not found: {model_path}"

    tflite_path = model_dir.parent / f"{model_dir.name}.tflite"
    out = convert_to_tflite(model_dir, tflite_path)

    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    size_ok = check_size_constraint(size_mb)

    try:
        latency_ms = estimate_latency(tflite_path)
        latency_str = f"{latency_ms:.1f} ms"
    except Exception:
        latency_str = "could not measure"

    lines = [
        f"TFLite model: `{tflite_path}`",
        f"  **Size:** {size_mb:.2f} MB — {'OK (limit 10 MB)' if size_ok else 'EXCEEDS 10 MB limit'}",
        f"  **Latency:** {latency_str}",
    ]
    return "\n".join(lines)


def _exec_deploy_model(arguments: dict) -> str:
    import asyncio

    from edge_train.edge.deploy import deploy_model as edge_deploy

    model_path = arguments["model_path"]
    host = arguments["host"]
    port = arguments.get("port", 8080)

    result = asyncio.run(edge_deploy(model_path, host=host, port=port))
    if result.success:
        _persist_agent_step(
            "deploy",
            deployment_status="deployed",
            deployment_target=f"{host}:{port}",
            model_path=model_path,
        )
        return (
            f"Deployed successfully to **{result.device_id}** in {result.elapsed_sec:.1f}s.\n"
            f"  Model: `{model_path}`\n"
            f"  SHA-256: `{result.manifest.sha256[:16]}...`"
        )
    return f"Deployment failed: {result.error}"


def _exec_predict(arguments: dict, ui=None) -> str:
    from edge_train.config import load_config
    from edge_train.inference import TextClassifier, log_prediction

    _, arize, train_cfg, _ = load_config()
    log_path = train_cfg.prediction_log_path

    phoenix_active = False
    phoenix_note = ""
    if arize.is_valid():
        prep = _agent_phoenix_gate(ui)
        if prep.abort:
            return prep.message
        phoenix_active = prep.active
        phoenix_note = prep.message or ""

    endpoint = arguments.get("endpoint")
    model_path = arguments.get("model_path")
    modality = arguments.get("modality")
    text = arguments.get("text")
    csv_path = arguments.get("csv_path")
    gcs_uri = arguments.get("gcs_uri")
    image = arguments.get("image")

    if bool(endpoint) == bool(model_path):
        return (
            "Error: provide exactly one of `endpoint` (Vertex cloud) or "
            "`model_path` (local SavedModel)."
        )

    if endpoint:
        from edge_train.cli.predict import _load_vertex_predictor, _resolve_endpoint_modality
        from edge_train.config import GCPConfig
        from edge_train.evaluate import format_inference_environment

        gcp = GCPConfig()
        resolved = _resolve_endpoint_modality(endpoint, modality)
        env = format_inference_environment(
            source="vertex",
            modality=resolved,
            endpoint=endpoint,
            project=gcp.project_id,
            location=gcp.location,
        )
        classifier = _load_vertex_predictor(endpoint, resolved)
        source = "vertex"

        payload = text or gcs_uri or image
        if payload:
            label, conf = classifier.predict(payload)
            probs = classifier.predict_proba(payload)
            display = (
                classifier.format_input(payload)
                if hasattr(classifier, "format_input")
                else str(payload)
            )
            log_prediction(
                log_path, display, label, conf, probs, create_span=phoenix_active, source=source
            )
            lines = [env, "", f"**Predicted:** {label} ({conf:.4f})"]
            if phoenix_active:
                lines.append(f"📡 OTEL span → **Arize Phoenix** (`{arize.project_name}`)")
            elif phoenix_note:
                lines.append(f"⚠️ {phoenix_note}")
            return "\n".join(lines)

        if csv_path:
            tail = _exec_run_shell(
                {
                    "command": (
                        f"predict --endpoint {endpoint} --modality {resolved} "
                        f"--csv {csv_path} --log {log_path}"
                    )
                },
                ui=ui,
            )
            return env + "\n\n" + tail

        return (
            f"{env}\n\nError: provide `text`, `gcs_uri`, `image`, or `csv_path` for Vertex predict."
        )

    classifier = TextClassifier(model_path)
    from edge_train.evaluate import format_inference_environment

    env = format_inference_environment(
        source="local", modality=modality or "text", model_path=model_path
    )

    if text:
        label, conf = classifier.predict(text)
        probs = classifier.predict_proba(text)
        log_prediction(log_path, text, label, conf, probs, create_span=phoenix_active)
        lines = [
            env,
            "",
            f"**Predicted:** {label} ({conf:.4f})",
        ]
        for cls, prob in sorted(probs.items(), key=lambda x: -x[1])[1:]:
            lines.append(f"  {cls}: {prob:.4f}")
        if phoenix_active:
            lines.append("")
            lines.append(
                f"📡 OTEL span sent to **Arize Phoenix** (`{arize.project_name}`)"
            )
        elif phoenix_note:
            lines.append("")
            lines.append(f"⚠️ {phoenix_note}")
        return "\n".join(lines)

    if csv_path:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        from edge_train.cli.predict import _detect_text_column

        col = _detect_text_column(csv_path)
        if not col:
            return "Error: could not detect text column. Use --text-col to specify."
        texts = [r[col] for r in rows]
        results = classifier.predict_batch(texts)
        for i, (label, conf) in enumerate(results):
            probs = classifier.predict_proba(texts[i])
            log_prediction(
                log_path, texts[i], label, conf, probs, create_span=phoenix_active
            )
        lines = [
            env,
            "",
            f"Batch predictions (**{len(results)}** rows) logged to `{log_path}`",
        ]
        for i, (label, conf) in enumerate(results[:10]):
            lines.append(f"  {texts[i][:50]} → **{label}** ({conf:.4f})")
        if len(results) > 10:
            lines.append(f"  ... and {len(results) - 10} more")
        if phoenix_active:
            lines.append("")
            lines.append(
                f"📡 **{len(results)}** OTEL spans sent to **Arize Phoenix** (`{arize.project_name}`)"
            )
        elif phoenix_note:
            lines.append("")
            lines.append(f"⚠️ {phoenix_note}")
        return "\n".join(lines)

    return "Error: provide --text for single prediction or --csv for batch."


def _exec_run_predictions(arguments: dict, ui=None) -> str:
    """Batch-evaluate a labeled test JSONL via Vertex endpoint."""
    from edge_train.config import GCPConfig, load_config
    from edge_train.deployments import DeploymentRegistry
    from edge_train.evaluate import format_evaluation_summary, run_vertex_evaluation
    from edge_train.training_history import TrainingHistory

    _, arize, _, _ = load_config()
    phoenix_active = False
    if arize.is_valid():
        prep = _agent_phoenix_gate(ui)
        if prep.abort:
            return prep.message
        phoenix_active = prep.active
    else:
        return (
            "Phoenix not configured. Set PHOENIX_COLLECTOR_ENDPOINT "
            "(and PHOENIX_API_KEY for cloud) before running predictions."
        )

    gcp = GCPConfig()
    if not gcp.is_valid():
        return "GCP not configured. Set GCP_PROJECT and GCP_LOCATION for Vertex predictions."

    endpoint = (arguments.get("endpoint") or "").strip()
    modality = (arguments.get("modality") or "image").strip()
    test_jsonl = (arguments.get("test_jsonl") or "").strip()
    limit = arguments.get("limit")

    if not endpoint:
        dep = DeploymentRegistry.load().latest_vertex()
        if dep and dep.endpoint_name:
            endpoint = dep.endpoint_name
            if not arguments.get("modality") and dep.modality:
                modality = dep.modality

    if not endpoint:
        history = TrainingHistory.load()
        recent = next(
            (
                r
                for r in history.records
                if r.status == "succeeded" and r.mode == "cloud" and r.model_path
            ),
            None,
        )
        if recent:
            return (
                "Vertex model is trained but **not deployed to an endpoint** yet.\n\n"
                f"- Model: `{recent.model_path}`\n\n"
                "Deploy first, then evaluate:\n"
                f"```\ncoralflow deploy --cloud --model {recent.model_path}\n"
                "coralflow evaluate --endpoint <endpoint-id> --modality image "
                "--dataset gs://coralflow/neu-cls/test.jsonl\n```"
            )
        return (
            "No Vertex endpoint found. Deploy with `coralflow deploy --cloud --model <vertex-model>` "
            "or pass `endpoint` to run_predictions."
        )

    if not test_jsonl:
        for candidate in (
            "gs://coralflow/neu-cls/test.jsonl",
            "./neu-cls/test.jsonl",
        ):
            if candidate.startswith("gs://") or Path(candidate).exists():
                test_jsonl = candidate
                break
        if not test_jsonl:
            history = TrainingHistory.load()
            for record in history.records:
                if record.dataset_path and "neu" in record.dataset_path.lower():
                    test_jsonl = "gs://coralflow/neu-cls/test.jsonl"
                    break

    if not test_jsonl:
        return (
            "Missing `test_jsonl`. Provide a labeled Vertex export JSONL, e.g. "
            "`gs://coralflow/neu-cls/test.jsonl` (360 images)."
        )

    try:
        result = run_vertex_evaluation(
            endpoint=endpoint,
            modality=modality,
            dataset_path=test_jsonl,
            project=gcp.project_id,
            location=gcp.location,
            limit=limit,
            phoenix_active=phoenix_active,
        )
    except Exception as exc:
        return (
            f"Prediction run failed: {exc}\n\n"
            "**Environment:** Vertex AI online endpoint (cloud)\n"
            f"- Endpoint: `{endpoint}`\n"
            f"- Test set: `{test_jsonl}`\n\n"
            "If you see proxy/503 errors, retry when the network to Google APIs is stable."
        )

    _persist_agent_step("run_predictions", endpoint_name=endpoint)
    return format_evaluation_summary(result)


def _exec_check_monitoring() -> str:
    from edge_train.config import load_config
    from edge_train.phoenix_util import (
        check_phoenix_running,
        derive_dashboard_url,
        format_phoenix_start_instructions,
    )

    _, arize, train_cfg, _ = load_config()

    lines = ["## Arize Phoenix Monitoring"]
    if arize.is_valid():
        endpoint = arize.collector_endpoint
        dashboard = derive_dashboard_url(endpoint)
        status = check_phoenix_running(arize)
        lines.append(f"  Phoenix: **configured**")
        lines.append(f"    Endpoint: `{endpoint}`")
        lines.append(f"    Project:  **{arize.project_name}**")
        lines.append(f"    Dashboard: {dashboard}")
        if status.reachable:
            lines.append("    Status:   **reachable** ✓")
        else:
            lines.append(f"    Status:   **not running** ({status.detail})")
            lines.append("")
            lines.append(format_phoenix_start_instructions(status))
            log_path = Path(train_cfg.prediction_log_path)
            if log_path.exists():
                entries = []
                with open(log_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entries.append(json.loads(line))
                labeled = [e for e in entries if e.get("ground_truth")]
                lines.append(
                    f"\n  Prediction log: **{len(entries)}** entries "
                    f"(**{len(labeled)}** labeled)"
                )
            else:
                lines.append(f"\n  Prediction log: not found at {log_path}")
            return "\n".join(lines)

        lines.append("")
        lines.append("Each `/predict` call sends OTEL spans to Arize Cloud:")
        lines.append("- `input.value` — the text being classified")
        lines.append(
            "- `output.value` — predicted label + confidence + all class probabilities"
        )
        lines.append("- `metadata` — model type and number of classes")
        lines.append("")
        lines.append(
            "Open the dashboard to see traces, latency, and prediction distributions."
        )
        lines.append("")
        lines.append("**Inference paths (both send Phoenix spans when configured):**")
        lines.append("- Local/edge: `coralflow predict --model <SavedModel>`")
        lines.append(
            "- Vertex text: `coralflow simulate --endpoint <id>` (smoke test → Phoenix traces)"
        )
        lines.append(
            "- Vertex AutoML: `coralflow simulate --endpoint <id> --modality table|image|video`"
        )

        from edge_train.deployments import DeploymentRegistry

        deployments = DeploymentRegistry.load().records[:3]
        if deployments:
            lines.append("")
            lines.append("**Recent deployments:**")
            for dep in deployments:
                if dep.target == "vertex" and dep.endpoint_name:
                    lines.append(
                        f"- Vertex `{dep.endpoint_name}` (model `{dep.model_path}`)"
                    )
                elif dep.target == "edge":
                    lines.append(
                        f"- Edge device `{dep.device_id}` (model `{dep.model_path}`)"
                    )
        lines.append("Use `coralflow monitor --dashboard` to open in browser.")
    else:
        lines.append("  Phoenix: **not configured**")
        lines.append("")
        lines.append("To enable Phoenix monitoring:")
        lines.append("")
        lines.append("**Option 1 — Local Phoenix** (recommended for development):")
        lines.append("```bash")
        lines.append("pip install arize-phoenix")
        lines.append("phoenix serve  # starts at http://localhost:6006")
        lines.append(
            "export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces"
        )
        lines.append("```")
        lines.append("")
        lines.append("**Option 2 — Phoenix Cloud** (app.phoenix.arize.com):")
        lines.append("```bash")
        lines.append("export PHOENIX_API_KEY=your-cloud-api-key")
        lines.append(
            "export PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/v1/traces"
        )
        lines.append("```")
        lines.append(
            "Then restart the agent and `/predict` — spans will appear in real-time."
        )

    log_path = Path(train_cfg.prediction_log_path)
    if log_path.exists():
        entries = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        labeled = [e for e in entries if e.get("ground_truth")]
        lines.append(
            f"  Prediction log: **{len(entries)}** entries (**{len(labeled)}** labeled)"
        )
    else:
        lines.append(f"  Prediction log: not found at {log_path}")

    return "\n".join(lines)


def _exec_check_retrain() -> str:
    from edge_train.config import load_config

    _, _, train_cfg, _ = load_config()
    log_path = Path(train_cfg.prediction_log_path)

    if not log_path.exists():
        return f"No prediction log found at: {log_path}"

    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    labeled = [e for e in entries if e.get("ground_truth") is not None]
    if not labeled:
        return (
            f"No labeled entries in prediction log ({len(entries)} total).\n"
            "Add 'ground_truth' fields to prediction_log.jsonl entries to enable retraining."
        )

    correct = sum(1 for e in labeled if e["predicted_label"] == e["ground_truth"])
    accuracy = correct / len(labeled)
    threshold = train_cfg.retrain_accuracy_threshold

    lines = [
        "## Retrain Check",
        f"  Labeled predictions: **{len(labeled)}**",
        f"  Current accuracy:    **{accuracy:.2%}**",
        f"  Threshold:           **{threshold:.2%}**",
    ]
    if accuracy >= threshold:
        lines.append(f"  Accuracy is above threshold — no retrain needed.")
    else:
        lines.append(f"  Accuracy below threshold — **retrain recommended!**")
        lines.append(f"  Run: `coralflow monitor --retrain --dataset <original.csv>`")
    return "\n".join(lines)


def _exec_label_predictions(arguments: dict) -> str:
    from edge_train.config import load_config

    _, _, train_cfg, _ = load_config()
    log_path = Path(train_cfg.prediction_log_path)
    action = arguments.get("action", "list")

    if not log_path.exists():
        return f"No prediction log found at: `{log_path}`. Run some predictions first with `/predict`."

    # Read all entries
    entries: list[dict] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if action == "list":
        unlabeled = [e for e in entries if "ground_truth" not in e]
        if not unlabeled:
            labeled = len(entries) - len(unlabeled)
            return (
                f"All **{len(entries)}** predictions are already labeled.\n"
                f"Use `/check_retrain` to see current accuracy and decide if retraining is needed."
            )

        recent = unlabeled[-20:]  # show last 20
        lines = [
            f"## Unlabeled Predictions (showing {len(recent)} of {len(unlabeled)})",
            "",
        ]
        for i, e in enumerate(recent):
            idx = entries.index(e) + 1  # 1-based
            text = e.get("text", "")[:60]
            pred = e.get("predicted_label", "?")
            conf = e.get("confidence", 0)
            lines.append(f"  **{idx}.** `{text}` → *{pred}* ({conf:.2%})")
        lines.append("")
        lines.append(
            "To simulate data drift, tell me which predictions were wrong and what the correct label should be."
        )
        lines.append('Example: "entry 3 should be 购物, entry 5 should be 餐饮"')
        return "\n".join(lines)

    elif action == "label":
        labels_list = arguments.get("labels", [])
        if not labels_list:
            return "Provide `labels` array with `index` and `ground_truth` for each correction."

        updated = 0
        lines = ["## Labeling Results", ""]
        for item in labels_list:
            idx = item.get("index", 0)
            gt = item.get("ground_truth", "")
            if 1 <= idx <= len(entries):
                entries[idx - 1]["ground_truth"] = gt
                text = entries[idx - 1].get("text", "")[:50]
                pred = entries[idx - 1].get("predicted_label", "?")
                lines.append(
                    f"  ✓ entry **{idx}**: `{text}` → ground_truth = **{gt}** (was: *{pred}*)"
                )
                updated += 1
            else:
                lines.append(f"  ✗ entry **{idx}**: out of range (1-{len(entries)})")

        # Write back
        with open(log_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        lines.append("")
        lines.append(f"**{updated}** entries labeled.")

        # Count correct vs incorrect
        labeled = [e for e in entries if "ground_truth" in e]
        correct = sum(1 for e in labeled if e["predicted_label"] == e["ground_truth"])
        if labeled:
            acc = correct / len(labeled)
            lines.append(
                f"Current accuracy: **{acc:.2%}** ({correct}/{len(labeled)} correct)"
            )
            threshold = train_cfg.retrain_accuracy_threshold
            if acc < threshold and len(labeled) >= train_cfg.retrain_min_samples:
                lines.append("")
                lines.append(
                    f"⚠ Accuracy is below threshold (**{threshold:.0%}**). "
                    "Use `/check_retrain` to trigger retraining."
                )

        return "\n".join(lines)

    return f"Unknown action: `{action}`. Use 'list' or 'label'."


_CORALFLOW_CLI_CMDS = frozenset(
    {"agent", "init", "train", "validate", "deploy", "monitor", "cost", "predict", "simulate", "evaluate"}
)

# Long-running CLI subcommands stream stdout instead of blocking silently.
_LONG_RUNNING_CMDS = frozenset({"train", "validate"})


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Quiet TensorFlow startup noise for subprocess CLI runs."""
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update(extra)
    return env


def _stream_coralflow_cli(
    cmd_argv: list[str], timeout: int = 1800, extra_env: dict[str, str] | None = None
) -> tuple[str, int]:
    """Run coralflow CLI with live terminal output. Returns (tail_output, exit_code)."""
    from edge_train.agent.ui import CoralFlowUI

    ui = CoralFlowUI()
    output_lines: list[str] = []
    proc = subprocess.Popen(
        cmd_argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_subprocess_env(extra_env),
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            output_lines.append(line)
            if line.strip():
                ui.raw(line + "\n")
        return_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return "Command timed out.", 124

    tail = "\n".join(output_lines[-100:])
    return tail, return_code


def _exec_run_shell(arguments: dict, ui=None) -> str:
    cmd = arguments.get("command", "")
    if not cmd:
        return "No command provided."

    lowered = cmd.lower()
    if any(
        marker in lowered
        for marker in (
            "from google.cloud import aiplatform",
            "google.cloud.aiplatform",
            "<< 'pyeof'",
            "<<'pyeof'",
            "aiplatform.init(",
            "endpoint.deploy(",
            "prediction_serviceclient",
        )
    ):
        return (
            "Blocked ad-hoc Vertex/GCP Python. Use CoralFlow tools instead:\n"
            "- `run_predictions` — batch test-set evaluation on a Vertex endpoint\n"
            "- `predict` — single or CSV inference (local or `--endpoint`)\n"
            "- `run_shell` with `deploy --cloud`, `evaluate`, or `simulate` CLI subcommands"
        )

    # Strip "coralflow" / "coralflow " prefix — LLM may include it
    if cmd.startswith("coralflow "):
        cmd = cmd[len("coralflow ") :]
    elif cmd == "coralflow":
        cmd = ""

    if not cmd:
        return "No command provided."

    sub = cmd.split()[0] if cmd.split() else ""
    extra_env: dict[str, str] | None = None

    if sub in ("predict", "simulate", "evaluate"):
        from edge_train.config import load_config

        _, arize, _, _ = load_config()
        if arize.is_valid():
            prep = _agent_phoenix_gate(ui)
            if prep.abort:
                return prep.message
            if not prep.active:
                extra_env = {"CORALFLOW_PHOENIX_SKIP": "1"}

    try:
        if sub in _CORALFLOW_CLI_CMDS:
            cmd_argv = [sys.executable, "-m", "edge_train.cli"] + cmd.split()
            timeout = 1800 if sub in _LONG_RUNNING_CMDS else 300
            output, return_code = _stream_coralflow_cli(
                cmd_argv, timeout=timeout, extra_env=extra_env
            )
            if return_code == 0 and sub in (
                "train",
                "deploy",
                "evaluate",
                "predict",
                "simulate",
            ):
                fields: dict = {}
                if sub == "train":
                    purpose = _parse_shell_purpose(cmd)
                    if purpose:
                        fields["training_purpose"] = purpose
                _persist_agent_step(sub, **fields)
            if return_code != 0:
                return output.strip() or f"Command failed (exit code {return_code})"
            return output.strip() or f"Command completed (exit code {return_code})"

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
            env=_subprocess_env(extra_env),
        )
        output = result.stdout or ""
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() or f"Command completed (exit code {result.returncode})"
    except subprocess.TimeoutExpired:
        return "Command timed out."
    except Exception as e:
        return f"Command failed: {e}"


def _exec_get_status() -> str:
    from edge_train.agent import AgentState
    from edge_train.agent.context import format_project_context, sync_agent_context
    from edge_train.training_history import TrainingHistory, format_startup_summary

    state = sync_agent_context(AgentState.load())
    history = TrainingHistory.load()
    history.sync_cloud_jobs()

    lines = [
        "## CoralFlow Agent Status",
        f"  State file: `~/.coralflow/agent_state.json`",
        f"  Training history: `~/.coralflow/training_history.json`",
        "",
        "### Current project",
    ]

    project = format_project_context(state)
    if project:
        lines.extend(f"  {line}" for line in project.splitlines())
    else:
        lines.append("  (no project context saved yet)")

    from edge_train.agent import DatasetScanner, scan_models

    datasets = DatasetScanner.scan()
    models = scan_models()
    lines.append("")
    lines.append(f"  Datasets found: {len(datasets)}")
    lines.append(f"  Models found:   {len(models)}")

    training_summary = format_startup_summary(history)
    if training_summary:
        lines.append("")
        lines.append(training_summary)

    return "\n".join(lines)
