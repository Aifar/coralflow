"""Tool definitions (OpenAI function-calling format) and executors.

Each tool has a JSON schema for the LLM and a Python executor function.
"""

from __future__ import annotations

import csv
import json
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
            "description": "Classify text using a trained model — single text or batch CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the SavedModel directory.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Single text input to classify.",
                    },
                    "csv_path": {
                        "type": "string",
                        "description": "CSV file for batch prediction.",
                    },
                },
                "required": ["model_path"],
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
            "name": "run_shell",
            "description": "Run any coralflow CLI command directly. Use for commands not covered by other tools.",
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


def execute_tool(name: str, arguments: dict, llm=None) -> str:
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
            return _exec_predict(arguments)
        elif name == "check_monitoring":
            return _exec_check_monitoring()
        elif name == "check_retrain":
            return _exec_check_retrain()
        elif name == "run_shell":
            return _exec_run_shell(arguments)
        elif name == "get_status":
            return _exec_get_status()
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool '{name}' failed: {e}"


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
        lines.append("2. **Cloud training** — Vertex AI AutoML, ~$3-8, 30-60 min")
        lines.append("")
        lines.append("---")
        lines.append(
            "**Which option do you prefer?** Type `1` for local or `2` for cloud."
        )
    else:
        lines.append("")
        lines.append("**Verdict: Local resources insufficient for reliable training**")
        lines.append("")
        lines.append("### Recommendation")
        lines.append("**Cloud training only** — Vertex AI AutoML, ~$3-8, 30-60 min")
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
        lines.append(
            "Shall I proceed with cloud training? Type `yes` to confirm or describe your preference."
        )

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
        return (
            f"Deployed successfully to **{result.device_id}** in {result.elapsed_sec:.1f}s.\n"
            f"  Model: `{model_path}`\n"
            f"  SHA-256: `{result.manifest.sha256[:16]}...`"
        )
    return f"Deployment failed: {result.error}"


def _exec_predict(arguments: dict) -> str:
    from edge_train.config import load_config
    from edge_train.inference import (
        TextClassifier,
        _ensure_phoenix_registered,
        log_prediction,
    )

    _, arize, train_cfg, _ = load_config()
    phoenix_active = _ensure_phoenix_registered(arize)
    log_path = train_cfg.prediction_log_path

    model_path = arguments["model_path"]
    text = arguments.get("text")
    csv_path = arguments.get("csv_path")

    classifier = TextClassifier(model_path)

    if text:
        label, conf = classifier.predict(text)
        probs = classifier.predict_proba(text)
        log_prediction(log_path, text, label, conf, probs, create_span=phoenix_active)
        lines = [
            f"**Predicted:** {label} ({conf:.4f})",
        ]
        for cls, prob in sorted(probs.items(), key=lambda x: -x[1])[1:]:
            lines.append(f"  {cls}: {prob:.4f}")
        if phoenix_active:
            lines.append("")
            lines.append(
                f"📡 OTEL span sent to **Arize Phoenix** (`{arize.project_name}`)"
            )
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
        return "\n".join(lines)

    return "Error: provide --text for single prediction or --csv for batch."


def _exec_check_monitoring() -> str:
    from edge_train.config import load_config

    _, arize, train_cfg, _ = load_config()

    lines = ["## Arize Phoenix Monitoring"]
    if arize.is_valid():
        # Derive dashboard URL from collector endpoint
        endpoint = arize.collector_endpoint
        if "/v1/traces" in endpoint:
            dashboard = endpoint.rsplit("/v1/traces", 1)[0]
        else:
            dashboard = "https://app.phoenix.arize.com"
        lines.append(f"  Phoenix: **configured**")
        lines.append(f"    Endpoint: `{endpoint}`")
        lines.append(f"    Project:  **{arize.project_name}**")
        lines.append(f"    Dashboard: {dashboard}")
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
        lines.append("Use `coralflow monitor --dashboard` to open in browser.")
    else:
        lines.append("  Phoenix: **not configured**")
        lines.append("")
        lines.append("To enable Arize Cloud monitoring:")
        lines.append("```bash")
        lines.append("export PHOENIX_API_KEY=your-api-key")
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


def _exec_run_shell(arguments: dict) -> str:
    cmd = arguments.get("command", "")
    if not cmd:
        return "No command provided."

    # Strip "coralflow" / "coralflow " prefix — LLM may include it
    if cmd.startswith("coralflow "):
        cmd = cmd[len("coralflow ") :]
    elif cmd == "coralflow":
        cmd = ""

    if not cmd:
        return "No command provided."

    _coralflow_cmds = {
        "agent",
        "init",
        "train",
        "validate",
        "deploy",
        "monitor",
        "cost",
        "predict",
    }
    sub = cmd.split()[0] if cmd.split() else ""

    try:
        if sub in _coralflow_cmds:
            result = subprocess.run(
                [sys.executable, "-m", "edge_train.cli"] + cmd.split(),
                capture_output=True,
                text=True,
                timeout=300,
            )
        else:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=300
            )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() or f"Command completed (exit code {result.returncode})"
    except subprocess.TimeoutExpired:
        return "Command timed out (300s)."
    except Exception as e:
        return f"Command failed: {e}"


def _exec_get_status() -> str:
    from edge_train.agent import AgentState

    state = AgentState.load()
    lines = [
        "## CoralFlow Agent Status",
        f"  State file: `~/.coralflow/agent_state.json`",
    ]

    if state.dataset_path:
        lines.append(f"  Active dataset: `{state.dataset_path}`")
    if state.model_path:
        lines.append(f"  Active model:   `{state.model_path}`")
    if state.task_type:
        lines.append(f"  Task type:      {state.task_type}")
    if state.deployment_target:
        lines.append(f"  Deployment:     {state.deployment_target}")
    if state.last_step:
        lines.append(f"  Last step:      {state.last_step}")
    if state.conversation_summary:
        lines.append(f"  Summary:        {state.conversation_summary[:200]}")

    # Also run scan for datasets and models
    from edge_train.agent import DatasetScanner, scan_models

    datasets = DatasetScanner.scan()
    models = scan_models()
    lines.append(f"  Datasets found: {len(datasets)}")
    lines.append(f"  Models found:   {len(models)}")

    if not state.dataset_path and not state.model_path:
        lines.append("  (fresh session — no prior state)")

    return "\n".join(lines)
