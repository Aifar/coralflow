"""Batch evaluation against labeled test sets (Vertex JSONL / CSV)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_BATCH_SIZE = 16


@dataclass
class EvaluationResult:
    total: int
    labeled: int
    correct: int
    accuracy: float
    log_path: str
    phoenix_active: bool
    environment: str
    per_class: dict[str, dict[str, int]]


def format_inference_environment(
    *,
    source: str,
    modality: str,
    endpoint: str | None = None,
    model_path: str | None = None,
    project: str | None = None,
    location: str | None = None,
) -> str:
    """Human-readable description of where inference runs."""
    if source == "vertex":
        lines = [
            "**Inference environment:** Google Cloud Vertex AI (online endpoint)",
            f"- **Project:** `{project or '?'}`",
            f"- **Region:** `{location or 'us-central1'}`",
            f"- **Endpoint:** `{endpoint or '?'}`",
            f"- **Modality:** {modality}",
            "- Predictions execute on Vertex managed compute (not this machine's CPU/GPU).",
        ]
    elif source == "edge":
        lines = [
            "**Inference environment:** Edge device (TFLite over HTTP)",
            f"- **Model:** `{model_path or '?'}`",
            f"- **Modality:** {modality}",
        ]
    else:
        lines = [
            "**Inference environment:** Local (TensorFlow SavedModel on this machine)",
            f"- **Model:** `{model_path or '?'}`",
            f"- **Modality:** {modality}",
        ]
    lines.append(
        "- Results append to `prediction_log.jsonl`; OTEL spans go to **Arize Phoenix** when configured."
    )
    return "\n".join(lines)


def _parse_gs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Not a GCS URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load JSONL from a local path or gs:// URI."""
    text = _read_text_blob(path)
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_text_blob(path: str) -> str:
    if path.startswith("gs://"):
        from google.cloud import storage

        bucket_name, blob_name = _parse_gs_uri(path)
        client = storage.Client()
        blob = client.bucket(bucket_name).blob(blob_name)
        return blob.download_as_text(encoding="utf-8")
    return Path(path).read_text(encoding="utf-8")


def parse_labeled_sample(record: dict[str, Any]) -> tuple[Any, str | None]:
    """Extract (model input, ground_truth) from Vertex export JSONL or simple rows."""
    ground_truth = None
    ann = record.get("classificationAnnotation") or record.get(
        "classification_annotation"
    )
    if isinstance(ann, dict):
        ground_truth = ann.get("displayName") or ann.get("display_name")
    if ground_truth is None:
        for key in ("label", "ground_truth", "target", "class"):
            if record.get(key):
                ground_truth = str(record[key])
                break

    if "imageGcsUri" in record:
        return record["imageGcsUri"], ground_truth
    if "gcsUri" in record:
        return record["gcsUri"], ground_truth
    if "textContent" in record:
        return record["textContent"], ground_truth
    if "text" in record:
        return record["text"], ground_truth
    if "content" in record and isinstance(record["content"], str):
        return record["content"], ground_truth

    raise ValueError(f"Unsupported test record keys: {sorted(record.keys())}")


def _update_class_stats(
    stats: dict[str, dict[str, int]], truth: str, predicted: str
) -> None:
    bucket = stats.setdefault(truth, {"total": 0, "correct": 0})
    bucket["total"] += 1
    if truth == predicted:
        bucket["correct"] += 1


def run_vertex_evaluation(
    *,
    endpoint: str,
    modality: str,
    dataset_path: str,
    project: str,
    location: str,
    log_path: str | None = None,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    phoenix_active: bool | None = None,
) -> EvaluationResult:
    """Run batch predictions on a labeled JSONL test set via a Vertex endpoint."""
    from edge_train.cli.predict import (
        _load_vertex_predictor,
        _resolve_endpoint_modality,
    )
    from edge_train.config import load_config
    from edge_train.inference import log_prediction
    from edge_train.inference.phoenix import prepare_phoenix_for_inference
    from edge_train.phoenix_util import derive_dashboard_url

    _, arize, train_cfg, _ = load_config()
    log_file = log_path or train_cfg.prediction_log_path
    resolved_modality = _resolve_endpoint_modality(endpoint, modality)

    if phoenix_active is None:
        if not arize.is_valid():
            raise RuntimeError(
                "Phoenix not configured. Set PHOENIX_COLLECTOR_ENDPOINT "
                "(and PHOENIX_API_KEY for Phoenix Cloud) before evaluate."
            )
        prep = prepare_phoenix_for_inference(required=True, interactive=False)
        if prep.abort:
            raise RuntimeError(prep.message)
        phoenix_active = prep.active
    elif phoenix_active and not arize.is_valid():
        raise RuntimeError(
            "Phoenix not configured. Set PHOENIX_COLLECTOR_ENDPOINT "
            "(and PHOENIX_API_KEY for Phoenix Cloud) before evaluate."
        )

    environment = format_inference_environment(
        source="vertex",
        modality=resolved_modality,
        endpoint=endpoint,
        project=project,
        location=location,
    )

    classifier = _load_vertex_predictor(endpoint, resolved_modality)
    records = load_jsonl(dataset_path)
    if limit is not None:
        records = records[:limit]

    samples: list[tuple[Any, str | None]] = []
    for row in records:
        try:
            samples.append(parse_labeled_sample(row))
        except ValueError:
            continue

    if not samples:
        raise ValueError(f"No usable labeled rows in {dataset_path}")

    correct = 0
    labeled = 0
    per_class: dict[str, dict[str, int]] = {}

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        payloads = [p for p, _ in chunk]
        truths = [t for _, t in chunk]
        results = classifier.predict_batch(payloads)
        for i, (label, conf) in enumerate(results):
            payload = payloads[i]
            truth = truths[i]
            probs = classifier.predict_proba(payload)
            display = (
                classifier.format_input(payload)
                if hasattr(classifier, "format_input")
                else str(payload)
            )
            log_prediction(
                log_file,
                display,
                label,
                conf,
                probs,
                ground_truth=truth,
                create_span=phoenix_active,
                source="vertex",
            )
            if truth:
                labeled += 1
                if label == truth:
                    correct += 1
                _update_class_stats(per_class, truth, label)

    accuracy = (correct / labeled) if labeled else 0.0
    _ = derive_dashboard_url(arize.collector_endpoint)

    return EvaluationResult(
        total=len(samples),
        labeled=labeled,
        correct=correct,
        accuracy=accuracy,
        log_path=log_file,
        phoenix_active=phoenix_active,
        environment=environment,
        per_class=per_class,
    )


def format_evaluation_summary(result: EvaluationResult) -> str:
    lines = [
        result.environment,
        "",
        f"**Test set:** {result.total} samples evaluated",
        f"**Accuracy:** {result.correct}/{result.labeled} correct "
        f"→ **{result.accuracy:.1%}**",
        f"**Log:** `{result.log_path}`",
    ]
    if result.phoenix_active:
        lines.append("**Phoenix:** OTEL spans sent for each prediction.")
    if result.per_class:
        lines.append("")
        lines.append("### Per-class accuracy")
        for cls in sorted(result.per_class):
            stats = result.per_class[cls]
            total = stats["total"]
            acc = stats["correct"] / total if total else 0.0
            lines.append(f"- **{cls}:** {stats['correct']}/{total} ({acc:.1%})")
    return "\n".join(lines)
