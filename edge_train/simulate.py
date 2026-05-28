"""Smoke-test inference with sample inputs and Phoenix OTEL spans."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_train.datasets import resolve_dataset_path

DEFAULT_TEXT_SAMPLES = [
    "服务器挂了快来看看",
    "明天下午三点开会",
    "快递到了放门口",
    "双十一大促开始了",
    "美团外卖 35元",
    "滴滴出行 12元",
    "项目进度更新",
    "数据库连接失败",
]

DEFAULT_TABULAR_SAMPLES: list[dict[str, Any]] = [
    {
        "age": 25,
        "tenure": 12,
        "MonthlyCharges": 53.85,
        "TotalCharges": 646.2,
    },
    {
        "age": 42,
        "tenure": 24,
        "MonthlyCharges": 79.10,
        "TotalCharges": 1898.4,
    },
    {
        "age": 33,
        "tenure": 3,
        "MonthlyCharges": 29.60,
        "TotalCharges": 88.8,
    },
    {
        "age": 58,
        "tenure": 48,
        "MonthlyCharges": 104.80,
        "TotalCharges": 5032.0,
    },
    {
        "age": 19,
        "tenure": 1,
        "MonthlyCharges": 20.15,
        "TotalCharges": 20.15,
    },
]


@dataclass
class SimulationResult:
    count: int
    log_path: str
    phoenix_active: bool
    dashboard_url: str
    project_name: str


def format_simulate_command(
    *,
    endpoint: str = "",
    model: str = "",
    modality: str = "text",
    count: int = 5,
) -> str:
    """Return a copy-paste coralflow simulate command for post-deploy hints."""
    parts = ["coralflow simulate"]
    if endpoint:
        parts.extend(["--endpoint", endpoint])
    elif model:
        parts.extend(["--model", model])
    if endpoint and modality and modality != "text":
        parts.extend(["--modality", modality])
    if count != 5:
        parts.extend(["--count", str(count)])
    return " ".join(parts)


def _text_samples_from_dataset(dataset_path: str, count: int) -> list[str]:
    path, _ = resolve_dataset_path(dataset_path)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
    text_col = next(
        (
            h
            for h in headers
            if h.lower().strip()
            in {"text", "message", "content", "sentence", "review", "comment"}
        ),
        headers[0] if headers else None,
    )
    if not text_col:
        return DEFAULT_TEXT_SAMPLES[:count]

    samples: list[str] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = (row.get(text_col) or "").strip()
            if val:
                samples.append(val)
            if len(samples) >= count:
                break
    return samples or DEFAULT_TEXT_SAMPLES[:count]


def _tabular_samples_from_dataset(
    dataset_path: str, count: int
) -> list[dict[str, Any]]:
    path, _ = resolve_dataset_path(dataset_path)
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feature_row = {
                k: _coerce_feature(v)
                for k, v in row.items()
                if k.lower() not in {"label", "target", "class", "category", "urgency"}
            }
            if feature_row:
                rows.append(feature_row)
            if len(rows) >= count:
                break
    return rows or DEFAULT_TABULAR_SAMPLES[:count]


def _coerce_feature(value: str) -> Any:
    text = (value or "").strip()
    if not text:
        return text
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def get_simulation_samples(
    modality: str,
    *,
    count: int = 5,
    dataset: str | None = None,
    image: str | None = None,
    gcs_uri: str | None = None,
) -> list[Any]:
    mod = modality.lower().strip()
    if mod == "text":
        if dataset:
            return _text_samples_from_dataset(dataset, count)
        return DEFAULT_TEXT_SAMPLES[:count]

    if mod == "table":
        if dataset:
            return _tabular_samples_from_dataset(dataset, count)
        return DEFAULT_TABULAR_SAMPLES[:count]

    if mod == "image":
        if image:
            return [image]
        if gcs_uri:
            return [gcs_uri]
        raise ValueError(
            "Image simulate needs --image or --gcs-uri (or --dataset CSV with image paths)."
        )

    if mod == "video":
        if gcs_uri:
            return [gcs_uri]
        raise ValueError(
            "Video simulate needs --gcs-uri (or --dataset CSV with GCS video URIs)."
        )

    raise ValueError(f"Unsupported modality for simulate: {modality}")


def run_simulation(
    *,
    model: str | None = None,
    endpoint: str | None = None,
    modality: str | None = None,
    count: int = 5,
    dataset: str | None = None,
    image: str | None = None,
    gcs_uri: str | None = None,
    log_path: str | None = None,
) -> SimulationResult:
    """Run sample predictions and emit Phoenix OTEL spans for each."""
    from edge_train.config import load_config
    from edge_train.inference import TextClassifier, log_prediction
    from edge_train.inference.phoenix import prepare_phoenix_for_inference
    from edge_train.phoenix_util import derive_dashboard_url

    if bool(model) == bool(endpoint):
        raise ValueError("Provide exactly one of model (local) or endpoint (Vertex).")

    _, arize, train_cfg, _ = load_config()
    log_file = log_path or train_cfg.prediction_log_path

    phoenix_active = False
    if arize.is_valid():
        phoenix_active, phoenix_err = prepare_phoenix_for_inference(required=True)
        if not phoenix_active:
            raise RuntimeError(phoenix_err)
    else:
        raise RuntimeError(
            "Phoenix not configured. Set PHOENIX_COLLECTOR_ENDPOINT "
            "(and PHOENIX_API_KEY for Phoenix Cloud) before simulate."
        )

    source = "vertex" if endpoint else "local"
    resolved_modality = modality or "text"

    if endpoint:
        from edge_train.cli.predict import (
            _load_vertex_predictor,
            _resolve_endpoint_modality,
        )

        resolved_modality = _resolve_endpoint_modality(endpoint, modality)
        classifier = _load_vertex_predictor(endpoint, resolved_modality)
    else:
        classifier = TextClassifier(model)
        resolved_modality = modality or "text"

    samples = get_simulation_samples(
        resolved_modality,
        count=count,
        dataset=dataset,
        image=image,
        gcs_uri=gcs_uri,
    )

    for payload in samples:
        label, conf = classifier.predict(payload)
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
            create_span=phoenix_active,
            source=source,
        )

    dashboard = derive_dashboard_url(arize.collector_endpoint)
    return SimulationResult(
        count=len(samples),
        log_path=log_file,
        phoenix_active=phoenix_active,
        dashboard_url=dashboard,
        project_name=arize.project_name,
    )


def guess_local_saved_model() -> str | None:
    """Best-effort SavedModel path for edge deploy simulate hints."""
    from edge_train.training_history import TrainingHistory

    for record in TrainingHistory.load().records:
        if (
            record.mode == "local"
            and record.status == "succeeded"
            and record.model_path
        ):
            path = Path(record.model_path)
            if path.is_dir():
                return str(path)
    return None
