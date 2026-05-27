"""Vertex AI Gemini supervised fine-tuning for text classification."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any


def is_finetune_job(job_name: str) -> bool:
    return "/tuningJobs/" in job_name


def convert_csv_to_jsonl(
    csv_path: str,
    target_column: str | None = None,
) -> tuple[str, list[str]]:
    """Convert a text classification CSV to Gemini SFT JSONL."""
    import csv

    from edge_train.trainer import _detect_columns

    text_col, label_col = _detect_columns(csv_path)
    if target_column:
        label_col = target_column

    classes: set[str] = set()
    examples: list[tuple[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get(text_col) or "").strip()
            label = (row.get(label_col) or "").strip()
            if not text or not label:
                continue
            classes.add(label)
            examples.append((text, label))

    if not examples:
        raise ValueError(f"No training rows found in {csv_path}")

    class_list = sorted(classes)
    system_text = (
        "Classify the input text. "
        f"Reply with exactly one label from: {', '.join(class_list)}. "
        "Output only the label."
    )

    records = []
    for text, label in examples:
        records.append(
            {
                "systemInstruction": {
                    "role": "system",
                    "parts": [{"text": system_text}],
                },
                "contents": [
                    {"role": "user", "parts": [{"text": text}]},
                    {"role": "model", "parts": [{"text": label}]},
                ],
            }
        )

    tmp = Path(tempfile.mkdtemp(prefix="coralflow-sft-")) / "train.jsonl"
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return str(tmp), class_list


def submit_finetune_job(
    project: str,
    location: str,
    dataset_path: str,
    staging_bucket: str,
    target_column: str | None = None,
    base_model: str = "gemini-2.0-flash-001",
) -> str:
    """Submit a Gemini supervised fine-tuning job. Returns tuning job resource name."""
    import vertexai
    from vertexai.tuning import sft

    from edge_train.cloud import _upload_to_gcs

    jsonl_path, classes = convert_csv_to_jsonl(dataset_path, target_column)
    train_uri = _upload_to_gcs(
        jsonl_path, project, staging_bucket, blob_prefix="finetune"
    )

    from edge_train.cloud.publisher_models import resolve_finetune_base_model

    source_model = resolve_finetune_base_model(base_model, project, location)

    vertexai.init(project=project, location=location, staging_bucket=staging_bucket)
    job = sft.train(
        source_model=source_model,
        train_dataset=train_uri,
        tuned_model_display_name=f"coralflow-{int(time.time())}",
    )
    # Attach metadata for callers (not persisted by Vertex)
    job._coralflow_classes = classes  # type: ignore[attr-defined]
    return job.resource_name


def poll_finetune_job(job_name: str, deadline: float) -> dict[str, Any]:
    """Poll a Gemini SFT job until completion or timeout."""
    from vertexai.tuning import sft

    job = sft.SupervisedTuningJob(job_name)
    while time.time() < deadline:
        job.refresh()
        if job.has_ended:
            if job.has_succeeded:
                return {
                    "job_name": job.resource_name,
                    "model_path": job.tuned_model_name
                    or job.tuned_model_endpoint_name
                    or "",
                    "accuracy": 0.0,
                }
            detail = job.error or job.state
            raise RuntimeError(f"Fine-tuning job failed: {detail}")
        time.sleep(30)

    raise TimeoutError("Fine-tuning job did not complete within deadline")
