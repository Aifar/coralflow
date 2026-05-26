"""Shared prediction-log accuracy checks and retrain merge logic."""

from __future__ import annotations

import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path


def read_prediction_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def labeled_entries(entries: list[dict]) -> list[dict]:
    return [e for e in entries if e.get("ground_truth") is not None]


def compute_accuracy(entries: list[dict]) -> tuple[float, int, int]:
    """Return (accuracy, correct, total) for entries with ground_truth."""
    labeled = labeled_entries(entries)
    if not labeled:
        return 0.0, 0, 0
    correct = sum(1 for e in labeled if e["predicted_label"] == e["ground_truth"])
    return correct / len(labeled), correct, len(labeled)


def evaluate_cases(
    classifier, cases: list[tuple[str, str]]
) -> tuple[float, list[dict]]:
    """Run classifier on (text, ground_truth) pairs; return accuracy and row details."""
    details: list[dict] = []
    correct = 0
    for text, truth in cases:
        pred, conf = classifier.predict(text)
        ok = pred == truth
        if ok:
            correct += 1
        details.append(
            {
                "text": text,
                "ground_truth": truth,
                "predicted_label": pred,
                "confidence": round(conf, 4),
                "correct": ok,
            }
        )
    acc = correct / len(cases) if cases else 0.0
    return acc, details


def write_prediction_log(
    log_path: Path,
    cases: list[tuple[str, str]],
    classifier,
    *,
    clear: bool = True,
) -> list[dict]:
    """Predict each case, append JSONL with ground_truth set."""
    from edge_train.inference import log_prediction

    if clear and log_path.exists():
        log_path.unlink()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for text, truth in cases:
        pred, conf = classifier.predict(text)
        probs = classifier.predict_proba(text)
        log_prediction(
            str(log_path),
            text,
            pred,
            conf,
            probs,
            ground_truth=truth,
            create_span=False,
        )
        entries.append(
            {
                "text": text,
                "predicted_label": pred,
                "ground_truth": truth,
                "confidence": round(conf, 4),
            }
        )
    return entries


def merge_labeled_csv(labeled_entries: list[dict], dataset_path: str) -> Path:
    """Merge labeled log rows into a copy of the training CSV."""
    merged_path = Path(tempfile.mkdtemp()) / "merged.csv"
    with open(dataset_path, encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        headers = list(reader.fieldnames or [])
        rows = list(reader)

    from edge_train.trainer import _resolve_columns

    text_col, label_col = _resolve_columns(dataset_path, None)

    for e in labeled_entries:
        row = {h: "" for h in headers}
        row[text_col] = e["text"]
        row[label_col] = e["ground_truth"]
        rows.append(row)

    with open(merged_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return merged_path


def retrain_from_labeled(
    labeled: list[dict],
    dataset_path: str,
    output_dir: str,
    epochs: int,
    *,
    oversample: int = 1,
) -> Path:
    """Merge labeled predictions into dataset and train a new model."""
    from edge_train.trainer import train_text_classifier, _resolve_columns

    if oversample > 1:
        expanded: list[dict] = []
        for entry in labeled:
            for _ in range(oversample):
                expanded.append(entry)
        labeled = expanded

    merged_path = merge_labeled_csv(labeled, dataset_path)
    _, label_col = _resolve_columns(dataset_path, None)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(output_dir)
    retrain_out = base.parent / f"{base.name}_retrained_{ts}"

    return train_text_classifier(
        dataset_path=str(merged_path),
        target_column=label_col,
        output_dir=str(retrain_out),
        epochs=epochs,
    )


def filter_csv_by_labels(
    source_path: str, dest_path: str, allowed_labels: tuple[str, ...]
) -> int:
    """Write rows whose label is in allowed_labels. Returns row count."""
    from edge_train.trainer import _resolve_columns

    _, label_col = _resolve_columns(source_path, None)
    allowed = set(allowed_labels)

    with open(source_path, encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        headers = reader.fieldnames or []
        kept = [row for row in reader if row.get(label_col, "") in allowed]

    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=headers)
        writer.writeheader()
        writer.writerows(kept)

    return len(kept)
