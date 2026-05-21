"""Local training — no cloud needed."""

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def _detect_columns(csv_path: str) -> tuple[str, str]:
    """Auto-detect text and label columns from a CSV header."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    text_col = None
    label_col = None
    for h in headers:
        lower = h.lower().strip()
        if lower in (
            "text",
            "message",
            "content",
            "sentence",
            "review",
            "comment",
            "description",
        ):
            text_col = h
        elif lower in ("label", "class", "category", "target", "intent"):
            label_col = h

    if text_col is None:
        raise ValueError(f"No text column found in {csv_path}. Headers: {headers}")
    if label_col is None:
        raise ValueError(f"No label column found in {csv_path}. Headers: {headers}")

    return text_col, label_col


def _load_csv(
    csv_path: str, text_col: str, label_col: str
) -> tuple[list[str], list[str], list[str]]:
    """Load texts, labels, and unique class names from CSV."""
    texts = []
    labels = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row[text_col])
            labels.append(row[label_col])

    unique_labels = sorted(set(labels))
    return texts, labels, unique_labels


def _resolve_columns(dataset_path: str, target_column: str | None) -> tuple[str, str]:
    """Resolve text and label column names from CSV."""
    if target_column:
        with open(dataset_path, encoding="utf-8") as f:
            headers = csv.DictReader(f).fieldnames or []
        text_col = None
        for h in headers:
            if h.lower().strip() in (
                "text",
                "message",
                "content",
                "sentence",
                "review",
                "comment",
                "description",
            ):
                text_col = h
                break
        if text_col is None:
            text_col = (
                headers[0]
                if headers[0] != target_column
                else (headers[1] if len(headers) > 1 else headers[0])
            )
        return text_col, target_column
    return _detect_columns(dataset_path)


def train_text_classifier(
    dataset_path: str,
    target_column: str | None = None,
    output_dir: str = "./model_output",
    epochs: int = 10,
    callbacks: list | None = None,
) -> Path:
    """Train a small text classifier locally and export as SavedModel.

    Tokenization is separated from the model so the exported SavedModel
    takes integer token sequences — compatible with TFLite conversion.

    Returns the path to the SavedModel directory.
    """
    text_col, label_col = _resolve_columns(dataset_path, target_column)
    texts, labels, class_names = _load_csv(dataset_path, text_col, label_col)
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    y = np.array([label_to_idx[l] for l in labels])
    num_classes = len(class_names)

    # Adapt vocabulary on the text corpus
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=5000, output_mode="int", output_sequence_length=64
    )
    vectorizer.adapt(tf.constant(texts))

    # Tokenize all texts
    tokenized = vectorizer(tf.constant(texts)).numpy()

    # Build model that accepts pre-tokenized integer sequences
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(64,), dtype=tf.int32),
            tf.keras.layers.Embedding(5000, 64, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        tokenized,
        y,
        epochs=epochs,
        validation_split=0.2,
        verbose=0,
        callbacks=callbacks or [],
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.export(str(out))

    # Save vocabulary + class labels alongside the model
    vocab = vectorizer.get_vocabulary()
    meta = {"classes": class_names, "vocabulary": vocab, "max_sequence_length": 64}
    (out / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return out
