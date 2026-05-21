"""Training progress callback and progress-aware training wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Prints loss and accuracy at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0)
        acc = logs.get("accuracy", 0)
        val_loss = logs.get("val_loss", 0)
        val_acc = logs.get("val_accuracy", 0)

        total = self.params.get("epochs", "?")
        line = f"  [{epoch + 1}/{total}] loss: {loss:.4f} - accuracy: {acc:.4f}"
        if val_loss:
            line += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
        print(line)

    def on_train_end(self, logs=None):
        logs = logs or {}
        loss = logs.get("loss", 0)
        acc = logs.get("accuracy", 0)
        val_loss = logs.get("val_loss", 0)
        val_acc = logs.get("val_accuracy", 0)
        print(
            f"  Training finished — final loss: {loss:.4f}, accuracy: {acc:.4f}"
            + (
                f", val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}"
                if val_loss
                else ""
            )
        )


def run_training_with_progress(
    dataset_path: str,
    target_column: str | None = None,
    output_dir: str = "./model_output",
    epochs: int = 10,
) -> Path:
    """Train a text classifier with live epoch progress output.

    Wraps train_text_classifier() with a TrainingProgressCallback.
    """
    from edge_train.trainer import train_text_classifier

    progress_cb = TrainingProgressCallback()
    return train_text_classifier(
        dataset_path=dataset_path,
        target_column=target_column,
        output_dir=output_dir,
        epochs=epochs,
        callbacks=[progress_cb],
    )
