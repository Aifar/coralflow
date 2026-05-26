"""Local inference — run trained models on real text input."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf


class TextClassifier:
    """Load a SavedModel + model_meta.json and classify text locally."""

    def __init__(self, model_path: str):
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        if not (model_dir / "saved_model.pb").exists():
            raise FileNotFoundError(
                f"Not a SavedModel directory (missing saved_model.pb): {model_path}"
            )

        meta_path = model_dir / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"model_meta.json not found in {model_path}")

        meta = json.loads(meta_path.read_text())
        self.class_names = meta["classes"]
        self.max_sequence_length = meta.get("max_sequence_length", 64)
        vocab = meta["vocabulary"]

        self._vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max(5000, len(vocab)),
            output_mode="int",
            output_sequence_length=self.max_sequence_length,
        )
        self._vectorizer.set_vocabulary(vocab)

        self._model = tf.saved_model.load(str(model_dir))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def predict(self, text: str) -> tuple[str, float]:
        """Return (predicted_class_label, confidence)."""
        probs = self._infer([text])[0]
        top_idx = int(np.argmax(probs))
        return self.class_names[top_idx], float(probs[top_idx])

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return {class_label: probability} for all classes."""
        probs = self._infer([text])[0]
        return {name: float(probs[i]) for i, name in enumerate(self.class_names)}

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Return [(predicted_class_label, confidence), ...] for a batch."""
        all_probs = self._infer(texts)
        results = []
        for probs in all_probs:
            top_idx = int(np.argmax(probs))
            results.append((self.class_names[top_idx], float(probs[top_idx])))
        return results

    def _infer(self, texts: list[str]) -> np.ndarray:
        tokenized = self._vectorizer(tf.constant(texts))
        # set_vocabulary() produces int64; the model signature expects int32
        tokenized = tf.cast(tokenized, tf.int32)
        serving_fn = self._model.signatures["serving_default"]
        # Get the input tensor name from the serving signature
        input_key = list(serving_fn.structured_input_signature[1].keys())[0]
        output = serving_fn(**{input_key: tokenized})
        if isinstance(output, dict):
            output = next(iter(output.values()))
        return output.numpy()


def _ensure_phoenix_registered(arize_config) -> bool:
    """Register Phoenix OTEL if configured, reachable, and not already registered."""
    from edge_train.phoenix_util import ensure_phoenix_ready

    active, _ = ensure_phoenix_ready(arize_config)
    return active


def _create_prediction_span(
    text: str, predicted_label: str, confidence: float, all_probs: dict[str, float]
):
    """Create an OpenInference span for a single prediction."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("edge-train.inference")
        with tracer.start_as_current_span("classify") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            span.set_attribute("input.value", text)
            span.set_attribute("input.mime_type", "text/plain")
            span.set_attribute(
                "output.value",
                json.dumps(
                    {
                        "predicted_label": predicted_label,
                        "confidence": round(confidence, 4),
                        "all_probs": {k: round(v, 4) for k, v in all_probs.items()},
                    }
                ),
            )
            span.set_attribute("output.mime_type", "application/json")
            span.set_attribute(
                "metadata",
                json.dumps(
                    {
                        "model_type": "text-classifier",
                        "num_classes": len(all_probs),
                    }
                ),
            )
    except Exception:
        pass  # OTEL spans are best-effort; never crash inference


def log_prediction(
    log_path: str,
    text: str,
    predicted_label: str,
    confidence: float,
    all_probs: dict[str, float],
    ground_truth: str | None = None,
    create_span: bool = False,
):
    """Append a prediction to the JSON lines log file. Optionally create an OTEL span."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "all_probs": {k: round(v, 4) for k, v in all_probs.items()},
    }
    if ground_truth is not None:
        entry["ground_truth"] = ground_truth

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if create_span:
        _create_prediction_span(text, predicted_label, confidence, all_probs)
