"""CoralFlow edge HTTP gateway — receive TFLite models from coralflow deploy."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(
    os.environ.get("EDGE_MODEL_PATH", "/var/lib/coralflow/model.tflite")
)
DEFAULT_HOST = os.environ.get("EDGE_GATEWAY_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("EDGE_GATEWAY_PORT", "8080"))

_interpreter = None
_model_sha256 = ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_checksum(path: Path) -> str:
    global _model_sha256
    if path.exists() and path.stat().st_size > 0:
        _model_sha256 = _sha256_file(path)
    else:
        _model_sha256 = ""
    return _model_sha256


def _try_reload_interpreter(path: Path) -> tuple[bool, str]:
    """Best-effort TFLite hot reload (optional; requires TensorFlow on device)."""
    global _interpreter
    if not path.exists():
        _interpreter = None
        return False, "model file missing"

    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        _interpreter = interpreter
        return True, "interpreter loaded"
    except Exception as exc:
        logger.warning("TFLite reload skipped: %s", exc)
        _interpreter = None
        return True, f"model saved (inference reload skipped: {exc})"


def create_app(model_path: Path | None = None) -> Flask:
    app = Flask(__name__)
    model_file = model_path or DEFAULT_MODEL_PATH

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.post("/api/v1/model")
    def upload_model():
        upload = request.files.get("model")
        if upload is None:
            return jsonify({"error": "missing multipart field 'model'"}), 400

        model_file.parent.mkdir(parents=True, exist_ok=True)
        upload.save(model_file)
        checksum = _refresh_checksum(model_file)
        ok, message = _try_reload_interpreter(model_file)
        status = 200 if ok else 500
        return (
            jsonify(
                {
                    "status": "ok" if ok else "error",
                    "sha256": checksum,
                    "bytes": model_file.stat().st_size,
                    "reload": message,
                }
            ),
            status,
        )

    @app.get("/api/v1/checksum")
    def checksum():
        if model_file.exists() and not _model_sha256:
            _refresh_checksum(model_file)
        return jsonify({"sha256": _model_sha256}), 200

    @app.post("/api/v1/reload")
    @app.post("/reload")
    def reload_model():
        if not model_file.exists():
            return jsonify({"error": "no model on device"}), 404
        checksum = _refresh_checksum(model_file)
        ok, message = _try_reload_interpreter(model_file)
        status = 200 if ok else 500
        return (
            jsonify(
                {
                    "status": "ok" if ok else "error",
                    "sha256": checksum,
                    "reload": message,
                }
            ),
            status,
        )

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    app = create_app()
    logger.info("CoralFlow edge gateway listening on %s:%s", DEFAULT_HOST, DEFAULT_PORT)
    logger.info("Model path: %s", DEFAULT_MODEL_PATH)
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, threaded=True)


if __name__ == "__main__":
    main()
