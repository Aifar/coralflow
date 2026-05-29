"""Tests for CoralFlow edge gateway (Flask)."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

_GATEWAY_DIR = Path(__file__).resolve().parents[1] / "examples" / "edge_gateway"


def _load_gateway_app():
    spec = importlib.util.spec_from_file_location(
        "coralflow_edge_gateway", _GATEWAY_DIR / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["coralflow_edge_gateway"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gateway_app(tmp_path):
    module = _load_gateway_app()
    model_path = tmp_path / "model.tflite"
    app = module.create_app(model_path)
    app.config["TESTING"] = True
    return app, model_path, module


class TestEdgeGateway:
    def test_health(self, gateway_app):
        app, _, _ = gateway_app
        client = app.test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_upload_and_checksum(self, gateway_app):
        from io import BytesIO

        app, model_path, _ = gateway_app
        client = app.test_client()
        payload = b"fake tflite bytes for gateway test"

        resp = client.post(
            "/api/v1/model",
            data={"model": (BytesIO(payload), "model.tflite")},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        expected = hashlib.sha256(payload).hexdigest()
        assert body["sha256"] == expected
        assert model_path.read_bytes() == payload

        checksum = client.get("/api/v1/checksum")
        assert checksum.status_code == 200
        assert checksum.get_json()["sha256"] == expected

    def test_upload_missing_field(self, gateway_app):
        app, _, _ = gateway_app
        client = app.test_client()
        resp = client.post("/api/v1/model", data={})
        assert resp.status_code == 400

    def test_reload_after_upload(self, gateway_app):
        from io import BytesIO

        app, _, _ = gateway_app
        client = app.test_client()
        client.post(
            "/api/v1/model",
            data={"model": (BytesIO(b"reload-test"), "model.tflite")},
        )

        for path in ("/api/v1/reload", "/reload"):
            resp = client.post(path)
            assert resp.status_code == 200
            assert resp.get_json()["status"] == "ok"

    def test_reload_without_model(self, gateway_app):
        app, _, _ = gateway_app
        client = app.test_client()
        resp = client.post("/api/v1/reload")
        assert resp.status_code == 404
