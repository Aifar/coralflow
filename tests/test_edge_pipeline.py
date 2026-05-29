"""Tests for coralflow-edge-pipeline."""

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from werkzeug.serving import make_server

from edge_train.cli.edge_pipeline import main as edge_pipeline_main

_GATEWAY_DIR = Path(__file__).resolve().parents[1] / "examples" / "edge_gateway"


def _load_gateway_module():
    spec = importlib.util.spec_from_file_location(
        "coralflow_edge_gateway_pipeline", _GATEWAY_DIR / "app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_gateway(tmp_path):
    module = _load_gateway_module()
    model_path = tmp_path / "device-model.tflite"
    app = module.create_app(model_path)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield server.server_port, model_path
    server.shutdown()


class TestEdgePipelineCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(edge_pipeline_main, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output.lower()

    def test_deploy_only_via_host(self, tmp_path, local_gateway):
        port, device_model = local_gateway
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"pipeline deploy test model")

        runner = CliRunner()
        result = runner.invoke(
            edge_pipeline_main,
            [
                str(tmp_path / "unused.csv"),
                "--skip-train",
                "--skip-validate",
                "--tflite",
                str(tflite),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Edge pipeline complete" in result.output
        assert device_model.exists()
        assert device_model.read_bytes() == tflite.read_bytes()

    @pytest.mark.slow
    def test_full_pipeline_against_gateway(self, tmp_path, local_gateway, monkeypatch):
        port, device_model = local_gateway
        work = tmp_path / "work"
        work.mkdir()
        tflite = work / "model.tflite"

        monkeypatch.setenv(
            "EDGE_DEVICES",
            f'[{{"device_id":"test-gw","host":"127.0.0.1","port":{port}}}]',
        )
        monkeypatch.setenv("EDGE_DEFAULT_DEVICE", "test-gw")
        monkeypatch.setenv(
            "CORALFLOW_TRAINING_HISTORY_PATH",
            str(work / "training_history.json"),
        )
        monkeypatch.setenv(
            "CORALFLOW_DEPLOYMENTS_PATH",
            str(work / "deployments.json"),
        )

        dataset = Path(__file__).resolve().parents[1] / "data" / "urgent.csv"
        assert dataset.exists(), "data/urgent.csv required for pipeline test"

        cmd = [
            sys.executable,
            "-m",
            "edge_train.cli.edge_pipeline",
            str(dataset),
            "--output",
            str(work / "model_output"),
            "--tflite",
            str(tflite),
            "--epochs",
            "1",
            "--force",
        ]
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert tflite.exists()
        assert device_model.exists()
        assert device_model.stat().st_size == tflite.stat().st_size
