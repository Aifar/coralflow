"""Tests for edge_train.cli commands."""

import builtins
import sys

import pytest
from click.testing import CliRunner

from edge_train.cli import main
from edge_train.cli.init import init
from edge_train.cli.cost import cost
from edge_train.cli.validate import validate
from edge_train.cli.monitor import monitor
from edge_train.cli.deploy import deploy
from edge_train.cli.train import train
from edge_train.cli.predict import predict


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_phoenix(mocker):
    """Inject mock phoenix.otel into sys.modules so tests don't need the real package."""
    phoenix_mod = mocker.MagicMock()
    otel_mod = mocker.MagicMock()
    phoenix_mod.otel = otel_mod
    sys.modules["phoenix"] = phoenix_mod
    sys.modules["phoenix.otel"] = otel_mod
    yield otel_mod
    sys.modules.pop("phoenix", None)
    sys.modules.pop("phoenix.otel", None)


class TestMainCLI:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "coralflow" in result.output

    def test_no_command_shows_help(self, runner):
        result = runner.invoke(main, [])
        # Click group with no subcommand prints help and exits 0
        assert result.exit_code in (0, 2)


class TestInitCommands:
    def test_list(self, runner):
        result = runner.invoke(init, ["list"])
        assert result.exit_code == 0
        assert "urgent" in result.output
        assert "expense" in result.output

    def test_download_unknown_dataset(self, runner):
        result = runner.invoke(init, ["download", "nonexistent", "-o", "/tmp"])
        assert result.exit_code != 0

    def test_download_urgent(self, runner, temp_dir):
        output = str(temp_dir / "urgent.csv")
        result = runner.invoke(init, ["download", "urgent", "-o", output])
        assert result.exit_code == 0, result.output


class TestCostCommand:
    def test_cost_known_dataset(self, runner):
        result = runner.invoke(cost, ["urgent"])
        assert result.exit_code == 0
        assert "$" in result.output
        assert "urgent" in result.output

    def test_cost_unknown_dataset(self, runner):
        result = runner.invoke(cost, ["unknown"])
        assert result.exit_code == 0
        assert "Unknown" in result.output

    def test_cost_no_args(self, runner):
        result = runner.invoke(cost, [])
        assert result.exit_code == 0
        assert "Hourly rates" in result.output


class TestValidateCommand:
    def test_validate_no_args(self, runner):
        result = runner.invoke(validate, [])
        assert result.exit_code != 0
        assert "Error" in result.output or "--model" in result.output


class TestTrainCommand:
    def test_train_help(self, runner):
        result = runner.invoke(train, ["--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output
        assert "--cloud" in result.output
        assert "--epochs" in result.output

    def test_local_train_success(self, runner, sample_text_csv, tmp_path):
        out = tmp_path / "model"
        result = runner.invoke(
            train, ["-d", sample_text_csv, "-o", str(out), "--epochs", "2"]
        )
        assert result.exit_code == 0, result.output
        assert "Model saved to:" in result.output
        assert out.exists()

    def test_local_train_unsupported_modality(self, runner, tmp_path):
        out = tmp_path / "model"
        result = runner.invoke(
            train, ["-d", "/tmp/nonexistent.csv", "--type", "image", "-o", str(out)]
        )
        assert result.exit_code != 0
        assert "not yet supported" in result.output.lower()

    def test_cloud_requires_gcp(self, runner, sample_text_csv, clear_env):
        result = runner.invoke(train, ["-d", sample_text_csv, "--cloud"])
        assert result.exit_code != 0
        assert "GCP not configured" in result.output


class TestMonitorCommand:
    def test_monitor_not_configured(self, runner, clear_env):
        result = runner.invoke(monitor, [])
        assert result.exit_code == 0
        assert "not configured" in result.output.lower()

    def test_monitor_status_not_configured(self, runner, clear_env):
        result = runner.invoke(monitor, ["--status"])
        assert result.exit_code == 0
        assert "not configured" in result.output.lower()
        assert "edge-train" in result.output

    def test_monitor_status_configured(
        self, runner, clear_env, monkeypatch, mock_phoenix
    ):
        monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://example.com/v1/traces"
        )
        monkeypatch.setenv("PHOENIX_PROJECT_NAME", "my-project")

        result = runner.invoke(monitor, ["--status"])
        assert result.exit_code == 0
        assert "https://example.com/v1/traces" in result.output
        assert "my-project" in result.output
        assert "configured" in result.output
        assert "connected" in result.output
        mock_phoenix.register.assert_called_once()

    def test_monitor_register_prints_info(
        self, runner, clear_env, monkeypatch, mock_phoenix
    ):
        monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://example.com/v1/traces"
        )

        result = runner.invoke(monitor, [])
        assert result.exit_code == 0
        assert "registered" in result.output.lower()
        assert "https://example.com/v1/traces" in result.output
        mock_phoenix.register.assert_called_once_with(
            endpoint="https://example.com/v1/traces",
            project_name="edge-train",
            auto_instrument=False,
        )

    def test_monitor_dashboard(
        self, runner, clear_env, monkeypatch, mock_phoenix, mocker
    ):
        monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/v1/traces"
        )
        mock_open = mocker.patch("webbrowser.open")

        result = runner.invoke(monitor, ["--dashboard"])
        assert result.exit_code == 0
        assert "https://app.phoenix.arize.com" in result.output
        mock_open.assert_called_once_with("https://app.phoenix.arize.com")

    def test_monitor_dashboard_derived_url(
        self, runner, clear_env, monkeypatch, mock_phoenix, mocker
    ):
        monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.example.com/v1/traces"
        )
        mock_open = mocker.patch("webbrowser.open")

        result = runner.invoke(monitor, ["--dashboard"])
        assert result.exit_code == 0
        mock_open.assert_called_once_with("https://phoenix.example.com")

    def test_monitor_registration_failure(
        self, runner, clear_env, monkeypatch, mock_phoenix
    ):
        monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://example.com/v1/traces"
        )
        mock_phoenix.register.side_effect = RuntimeError("Connection refused")

        result = runner.invoke(monitor, [])
        assert result.exit_code == 0
        assert "Connection refused" in result.output

    def test_monitor_import_error(self, runner, clear_env, monkeypatch):
        monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://example.com/v1/traces"
        )

        # Remove phoenix from sys.modules so the import path is exercised
        sys.modules.pop("phoenix", None)
        sys.modules.pop("phoenix.otel", None)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("phoenix", "phoenix.otel"):
                raise ImportError("No module named 'phoenix'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = runner.invoke(monitor, [])
        assert result.exit_code == 0
        assert "arize-phoenix-otel" in result.output


class TestDeployCommand:
    def test_deploy_no_args(self, runner):
        result = runner.invoke(deploy, [])
        assert result.exit_code != 0
        assert "--model" in result.output

    def test_deploy_missing_model(self, runner):
        result = runner.invoke(
            deploy, ["--model", "/nonexistent/model.tflite", "--host", "10.0.0.1"]
        )
        assert result.exit_code != 0

    def test_deploy_no_device(self, runner, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")
        result = runner.invoke(deploy, ["--model", str(tflite)])
        assert result.exit_code != 0
        assert "device" in result.output.lower()

    def test_deploy_success(self, runner, tmp_path, monkeypatch):
        from edge_train.edge.deploy import DeployResult

        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")

        async def mock_deploy(model_path, **kwargs):
            return DeployResult(
                success=True, device_id="ephemeral-10.0.0.1", elapsed_sec=2.35
            )

        monkeypatch.setattr(
            sys.modules["edge_train.cli.deploy"], "_deploy_model", mock_deploy
        )

        result = runner.invoke(deploy, ["--model", str(tflite), "--host", "10.0.0.1"])
        assert result.exit_code == 0, result.output
        assert "Deployed" in result.output or "done" in result.output.lower()
        assert "ephemeral-10.0.0.1" in result.output

    def test_deploy_failure(self, runner, tmp_path, monkeypatch):
        from edge_train.edge.deploy import DeployResult

        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")

        async def mock_deploy(model_path, **kwargs):
            return DeployResult(success=False, error="Connection failed")

        monkeypatch.setattr(
            sys.modules["edge_train.cli.deploy"], "_deploy_model", mock_deploy
        )

        result = runner.invoke(deploy, ["--model", str(tflite), "--host", "10.0.0.1"])
        assert result.exit_code != 0
        assert "Connection failed" in result.output


class TestPredictCommand:
    def test_predict_help(self, runner):
        result = runner.invoke(predict, ["--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--text" in result.output
        assert "--csv" in result.output

    def test_single_prediction(self, runner, sample_text_csv, tmp_path):
        from edge_train.trainer import train_text_classifier

        model_path = train_text_classifier(
            sample_text_csv, output_dir=str(tmp_path / "model"), epochs=2
        )

        result = runner.invoke(predict, ["-m", str(model_path), "-t", "hello world"])
        assert result.exit_code == 0, result.output
        assert "Predicted:" in result.output

    def test_batch_csv_prediction(self, runner, sample_text_csv, tmp_path):
        from edge_train.trainer import train_text_classifier

        model_path = train_text_classifier(
            sample_text_csv, output_dir=str(tmp_path / "model"), epochs=2
        )

        out_csv = tmp_path / "preds.csv"
        result = runner.invoke(
            predict,
            ["-m", str(model_path), "-c", sample_text_csv, "-o", str(out_csv)],
        )
        assert result.exit_code == 0, result.output
        assert out_csv.exists()

    def test_missing_model(self, runner):
        result = runner.invoke(predict, ["-m", "/nonexistent", "-t", "hello"])
        assert result.exit_code != 0

    def test_no_input_raises(self, runner, sample_text_csv, tmp_path):
        from edge_train.trainer import train_text_classifier

        model_path = train_text_classifier(
            sample_text_csv, output_dir=str(tmp_path / "model"), epochs=2
        )
        result = runner.invoke(predict, ["-m", str(model_path)])
        assert result.exit_code != 0
        assert "Error" in result.output


class TestMonitorRetrain:
    def test_retrain_no_log_file(self, runner, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "EDGE_PREDICTION_LOG_PATH", str(tmp_path / "nonexistent.jsonl")
        )
        result = runner.invoke(monitor, ["--retrain", "--threshold", "0.80"])
        assert result.exit_code == 0
        assert "No prediction log" in result.output

    def test_retrain_no_labeled_entries(self, runner, tmp_path, monkeypatch):
        log_path = tmp_path / "preds.jsonl"
        log_path.write_text(
            '{"text":"hello","predicted_label":"greeting","confidence":0.9,"all_probs":{"greeting":0.9,"question":0.1},"timestamp":"2026-01-01T00:00:00Z"}\n'
        )
        monkeypatch.setenv("EDGE_PREDICTION_LOG_PATH", str(log_path))

        result = runner.invoke(monitor, ["--retrain", "--threshold", "0.80"])
        assert result.exit_code == 0
        assert "No labeled entries" in result.output

    def test_retrain_below_threshold_needs_dataset(self, runner, tmp_path, monkeypatch):
        log_path = tmp_path / "preds.jsonl"
        # 6/10 correct = 60% accuracy, below 0.80 threshold
        lines = []
        for i in range(10):
            correct = i < 6
            lines.append(
                '{"text":"text%d","predicted_label":"A","confidence":0.7,"all_probs":{"A":0.7,"B":0.3},"timestamp":"2026-01-01T00:00:%02dZ","ground_truth":"%s"}\n'
                % (i, i, "A" if correct else "B")
            )
        log_path.write_text("".join(lines))
        monkeypatch.setenv("EDGE_PREDICTION_LOG_PATH", str(log_path))

        result = runner.invoke(monitor, ["--retrain", "--threshold", "0.80"])
        assert result.exit_code != 0
        assert "Current accuracy" in result.output
        assert "below threshold" in result.output.lower()
        assert "dataset" in result.output.lower()
