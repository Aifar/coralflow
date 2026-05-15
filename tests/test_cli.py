"""Tests for edge_train.cli commands."""

import sys

import pytest
from click.testing import CliRunner

from edge_train.cli import main
from edge_train.cli.init import init
from edge_train.cli.cost import cost
from edge_train.cli.validate import validate
from edge_train.cli.monitor import monitor
from edge_train.cli.deploy import deploy


@pytest.fixture
def runner():
    return CliRunner()


class TestMainCLI:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "edge-train" in result.output

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


class TestMonitorCommand:
    def test_monitor_no_args(self, runner):
        result = runner.invoke(monitor, [])
        assert result.exit_code == 0


class TestDeployCommand:
    def test_deploy_no_args(self, runner):
        result = runner.invoke(deploy, [])
        assert result.exit_code != 0
        assert "--model" in result.output

    def test_deploy_missing_model(self, runner):
        result = runner.invoke(deploy, ["--model", "/nonexistent/model.tflite", "--host", "10.0.0.1"])
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

        monkeypatch.setattr(sys.modules["edge_train.cli.deploy"], "_deploy_model", mock_deploy)

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

        monkeypatch.setattr(sys.modules["edge_train.cli.deploy"], "_deploy_model", mock_deploy)

        result = runner.invoke(deploy, ["--model", str(tflite), "--host", "10.0.0.1"])
        assert result.exit_code != 0
        assert "Connection failed" in result.output
