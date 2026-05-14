"""Tests for edge_train.cli commands."""

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
    def test_deploy(self, runner):
        # --model is required, so no args should show help/error
        result = runner.invoke(deploy, ["--model", "/fake/model.tflite"])
        assert result.exit_code == 0
        assert "coming in edge-train v0.2.0" in result.output
