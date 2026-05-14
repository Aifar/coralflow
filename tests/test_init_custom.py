"""Tests for edge_train.cli.init custom CSV validation."""

import pytest
from click.testing import CliRunner

from edge_train.cli.init import init


class TestInitCustom:
    def test_custom_valid_csv(self, sample_csv):
        runner = CliRunner()
        result = runner.invoke(init, ["custom", sample_csv])
        assert result.exit_code == 0, result.output

    def test_custom_nonexistent(self):
        runner = CliRunner()
        result = runner.invoke(init, ["custom", "/nonexistent/file.csv"])
        assert result.exit_code != 0

    def test_custom_empty_csv(self, temp_dir):
        path = str(temp_dir / "empty.csv")
        with open(path, "w") as f:
            f.write("header1,header2\n")
        runner = CliRunner()
        result = runner.invoke(init, ["custom", path])
        # CSV with only headers has 0 data rows → validation fails
        assert result.exit_code == 1
