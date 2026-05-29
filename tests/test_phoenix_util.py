"""Tests for Arize Phoenix connectivity helpers."""

from unittest.mock import MagicMock, patch

import pytest

from edge_train.config import ArizeConfig
from edge_train.phoenix_util import (
    check_phoenix_running,
    ensure_phoenix_ready,
    format_phoenix_start_instructions,
    is_local_collector,
)


class TestPhoenixUtil:
    def test_is_local_collector(self):
        assert is_local_collector("http://localhost:6006/v1/traces")
        assert not is_local_collector("https://app.phoenix.arize.com/v1/traces")

    def test_check_not_configured(self):
        cfg = ArizeConfig(api_key="", collector_endpoint="")
        status = check_phoenix_running(cfg)
        assert not status.configured
        assert not status.reachable

    @patch("edge_train.phoenix_util._probe_url")
    def test_check_local_unreachable(self, mock_probe, monkeypatch):
        mock_probe.return_value = (False, "connection refused")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
        )
        cfg = ArizeConfig(
            api_key="key",
            collector_endpoint="http://localhost:6006/v1/traces",
        )
        status = check_phoenix_running(cfg)
        assert status.configured
        assert not status.reachable
        assert status.is_local
        assert "phoenix serve" in format_phoenix_start_instructions(status)

    @patch("edge_train.phoenix_util._probe_url")
    def test_check_cloud_reachable(self, mock_probe, monkeypatch):
        mock_probe.return_value = (True, "HTTP 200")
        monkeypatch.setenv("PHOENIX_API_KEY", "key")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/v1/traces"
        )
        cfg = ArizeConfig(
            api_key="key",
            collector_endpoint="https://app.phoenix.arize.com/v1/traces",
        )
        status = check_phoenix_running(cfg)
        assert status.reachable
        assert not status.is_local

    def test_ensure_skips_when_not_configured(self):
        cfg = ArizeConfig(api_key="", collector_endpoint="")
        active, err = ensure_phoenix_ready(cfg)
        assert not active
        assert err == ""

    @patch("edge_train.phoenix_util._register_phoenix")
    @patch("edge_train.phoenix_util.check_phoenix_running")
    def test_ensure_blocks_when_down(self, mock_check, mock_register, monkeypatch):
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"
        )
        mock_check.return_value = MagicMock(
            configured=True,
            reachable=False,
            is_local=True,
            collector_endpoint="http://localhost:6006/v1/traces",
            dashboard_url="http://localhost:6006",
            detail="refused",
        )
        cfg = ArizeConfig(
            api_key="key",
            collector_endpoint="http://localhost:6006/v1/traces",
        )
        active, err = ensure_phoenix_ready(cfg)
        assert not active
        assert "not running" in err
        mock_register.assert_not_called()

    @patch("edge_train.phoenix_util._register_phoenix")
    @patch("edge_train.phoenix_util.check_phoenix_running")
    def test_ensure_registers_when_up(self, mock_check, mock_register, monkeypatch):
        monkeypatch.setenv("PHOENIX_API_KEY", "key")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/v1/traces"
        )
        mock_check.return_value = MagicMock(
            configured=True,
            reachable=True,
            is_local=False,
            collector_endpoint="https://app.phoenix.arize.com/v1/traces",
            dashboard_url="https://app.phoenix.arize.com",
            detail="HTTP 200",
        )
        mock_register.return_value = (True, "")
        cfg = ArizeConfig(
            api_key="key",
            collector_endpoint="https://app.phoenix.arize.com/v1/traces",
        )
        active, err = ensure_phoenix_ready(cfg)
        assert active
        assert err == ""
        mock_register.assert_called_once()
