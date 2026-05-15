"""Tests for edge configuration."""

from edge_train.edge.config import EdgeConfig


class TestEdgeConfig:
    def test_defaults(self):
        cfg = EdgeConfig()
        assert cfg.default_transport == "http"
        assert cfg.connection_timeout_sec == 30
        assert cfg.retry_count == 3
        assert cfg.verify_deployment is True
        assert cfg.push_endpoint == "/api/v1/model"
        assert cfg.checksum_endpoint == "/api/v1/checksum"

    def test_registry_path_env(self, monkeypatch):
        monkeypatch.setenv("EDGE_REGISTRY_PATH", "/custom/path/devices.json")
        cfg = EdgeConfig()
        assert cfg.device_registry_path == "/custom/path/devices.json"
