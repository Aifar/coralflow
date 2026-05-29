"""Tests for EDGE_DEVICES .env parsing and deploy target resolution."""

import json

import pytest

from edge_train.edge.registry import (
    DeviceRegistry,
    load_device_registry,
    parse_edge_devices,
    resolve_deploy_targets,
)


class TestParseEdgeDevices:
    def test_json_array(self):
        raw = json.dumps(
            [
                {"device_id": "gw1", "host": "192.168.1.50", "port": 8080},
                {"id": "gw2", "host": "192.168.1.51", "port": 9090, "label": "Line 2"},
            ]
        )
        devices = parse_edge_devices(raw)
        assert len(devices) == 2
        assert devices[0].device_id == "gw1"
        assert devices[0].host == "192.168.1.50"
        assert devices[1].device_id == "gw2"
        assert devices[1].port == 9090
        assert devices[1].label == "Line 2"

    def test_compact_format(self):
        raw = "gw1@192.168.1.50:8080,gw2@192.168.1.51:9090"
        devices = parse_edge_devices(raw)
        assert [d.device_id for d in devices] == ["gw1", "gw2"]
        assert devices[1].port == 9090

    def test_empty(self):
        assert parse_edge_devices("") == []
        assert parse_edge_devices("   ") == []

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="JSON is invalid"):
            parse_edge_devices("[not-json")

    def test_invalid_compact(self):
        with pytest.raises(ValueError, match="id@host:port"):
            parse_edge_devices("bad-entry")


class TestLoadDeviceRegistry:
    def test_from_edge_devices_env(self, monkeypatch):
        monkeypatch.setenv(
            "EDGE_DEVICES",
            '[{"device_id":"gw1","host":"10.0.0.1","port":8080}]',
        )
        reg = load_device_registry()
        assert reg.resolve("gw1") is not None
        assert reg.resolve("gw1").host == "10.0.0.1"

    def test_edge_devices_takes_priority_over_legacy_json(self, monkeypatch, tmp_path):
        legacy = tmp_path / "devices.json"
        legacy.write_text(
            json.dumps(
                {
                    "devices": {
                        "legacy-gw": {
                            "device_id": "legacy-gw",
                            "host": "10.0.0.99",
                            "port": 8080,
                            "transport": "http",
                            "api_key": None,
                            "label": "",
                        }
                    }
                }
            )
        )
        monkeypatch.setenv("EDGE_REGISTRY_PATH", str(legacy))
        monkeypatch.setenv("EDGE_DEVICES", "env-gw@10.0.0.2:8080")
        reg = load_device_registry()
        assert reg.resolve("env-gw") is not None
        assert reg.resolve("legacy-gw") is None

    def test_legacy_json_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delenv("EDGE_DEVICES", raising=False)
        legacy = tmp_path / "devices.json"
        legacy.write_text(
            json.dumps(
                {
                    "devices": {
                        "legacy-gw": {
                            "device_id": "legacy-gw",
                            "host": "10.0.0.99",
                            "port": 8080,
                            "transport": "http",
                            "api_key": None,
                            "label": "",
                        }
                    }
                }
            )
        )
        monkeypatch.setenv("EDGE_REGISTRY_PATH", str(legacy))
        reg = load_device_registry()
        assert reg.resolve("legacy-gw").host == "10.0.0.99"


class TestResolveDeployTargets:
    def test_default_device(self):
        reg = DeviceRegistry.from_list(parse_edge_devices("gw1@10.0.0.1:8080"))
        targets = resolve_deploy_targets(reg, None, "gw1")
        assert len(targets) == 1
        assert targets[0].device_id == "gw1"

    def test_all_devices(self):
        reg = DeviceRegistry.from_list(
            parse_edge_devices("gw1@10.0.0.1:8080,gw2@10.0.0.2:8080")
        )
        targets = resolve_deploy_targets(reg, "all", "")
        assert len(targets) == 2

    def test_missing_device_raises(self):
        reg = DeviceRegistry.from_list([])
        with pytest.raises(ValueError, match="EDGE_DEFAULT_DEVICE"):
            resolve_deploy_targets(reg, None, "")

    def test_unknown_device_raises(self):
        reg = DeviceRegistry.from_list(parse_edge_devices("gw1@10.0.0.1:8080"))
        with pytest.raises(ValueError, match="Unknown device"):
            resolve_deploy_targets(reg, "missing", "")
