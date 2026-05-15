"""Tests for device registry."""

import json
from edge_train.edge.registry import DeviceInfo, DeviceRegistry


class TestDeviceInfo:
    def test_defaults(self):
        d = DeviceInfo(device_id="sensor-01")
        assert d.device_id == "sensor-01"
        assert d.transport == "http"
        assert d.host == "localhost"
        assert d.port == 8080
        assert d.api_key is None
        assert d.label == ""


class TestDeviceRegistry:
    def test_empty_registry(self, tmp_path):
        path = tmp_path / "devices.json"
        reg = DeviceRegistry(path)
        assert reg.list_devices() == []
        assert reg.resolve("nonexistent") is None

    def test_register_and_resolve(self, tmp_path):
        reg = DeviceRegistry(tmp_path / "devices.json")
        d = DeviceInfo(device_id="sensor-01", host="192.168.1.42", port=8080)
        reg.register(d)
        resolved = reg.resolve("sensor-01")
        assert resolved is not None
        assert resolved.host == "192.168.1.42"
        assert resolved.port == 8080

    def test_unregister(self, tmp_path):
        reg = DeviceRegistry(tmp_path / "devices.json")
        reg.register(DeviceInfo(device_id="sensor-01"))
        reg.unregister("sensor-01")
        assert reg.resolve("sensor-01") is None

    def test_list_devices(self, tmp_path):
        reg = DeviceRegistry(tmp_path / "devices.json")
        reg.register(DeviceInfo(device_id="a"))
        reg.register(DeviceInfo(device_id="b"))
        assert len(reg.list_devices()) == 2

    def test_persistence(self, tmp_path):
        path = tmp_path / "devices.json"
        reg1 = DeviceRegistry(path)
        reg1.register(DeviceInfo(device_id="persist-test", host="10.0.0.1"))

        reg2 = DeviceRegistry(path)
        resolved = reg2.resolve("persist-test")
        assert resolved is not None
        assert resolved.host == "10.0.0.1"

    def test_register_overwrite(self, tmp_path):
        reg = DeviceRegistry(tmp_path / "devices.json")
        reg.register(DeviceInfo(device_id="d", host="10.0.0.1"))
        reg.register(DeviceInfo(device_id="d", host="10.0.0.2"))
        assert reg.resolve("d").host == "10.0.0.2"

    def test_corrupt_json(self, tmp_path):
        path = tmp_path / "devices.json"
        path.write_text("not valid json")
        import json as j
        import pytest
        with pytest.raises(j.JSONDecodeError):
            DeviceRegistry(path)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "devices.json"
        path.write_text("")
        reg = DeviceRegistry(path)
        assert reg.list_devices() == []
