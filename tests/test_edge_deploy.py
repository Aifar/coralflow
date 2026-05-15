"""Tests for deploy orchestrator."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from edge_train.edge.deploy import deploy_model, DeployResult
from edge_train.edge.config import EdgeConfig
from edge_train.edge.registry import DeviceInfo, DeviceRegistry


class TestDeployModel:
    async def test_happy_path(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model content")

        transport = AsyncMock()
        transport.connect.return_value = True
        transport.push_model.return_value = True
        transport.verify_checksum.return_value = True
        transport.disconnect = AsyncMock()

        result = await deploy_model(
            str(tflite),
            host="10.0.0.1",
            port=8080,
            version="1.0.0",
            modality="text",
            transport=transport,
        )
        assert result.success
        assert result.device_id == "ephemeral-10.0.0.1"
        assert result.elapsed_sec > 0
        assert result.manifest is not None
        assert result.manifest.version == "1.0.0"
        transport.connect.assert_awaited_once()
        transport.push_model.assert_awaited_once()
        transport.verify_checksum.assert_awaited_once()
        transport.disconnect.assert_awaited_once()

    async def test_model_not_found(self):
        result = await deploy_model(
            "/nonexistent/model.tflite",
            host="10.0.0.1",
        )
        assert not result.success
        assert "Model not found" in result.error

    async def test_not_tflite(self, tmp_path):
        f = tmp_path / "model.txt"
        f.write_bytes(b"data")
        result = await deploy_model(str(f), host="10.0.0.1")
        assert not result.success
        assert "Not a TFLite" in result.error

    async def test_no_device_specified(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"data")
        result = await deploy_model(str(tflite))
        assert not result.success
        assert "No device specified" in result.error

    async def test_unknown_device_id(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"data")
        reg = DeviceRegistry(tmp_path / "registry.json")

        result = await deploy_model(str(tflite), device_id="unknown", registry=reg)
        assert not result.success
        assert "Unknown device" in result.error

    async def test_device_from_registry(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")
        reg = DeviceRegistry(tmp_path / "registry.json")
        reg.register(DeviceInfo(device_id="sensor-01", host="10.0.0.1"))

        transport = AsyncMock()
        transport.connect.return_value = True
        transport.push_model.return_value = True
        transport.verify_checksum.return_value = True
        transport.disconnect = AsyncMock()

        result = await deploy_model(
            str(tflite),
            device_id="sensor-01",
            registry=reg,
            transport=transport,
        )
        assert result.success
        assert result.device_id == "sensor-01"

    async def test_connection_refused(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")

        transport = AsyncMock()
        transport.connect.return_value = False
        transport.disconnect = AsyncMock()

        result = await deploy_model(str(tflite), host="10.0.0.1", transport=transport)
        assert not result.success
        assert "Connection refused" in result.error

    async def test_push_failure(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")

        transport = AsyncMock()
        transport.connect.return_value = True
        transport.push_model.return_value = False
        transport.disconnect = AsyncMock()

        result = await deploy_model(
            str(tflite),
            host="10.0.0.1",
            transport=transport,
            edge_config=EdgeConfig(retry_count=2),
        )
        assert not result.success
        assert "Push failed" in result.error

    async def test_checksum_mismatch(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")

        transport = AsyncMock()
        transport.connect.return_value = True
        transport.push_model.return_value = True
        transport.verify_checksum.return_value = False
        transport.disconnect = AsyncMock()

        result = await deploy_model(str(tflite), host="10.0.0.1", transport=transport)
        assert not result.success
        assert "Checksum mismatch" in result.error

    async def test_verify_skipped(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")

        transport = AsyncMock()
        transport.connect.return_value = True
        transport.push_model.return_value = True
        transport.disconnect = AsyncMock()

        result = await deploy_model(
            str(tflite),
            host="10.0.0.1",
            transport=transport,
            edge_config=EdgeConfig(verify_deployment=False),
        )
        assert result.success
        transport.verify_checksum.assert_not_awaited()

    async def test_transport_timeout(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")

        transport = AsyncMock()
        transport.connect.side_effect = asyncio.TimeoutError("Connection timed out")
        transport.disconnect = AsyncMock()

        result = await deploy_model(str(tflite), host="10.0.0.1", transport=transport)
        assert not result.success
        assert "Timeout" in result.error

    async def test_disconnect_on_failure(self, tmp_path):
        """Ensure disconnect is always called, even on failure."""
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")

        transport = AsyncMock()
        transport.connect.return_value = True
        transport.push_model.return_value = False
        transport.disconnect = AsyncMock()

        result = await deploy_model(str(tflite), host="10.0.0.1", transport=transport)
        assert not result.success
        transport.disconnect.assert_awaited_once()
