"""Tests for transport abstraction and HTTP transport."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from edge_train.edge.registry import DeviceInfo


def _mock_http_context(status: int = 200, json_data: dict | None = None):
    """Create a mock async context manager for aiohttp response."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data or {})

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    return mock_context


class TestBaseTransport:
    def test_cannot_instantiate_abc(self):
        from edge_train.edge.transport import BaseTransport

        with pytest.raises(TypeError):
            BaseTransport()  # type: ignore[abstract]

    def test_create_transport_http(self):
        from edge_train.edge.transport import create_transport

        t = create_transport("http", timeout=15, retries=2)
        from edge_train.edge.transport.http import HTTPTransport

        assert isinstance(t, HTTPTransport)

    def test_create_transport_unknown(self):
        from edge_train.edge.transport import create_transport

        with pytest.raises(ValueError, match="Unsupported transport"):
            create_transport("mqtt")


class TestHTTPTransport:
    def _mock_session(self):
        """Create a MagicMock session with async context-manager methods."""
        session = MagicMock()
        session.get.return_value = _mock_http_context(200)
        session.post.return_value = _mock_http_context(200)
        session.close = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_connect_success(self):
        from edge_train.edge.transport.http import HTTPTransport

        mock_session = self._mock_session()
        mock_session.get.return_value = _mock_http_context(200)

        t = HTTPTransport(timeout=10, retries=2, session=mock_session)
        device = DeviceInfo(device_id="d1", host="10.0.0.1", port=8080)
        result = await t.connect(device)
        assert result is True
        mock_session.get.assert_called_once()
        await t.disconnect()

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        from edge_train.edge.transport.http import HTTPTransport

        mock_session = self._mock_session()
        mock_session.get.return_value = _mock_http_context(500)

        t = HTTPTransport(timeout=10, retries=2, session=mock_session)
        device = DeviceInfo(device_id="d1", host="10.0.0.1", port=8080)
        result = await t.connect(device)
        assert result is False

    @pytest.mark.asyncio
    async def test_push_model_success(self, tmp_path):
        from edge_train.edge.transport.http import HTTPTransport
        from edge_train.edge.model import ModelPackage

        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"model bytes")

        mock_session = self._mock_session()
        t = HTTPTransport(timeout=10, retries=2, session=mock_session)
        await t.connect(DeviceInfo(device_id="d1", host="10.0.0.1", port=8080))

        pkg = ModelPackage(tflite, version="1.0", modality="text")
        result = await t.push_model(pkg)
        assert result is True

    @pytest.mark.asyncio
    async def test_push_model_failure(self, tmp_path):
        from edge_train.edge.transport.http import HTTPTransport
        from edge_train.edge.model import ModelPackage

        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"model bytes")

        mock_session = self._mock_session()
        mock_session.post.return_value = _mock_http_context(500)

        t = HTTPTransport(timeout=10, retries=2, session=mock_session)
        await t.connect(DeviceInfo(device_id="d1", host="10.0.0.1", port=8080))

        pkg = ModelPackage(tflite, version="1.0", modality="text")
        result = await t.push_model(pkg)
        assert result is False
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_verify_checksum_match(self):
        from edge_train.edge.transport.http import HTTPTransport

        mock_session = self._mock_session()
        mock_session.get.return_value = _mock_http_context(
            200, {"sha256": "abcdef123456"}
        )

        t = HTTPTransport(timeout=10, retries=2, session=mock_session)
        await t.connect(DeviceInfo(device_id="d1", host="10.0.0.1", port=8080))

        result = await t.verify_checksum("abcdef123456")
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_checksum_mismatch(self):
        from edge_train.edge.transport.http import HTTPTransport

        mock_session = self._mock_session()
        mock_session.get.return_value = _mock_http_context(
            200, {"sha256": "differenthash"}
        )

        t = HTTPTransport(timeout=10, retries=2, session=mock_session)
        await t.connect(DeviceInfo(device_id="d1", host="10.0.0.1", port=8080))

        result = await t.verify_checksum("expectedhash")
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_closes_session(self):
        from edge_train.edge.transport.http import HTTPTransport

        mock_session = self._mock_session()
        t = HTTPTransport(timeout=10, retries=2, session=mock_session)
        await t.disconnect()
        mock_session.close.assert_awaited_once()
