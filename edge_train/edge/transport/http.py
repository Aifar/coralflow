"""HTTP transport for pushing models to edge devices."""

import asyncio
import logging

import aiohttp

from edge_train.edge.model import ModelPackage
from edge_train.edge.registry import DeviceInfo
from edge_train.edge.transport import BaseTransport

logger = logging.getLogger(__name__)


class HTTPTransport(BaseTransport):
    """Push TFLite models to edge devices via HTTP REST API."""

    def __init__(
        self,
        timeout: int = 30,
        retries: int = 3,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._timeout = timeout
        self._retries = retries
        self._session = session
        self._device: DeviceInfo | None = None

    async def connect(self, device: DeviceInfo) -> bool:
        self._device = device
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        try:
            url = f"http://{device.host}:{device.port}/health"
            async with self._session.get(url) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def push_model(self, package: ModelPackage) -> bool:
        assert self._device is not None
        assert self._session is not None

        url = f"http://{self._device.host}:{self._device.port}/api/v1/model"

        for attempt in range(self._retries):
            try:
                data = aiohttp.FormData()
                data.add_field(
                    "model",
                    package.model_bytes,
                    filename="model.tflite",
                    content_type="application/octet-stream",
                )

                async with self._session.post(url, data=data) as resp:
                    if resp.status == 200:
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < self._retries - 1:
                    wait = 2**attempt
                    logger.info(
                        "Push attempt %d failed, retrying in %ds...",
                        attempt + 1,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error("All %d push attempts failed", self._retries)
        return False

    async def verify_checksum(self, expected_sha256: str) -> bool:
        assert self._device is not None
        assert self._session is not None

        url = f"http://{self._device.host}:{self._device.port}/api/v1/checksum"
        try:
            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return False
                data = await resp.json()
                actual = data.get("sha256", "")
                return actual.lower() == expected_sha256.lower()
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def disconnect(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
