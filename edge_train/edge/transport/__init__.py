"""Transport abstraction for pushing models to edge devices."""

from abc import ABC, abstractmethod

from edge_train.edge.registry import DeviceInfo
from edge_train.edge.model import ModelPackage


class BaseTransport(ABC):
    """Abstract transport for pushing models to edge devices."""

    @abstractmethod
    async def connect(self, device: DeviceInfo) -> bool:
        """Establish a connection/session with the device."""

    @abstractmethod
    async def push_model(self, package: ModelPackage) -> bool:
        """Transfer model bytes to the device."""

    @abstractmethod
    async def verify_checksum(self, expected_sha256: str) -> bool:
        """Ask the device to confirm its deployed model hash."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the connection."""


def create_transport(transport_type: str = "http", **kwargs: object) -> BaseTransport:
    """Factory: returns a transport instance by type name."""
    if transport_type == "http":
        from edge_train.edge.transport.http import HTTPTransport

        return HTTPTransport(**kwargs)

    raise ValueError(f"Unsupported transport: '{transport_type}'")
