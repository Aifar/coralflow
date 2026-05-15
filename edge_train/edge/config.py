"""Edge device configuration."""

import os
from dataclasses import dataclass, field


@dataclass
class EdgeConfig:
    """Settings for edge device communication and deployment defaults."""

    device_registry_path: str = field(
        default_factory=lambda: os.environ.get(
            "EDGE_REGISTRY_PATH",
            os.path.expanduser("~/.edge-train/devices.json"),
        )
    )
    default_transport: str = "http"
    connection_timeout_sec: int = 30
    retry_count: int = 3
    verify_deployment: bool = True
    default_version: str = "0.1.0"
    push_endpoint: str = "/api/v1/model"
    checksum_endpoint: str = "/api/v1/checksum"
