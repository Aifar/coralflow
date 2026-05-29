"""Edge device SDK — package, push, and verify TFLite models on devices."""

from edge_train.edge.deploy import deploy_model, DeployResult
from edge_train.edge.model import ModelPackage, ModelManifest
from edge_train.edge.registry import (
    DeviceInfo,
    DeviceRegistry,
    load_device_registry,
    parse_edge_devices,
    resolve_deploy_targets,
)
from edge_train.edge.transport import BaseTransport, create_transport
from edge_train.edge.config import EdgeConfig

__all__ = [
    "deploy_model",
    "DeployResult",
    "ModelPackage",
    "ModelManifest",
    "DeviceRegistry",
    "DeviceInfo",
    "load_device_registry",
    "parse_edge_devices",
    "resolve_deploy_targets",
    "BaseTransport",
    "create_transport",
    "EdgeConfig",
]
