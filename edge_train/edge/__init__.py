"""Edge device SDK — package, push, and verify TFLite models on devices."""

from edge_train.edge.deploy import deploy_model, DeployResult
from edge_train.edge.model import ModelPackage, ModelManifest
from edge_train.edge.registry import DeviceRegistry, DeviceInfo
from edge_train.edge.transport import BaseTransport, create_transport
from edge_train.edge.config import EdgeConfig

__all__ = [
    "deploy_model",
    "DeployResult",
    "ModelPackage",
    "ModelManifest",
    "DeviceRegistry",
    "DeviceInfo",
    "BaseTransport",
    "create_transport",
    "EdgeConfig",
]
