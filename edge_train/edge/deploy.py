"""Deploy orchestrator — composes transport, registry, and packaging."""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from edge_train.edge.config import EdgeConfig
from edge_train.edge.model import ModelManifest, ModelPackage
from edge_train.edge.registry import DeviceInfo, DeviceRegistry
from edge_train.edge.transport import BaseTransport, create_transport


@dataclass
class DeployResult:
    """Outcome of a deploy_model call."""

    success: bool
    device_id: str = ""
    model_path: str = ""
    elapsed_sec: float = 0.0
    manifest: ModelManifest | None = None
    error: str = ""


async def deploy_model(
    model_path: str,
    *,
    device_id: str | None = None,
    host: str | None = None,
    port: int = 8080,
    registry: DeviceRegistry | None = None,
    edge_config: EdgeConfig | None = None,
    version: str | None = None,
    modality: str = "text",
    transport: BaseTransport | None = None,
) -> DeployResult:
    """Deploy a TFLite model to an edge device.

    Steps:
    1. Validate model file
    2. Package model with manifest
    3. Resolve device (registry lookup or ephemeral)
    4. Create transport and connect
    5. Push model to device
    6. Verify checksum
    7. Disconnect and report
    """
    start = time.monotonic()
    cfg = edge_config or EdgeConfig()
    ver = version or cfg.default_version

    # Step 1: Validate model path
    model = Path(model_path)
    if not model.exists():
        return DeployResult(
            success=False,
            model_path=model_path,
            error=f"Model not found: {model_path}",
        )
    if model.suffix.lower() != ".tflite":
        return DeployResult(
            success=False,
            model_path=model_path,
            error=f"Not a TFLite model: {model_path}",
        )

    # Step 2: Package
    try:
        package = ModelPackage(model, version=ver, modality=modality)
    except (FileNotFoundError, ValueError) as e:
        return DeployResult(success=False, model_path=model_path, error=str(e))

    # Step 3: Resolve device
    resolved: DeviceInfo | None = None
    if device_id and registry:
        resolved = registry.resolve(device_id)
        if resolved is None:
            return DeployResult(
                success=False,
                model_path=model_path,
                error=f"Unknown device: {device_id}",
            )
    elif host:
        resolved = DeviceInfo(device_id=f"ephemeral-{host}", host=host, port=port)
    else:
        return DeployResult(
            success=False,
            model_path=model_path,
            error="No device specified: provide --device or --host",
        )

    # Step 4-7: Transport
    tport = transport or create_transport(
        cfg.default_transport,
        timeout=cfg.connection_timeout_sec,
        retries=cfg.retry_count,
    )

    try:
        connected = await tport.connect(resolved)
        if not connected:
            return DeployResult(
                success=False,
                device_id=resolved.device_id,
                model_path=model_path,
                error=f"Connection refused: {resolved.host}:{resolved.port}",
                elapsed_sec=time.monotonic() - start,
            )

        pushed = await tport.push_model(package)
        if not pushed:
            return DeployResult(
                success=False,
                device_id=resolved.device_id,
                model_path=model_path,
                error=f"Push failed after {cfg.retry_count} retries",
                elapsed_sec=time.monotonic() - start,
            )

        if cfg.verify_deployment:
            verified = await tport.verify_checksum(package.manifest.sha256)
            if not verified:
                return DeployResult(
                    success=False,
                    device_id=resolved.device_id,
                    model_path=model_path,
                    error="Checksum mismatch",
                    elapsed_sec=time.monotonic() - start,
                )
    except (asyncio.TimeoutError, OSError) as e:
        return DeployResult(
            success=False,
            device_id=resolved.device_id,
            model_path=model_path,
            error=f"Timeout after {cfg.connection_timeout_sec}s: {e}",
            elapsed_sec=time.monotonic() - start,
        )
    finally:
        await tport.disconnect()

    elapsed = time.monotonic() - start
    return DeployResult(
        success=True,
        device_id=resolved.device_id,
        model_path=str(model),
        elapsed_sec=elapsed,
        manifest=package.manifest,
    )
