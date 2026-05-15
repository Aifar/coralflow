"""deploy command — push a TFLite model to an edge device."""

import asyncio
import sys

import click

from edge_train.edge.config import EdgeConfig
from edge_train.edge.deploy import deploy_model as _deploy_model
from edge_train.edge.registry import DeviceRegistry


@click.command()
@click.option("--model", "-m", required=True, help="Path to .tflite model file")
@click.option("--device", "-d", default=None, help="Device ID from registry")
@click.option("--host", default=None, help="Device hostname or IP (direct deploy)")
@click.option("--port", default=8080, type=int, help="Device HTTP port")
@click.option("--version", default=None, help="Model version string")
@click.option(
    "--modality",
    default="text",
    type=click.Choice(["text", "image", "table"]),
    help="Model modality",
)
@click.option("--timeout", default=30, type=int, help="Connection timeout in seconds")
@click.option("--no-verify", is_flag=True, help="Skip checksum verification")
@click.option("--registry", default=None, help="Path to device registry JSON file")
def deploy(
    model: str,
    device: str | None,
    host: str | None,
    port: int,
    version: str | None,
    modality: str,
    timeout: int,
    no_verify: bool,
    registry: str | None,
):
    """Deploy a TFLite model to an edge device.

    Provide --device to look up a device from the registry, or --host
    to deploy to a device directly by address.
    """
    cfg = EdgeConfig(
        connection_timeout_sec=timeout,
        verify_deployment=not no_verify,
    )

    reg = None
    if device:
        reg_path = registry or cfg.device_registry_path
        reg = DeviceRegistry(reg_path)

    click.echo("  Packaging model...", nl=False)
    sys.stdout.flush()

    try:
        result = asyncio.run(
            _deploy_model(
                model,
                device_id=device,
                host=host,
                port=port,
                registry=reg,
                edge_config=cfg,
                version=version,
                modality=modality,
            )
        )
    except Exception as e:
        click.echo(" failed.")
        click.echo(f"  Error: {e}", err=True)
        sys.exit(1)

    if result.success:
        click.echo(" done.")
        click.echo(f"  Deployed model to {result.device_id}")
        click.echo(f"  Model: {result.model_path}")
        if result.manifest:
            click.echo(f"  Version: {result.manifest.version}")
            click.echo(f"  SHA-256: {result.manifest.sha256}")
            size_kb = result.manifest.model_size_bytes / 1024
            click.echo(f"  Size: {size_kb:.1f} KB")
            click.echo(f"  Modality: {result.manifest.modality}")
        click.echo(f"  Elapsed: {result.elapsed_sec:.2f}s")
    else:
        click.echo(" failed.")
        click.echo(f"  {result.error}", err=True)
        sys.exit(1)
