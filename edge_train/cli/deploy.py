"""deploy command — edge device (TFLite) or Vertex AI endpoint deployment."""

import asyncio
import os
import sys

import click

from edge_train.edge.config import EdgeConfig
from edge_train.edge.deploy import deploy_model as _deploy_model
from edge_train.edge.registry import DeviceRegistry


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="Local .tflite path (edge) or Vertex model resource name (with --cloud)",
)
@click.option(
    "--device", "-d", default=None, help="Device ID from registry (edge only)"
)
@click.option("--host", default=None, help="Device hostname or IP (edge only)")
@click.option("--port", default=8080, type=int, help="Device HTTP port (edge only)")
@click.option("--version", default=None, help="Model version string (edge only)")
@click.option(
    "--modality",
    default="text",
    type=click.Choice(["text", "image", "table", "video"]),
    help="Model modality",
)
@click.option("--timeout", default=30, type=int, help="Connection timeout in seconds")
@click.option(
    "--no-verify", is_flag=True, help="Skip checksum verification (edge only)"
)
@click.option("--registry", default=None, help="Path to device registry JSON file")
@click.option(
    "--cloud",
    is_flag=True,
    default=False,
    help="Deploy Vertex AI model to an online prediction endpoint",
)
@click.option(
    "--machine-type",
    default=None,
    help="Vertex machine type (default: n1-standard-2 or GCP_VERTEX_MACHINE_TYPE)",
)
@click.option(
    "--display-name",
    default=None,
    help="Vertex deployed model display name",
)
@click.option(
    "--simulate",
    is_flag=True,
    default=False,
    help="Run Phoenix smoke-test predictions after successful deploy",
)
@click.option(
    "--simulate-count",
    default=5,
    show_default=True,
    type=int,
    help="Number of sample predictions when using --simulate",
)
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
    cloud: bool,
    machine_type: str | None,
    display_name: str | None,
    simulate: bool,
    simulate_count: int,
):
    """Deploy a model for inference.

    Edge (default): push TFLite to a registered device or --host.

    Vertex (--cloud): deploy a trained Vertex model resource to an endpoint.
    Use coralflow predict --endpoint <endpoint> for Phoenix-monitored inference.
    """
    if cloud:
        _deploy_vertex(
            model,
            modality,
            machine_type,
            display_name,
            simulate=simulate,
            simulate_count=simulate_count,
        )
        return

    if not device and not host:
        click.echo(
            "Error: provide --device or --host for edge deploy, or use --cloud for Vertex.",
            err=True,
        )
        sys.exit(1)

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
        from edge_train.config import load_config
        from edge_train.deployments import DeploymentRecord, DeploymentRegistry

        _, arize, _, _ = load_config()
        DeploymentRegistry.load().add(
            DeploymentRecord(
                model_path=model,
                target="edge",
                modality=modality,
                device_id=result.device_id,
                phoenix_project=arize.project_name if arize.is_valid() else "",
            )
        )
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
        from edge_train.deployments import format_phoenix_monitoring_hint

        click.echo(format_phoenix_monitoring_hint(model_path=model))
        _print_simulate_hint(model=_edge_simulate_model_hint(), modality=modality)
        if simulate:
            _run_post_deploy_simulate(
                model=_edge_simulate_model_hint(),
                modality=modality,
                count=simulate_count,
            )
    else:
        click.echo(" failed.")
        click.echo(f"  {result.error}", err=True)
        sys.exit(1)


def _deploy_vertex(
    model_name: str,
    modality: str,
    machine_type: str | None,
    display_name: str | None,
    *,
    simulate: bool = False,
    simulate_count: int = 5,
) -> None:
    from edge_train.cloud.serving import deploy_model_to_vertex, is_vertex_resource
    from edge_train.config import GCPConfig, load_config
    from edge_train.deployments import (
        DeploymentRecord,
        DeploymentRegistry,
        format_phoenix_monitoring_hint,
    )

    if not is_vertex_resource(model_name):
        click.echo(
            "Error: --cloud expects a Vertex model resource name, e.g.\n"
            "  projects/PROJECT/locations/REGION/models/MODEL_ID",
            err=True,
        )
        sys.exit(1)

    gcp, arize, _, _ = load_config()
    if not gcp.is_valid():
        click.echo(
            "Error: GCP not configured. Set GCP_PROJECT and GCP_STAGING_BUCKET.",
            err=True,
        )
        sys.exit(1)

    machine = machine_type or os.environ.get("GCP_VERTEX_MACHINE_TYPE", "n1-standard-2")
    click.echo(f"  Deploying Vertex model to endpoint (machine: {machine})...")
    try:
        result = deploy_model_to_vertex(
            model_name,
            project=gcp.project_id,
            location=gcp.location,
            display_name=display_name,
            machine_type=machine,
        )
    except Exception as exc:
        click.echo(f"  Error: {exc}", err=True)
        sys.exit(1)

    DeploymentRegistry.load().add(
        DeploymentRecord(
            model_path=result.model_path,
            target="vertex",
            modality=modality,
            endpoint_name=result.endpoint_name,
            phoenix_project=arize.project_name if arize.is_valid() else "",
        )
    )

    click.echo(f"  Endpoint: {result.endpoint_name}")
    click.echo(
        format_phoenix_monitoring_hint(
            endpoint=result.endpoint_name,
            modality=modality,
        )
    )
    _print_simulate_hint(endpoint=result.endpoint_name, modality=modality)
    if simulate:
        _run_post_deploy_simulate(
            endpoint=result.endpoint_name,
            modality=modality,
            count=simulate_count,
        )


def _edge_simulate_model_hint() -> str | None:
    from edge_train.simulate import guess_local_saved_model

    return guess_local_saved_model()


def _print_simulate_hint(
    *,
    endpoint: str = "",
    model: str | None = None,
    modality: str = "text",
) -> None:
    from edge_train.simulate import format_simulate_command

    if endpoint:
        cmd = format_simulate_command(
            endpoint=endpoint,
            modality=modality,
        )
    elif model:
        cmd = format_simulate_command(model=model)
    else:
        cmd = "coralflow simulate --model <SavedModel> | --endpoint <vertex>"
    click.echo(f"  Simulate (Phoenix smoke test): {cmd}")


def _run_post_deploy_simulate(
    *,
    endpoint: str = "",
    model: str | None = None,
    modality: str = "text",
    count: int = 5,
) -> None:
    from edge_train.simulate import run_simulation

    if not endpoint and not model:
        click.echo(
            "  Simulate skipped: no local SavedModel found. "
            "Run coralflow simulate --endpoint <id> manually.",
            err=True,
        )
        return

    click.echo("  Running post-deploy simulate...")
    try:
        result = run_simulation(
            endpoint=endpoint or None,
            model=model,
            modality=modality if endpoint else None,
            count=count,
        )
    except Exception as exc:
        click.echo(f"  Simulate failed: {exc}", err=True)
        return

    click.echo(f"  Simulate sent {result.count} predictions to Phoenix.")
    click.echo(f"  Dashboard: {result.dashboard_url}")
