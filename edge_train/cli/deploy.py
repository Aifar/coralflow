"""deploy command — edge device (TFLite) or Vertex AI endpoint deployment."""

import asyncio
import os
import sys

import click

from edge_train.edge.config import EdgeConfig
from edge_train.edge.deploy import deploy_model as _deploy_model
from edge_train.edge.registry import (
    DeviceInfo,
    load_device_registry,
    resolve_deploy_targets,
)


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="Local .tflite path (edge) or Vertex model resource name (with --cloud)",
)
@click.option(
    "--device",
    "-d",
    default=None,
    help="Device ID from EDGE_DEVICES (.env), 'all', or EDGE_DEFAULT_DEVICE if omitted",
)
@click.option(
    "--host", default=None, help="Device hostname or IP (bypasses .env registry)"
)
@click.option(
    "--port", default=8080, type=int, help="Device HTTP port (with --host only)"
)
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
@click.option(
    "--list-devices",
    is_flag=True,
    help="List edge gateways configured in EDGE_DEVICES (.env) and exit",
)
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
    list_devices: bool,
    cloud: bool,
    machine_type: str | None,
    display_name: str | None,
    simulate: bool,
    simulate_count: int,
):
    """Deploy a model for inference.

    Edge (default): push TFLite to gateways listed in EDGE_DEVICES (.env).

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

    registry = load_device_registry()
    if list_devices:
        _print_edge_devices(registry.list_devices())
        return

    cfg = EdgeConfig(
        connection_timeout_sec=timeout,
        verify_deployment=not no_verify,
    )

    if host:
        ok = _deploy_edge_single(
            model=model,
            device_id=None,
            host=host,
            port=port,
            registry=None,
            cfg=cfg,
            version=version,
            modality=modality,
            simulate=simulate,
            simulate_count=simulate_count,
        )
        if not ok:
            sys.exit(1)
        return

    try:
        targets = resolve_deploy_targets(registry, device, cfg.default_device)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    failures = 0
    for index, target in enumerate(targets):
        if len(targets) > 1:
            click.echo(f"  [{index + 1}/{len(targets)}] {target.device_id} ...")
        ok = _deploy_edge_single(
            model=model,
            device_id=target.device_id,
            host=None,
            port=target.port,
            registry=registry,
            cfg=cfg,
            version=version,
            modality=modality,
            simulate=simulate and len(targets) == 1,
            simulate_count=simulate_count,
            quiet_packaging=index > 0,
        )
        if not ok:
            failures += 1

    if failures:
        sys.exit(1)


def _print_edge_devices(devices: list[DeviceInfo]) -> None:
    if not devices:
        click.echo("  No edge devices in EDGE_DEVICES (.env).")
        click.echo(
            "  Example:\n"
            '    EDGE_DEVICES=[{"device_id":"line1-gw","host":"192.168.1.50","port":8080}]\n'
            "    EDGE_DEFAULT_DEVICE=line1-gw"
        )
        return
    click.echo("  Edge gateways (EDGE_DEVICES):")
    for dev in devices:
        label = f" — {dev.label}" if dev.label else ""
        click.echo(f"    {dev.device_id}: http://{dev.host}:{dev.port}{label}")


def _deploy_edge_single(
    *,
    model: str,
    device_id: str | None,
    host: str | None,
    port: int,
    registry,
    cfg: EdgeConfig,
    version: str | None,
    modality: str,
    simulate: bool,
    simulate_count: int,
    quiet_packaging: bool = False,
) -> bool:
    if not quiet_packaging:
        click.echo("  Packaging model...", nl=False)
        sys.stdout.flush()

    try:
        result = asyncio.run(
            _deploy_model(
                model,
                device_id=device_id,
                host=host,
                port=port,
                registry=registry,
                edge_config=cfg,
                version=version,
                modality=modality,
            )
        )
    except Exception as e:
        if not quiet_packaging:
            click.echo(" failed.")
        click.echo(f"  Error: {e}", err=True)
        return False

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
        if not quiet_packaging:
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
        if simulate:
            _print_simulate_hint(model=_edge_simulate_model_hint(), modality=modality)
            _run_post_deploy_simulate(
                model=_edge_simulate_model_hint(),
                modality=modality,
                count=simulate_count,
            )
        return True

    if not quiet_packaging:
        click.echo(" failed.")
    click.echo(f"  {result.error}", err=True)
    return False


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
