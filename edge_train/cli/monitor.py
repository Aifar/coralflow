"""monitor command — OpenTelemetry tracing via Arize Phoenix."""

import webbrowser

import click


def _derive_dashboard_url(endpoint: str) -> str:
    """Derive the Phoenix dashboard URL from the collector endpoint."""
    if "/v1/traces" in endpoint:
        return endpoint.rsplit("/v1/traces", 1)[0]
    return "https://app.phoenix.arize.com"


@click.command()
@click.option("--dashboard", is_flag=True, help="Open Phoenix dashboard in browser")
@click.option("--status", is_flag=True, help="Print monitoring configuration status")
def monitor(dashboard: bool, status: bool):
    """Monitor deployed model performance via Arize Phoenix.

    Registers OpenTelemetry tracing with phoenix.otel.register() so
    spans are exported to the Phoenix collector.

    Requires PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT environment
    variables.  Optionally set PHOENIX_PROJECT_NAME (default: edge-train).
    """
    from edge_train.config import load_config

    _, arize, _, _ = load_config()

    if status:
        click.echo(f"  Phoenix Collector Endpoint: {arize.collector_endpoint}")
        click.echo(f"  Project Name:            {arize.project_name}")
        click.echo(
            f"  API Key:                 {'configured' if arize.api_key else 'not set'}"
        )

    if not arize.is_valid():
        click.echo(
            "  Phoenix not configured. "
            "Set PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT."
        )
        return

    try:
        from phoenix.otel import register

        register(
            endpoint=arize.collector_endpoint,
            project_name=arize.project_name,
            auto_instrument=False,
        )
    except ImportError:
        click.echo(
            "  arize-phoenix-otel is not installed. "
            "Run: pip install arize-phoenix-otel",
            err=True,
        )
        return
    except Exception as exc:
        if status:
            click.echo(f"  Status:                  not connected ({exc})")
        else:
            click.echo(f"  Failed to register Phoenix tracing: {exc}", err=True)
        return

    if status:
        click.echo("  Status:                  connected")
    else:
        click.echo("  Phoenix OTEL tracing registered.")
        click.echo(f"  Endpoint: {arize.collector_endpoint}")
        click.echo(f"  Project:  {arize.project_name}")

    if dashboard:
        dashboard_url = _derive_dashboard_url(arize.collector_endpoint)
        click.echo(f"  Opening Phoenix dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)
