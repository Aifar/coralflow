"""monitor command — view model health in Arize AI."""

import click


@click.command()
@click.option("--dashboard", is_flag=True, help="Open Arize dashboard in browser")
@click.option("--status", is_flag=True, help="Print model health summary")
def monitor(dashboard: bool, status: bool):
    """Monitor deployed model performance via Arize AI.

    Requires ARIZE_API_KEY and ARIZE_SPACE_KEY environment variables.
    Full implementation coming in Week 2.
    """
    from edge_train.config import load_config

    _, arize, _, _ = load_config()

    if not arize.is_valid():
        click.echo("  Arize not configured. Set ARIZE_API_KEY and ARIZE_SPACE_KEY.")
        return

    click.echo("  Monitoring support coming in edge-train v0.2.0.")
    if dashboard:
        click.echo(f"  Arize endpoint: {arize.endpoint}")
