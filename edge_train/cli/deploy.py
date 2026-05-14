"""deploy command — push a TFLite model to an edge device."""

import click


@click.command()
@click.option("--model", "-m", required=True, help="Path to TFLite model file")
@click.option("--device", "-d", default=None, help="Target device identifier")
def deploy(model: str, device: str | None):
    """Deploy a TFLite model to an edge device.

    Requires edge device SDK to be installed on the target.
    Full implementation coming in Week 2.
    """
    click.echo("  Deploy not yet available in this release.")
    click.echo("  The TFLite model is ready at the specified path.")
    click.echo("  Full OTA deploy support coming in edge-train v0.2.0.")
