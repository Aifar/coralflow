"""simulate command — smoke-test inference with Phoenix-visible sample runs."""

import sys

import click

from edge_train.config import load_config
from edge_train.simulate import format_simulate_command, run_simulation


@click.command()
@click.option("--model", "-m", default=None, help="Local SavedModel directory")
@click.option(
    "--endpoint",
    "-e",
    default=None,
    help="Vertex AI endpoint resource name",
)
@click.option(
    "--modality",
    type=click.Choice(["text", "table", "image", "video"]),
    default=None,
    help="Vertex endpoint modality (auto-detected from deployment registry when possible)",
)
@click.option(
    "--count",
    "-n",
    default=5,
    show_default=True,
    type=int,
    help="Number of sample predictions to run",
)
@click.option(
    "--dataset",
    "-d",
    default=None,
    help="Optional CSV to draw sample inputs from (text/table/image paths)",
)
@click.option("--image", default=None, help="Sample image path (image modality)")
@click.option(
    "--gcs-uri",
    default=None,
    help="Sample GCS URI for image or video modality",
)
@click.option("--log", "-l", "log_path", default=None, help="Prediction log file path")
def simulate(
    model: str | None,
    endpoint: str | None,
    modality: str | None,
    count: int,
    dataset: str | None,
    image: str | None,
    gcs_uri: str | None,
    log_path: str | None,
):
    """Run sample predictions so traces appear in Arize Phoenix.

    Use after deploy to verify monitoring end-to-end:

      coralflow simulate --endpoint projects/.../endpoints/ID --modality text
      coralflow simulate --model ./model_output

    Requires PHOENIX_COLLECTOR_ENDPOINT (+ PHOENIX_API_KEY for cloud).
    """
    if bool(model) == bool(endpoint):
        click.echo(
            "Error: provide exactly one of --model (local) or --endpoint (Vertex).",
            err=True,
        )
        sys.exit(1)

    if count < 1:
        click.echo("Error: --count must be at least 1.", err=True)
        sys.exit(1)

    _, arize, train_cfg, _ = load_config()
    if not arize.is_valid():
        click.echo(
            "Error: Phoenix not configured. Set PHOENIX_COLLECTOR_ENDPOINT "
            "(and PHOENIX_API_KEY for Phoenix Cloud).",
            err=True,
        )
        sys.exit(1)

    click.echo("  Running Phoenix smoke-test predictions...")
    try:
        result = run_simulation(
            model=model,
            endpoint=endpoint,
            modality=modality,
            count=count,
            dataset=dataset,
            image=image,
            gcs_uri=gcs_uri,
            log_path=log_path or train_cfg.prediction_log_path,
        )
    except ValueError as exc:
        click.echo(f"  Error: {exc}", err=True)
        sys.exit(1)
    except RuntimeError as exc:
        click.echo(f"  Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"  Simulate failed: {exc}", err=True)
        sys.exit(1)

    click.echo(f"  Sent {result.count} sample predictions.")
    click.echo(f"  Log file: {result.log_path}")
    click.echo(f"  Phoenix project: {result.project_name}")
    click.echo(f"  Dashboard: {result.dashboard_url}")
    click.echo("  Open the dashboard to view traces, latency, and prediction spans.")
