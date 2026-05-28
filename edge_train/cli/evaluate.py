"""evaluate command — batch test-set accuracy via Vertex endpoint."""

import sys

import click

from edge_train.config import load_config
from edge_train.evaluate import format_evaluation_summary, run_vertex_evaluation
from edge_train.inference.phoenix import (
    apply_phoenix_prepare,
    prepare_phoenix_for_inference,
)


@click.command()
@click.option(
    "--endpoint",
    "-e",
    required=True,
    help="Vertex AI endpoint resource name",
)
@click.option(
    "--modality",
    type=click.Choice(["text", "table", "image", "video"]),
    default=None,
    help="Vertex endpoint modality (auto-detected from deployment registry when possible)",
)
@click.option(
    "--dataset",
    "-d",
    "dataset_path",
    required=True,
    help="Labeled test JSONL (local path or gs://.../test.jsonl)",
)
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Evaluate only the first N rows (default: all)",
)
@click.option(
    "--batch-size",
    default=16,
    show_default=True,
    type=int,
    help="Instances per Vertex predict request",
)
@click.option("--log", "-l", "log_path", default=None, help="Prediction log file path")
def evaluate(
    endpoint: str,
    modality: str | None,
    dataset_path: str,
    limit: int | None,
    batch_size: int,
    log_path: str | None,
):
    """Evaluate a Vertex endpoint on a labeled test JSONL and log to Phoenix.

    Example (NEU defect classification):

      coralflow evaluate \\
        --endpoint projects/.../locations/us-central1/endpoints/ID \\
        --modality image \\
        --dataset gs://coralflow/neu-cls/test.jsonl
    """
    gcp, arize, _, _ = load_config()
    if not gcp.is_valid():
        click.echo(
            "Error: GCP not configured. Set GCP_PROJECT for Vertex evaluate.",
            err=True,
        )
        sys.exit(1)
    if not arize.is_valid():
        click.echo(
            "Error: Phoenix not configured. Set PHOENIX_COLLECTOR_ENDPOINT "
            "(and PHOENIX_API_KEY for Phoenix Cloud).",
            err=True,
        )
        sys.exit(1)

    phoenix_active = apply_phoenix_prepare(
        prepare_phoenix_for_inference(required=True, interactive=True)
    )

    try:
        result = run_vertex_evaluation(
            endpoint=endpoint,
            modality=modality or "image",
            dataset_path=dataset_path,
            project=gcp.project_id,
            location=gcp.location,
            log_path=log_path,
            limit=limit,
            batch_size=batch_size,
            phoenix_active=phoenix_active,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(format_evaluation_summary(result))
