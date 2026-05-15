"""train command — submit a training job to Vertex AI AutoML."""

import sys
import time

import click

from edge_train.config import load_config
from edge_train.cloud import submit_automl_job, poll_job, Modality


@click.command()
@click.option("--dataset", "-d", required=True, help="Path to training dataset CSV")
@click.option("--type", "modality", type=click.Choice(["text", "image", "table"]), default=None,
              help="Override modality auto-detection")
@click.option("--target", default=None, help="Target column name (for CSV)")
@click.option("--timeout", default=30, help="Max training wait time in minutes")
def train(dataset: str, modality: str | None, target: str | None, timeout: int):
    """Train a model using Vertex AI AutoML.

    The modality is auto-detected from the dataset unless --type is specified.
    """
    gcp, _, train_cfg, _ = load_config()
    if not gcp.is_valid():
        click.echo(
            "Error: GCP not configured. Set GCP_PROJECT and GCP_STAGING_BUCKET env vars.",
            err=True,
        )
        sys.exit(1)

    from edge_train.datasets import infer_modality_from_path

    resolved_modality = Modality(modality) if modality else Modality(infer_modality_from_path(dataset))
    click.echo(f"  Modality: {resolved_modality.value}")
    click.echo(f"  Dataset: {dataset}")
    click.echo(f"  GCP Project: {gcp.project_id}")
    click.echo("  Submitting training job to Vertex AI AutoML...")

    try:
        job_name = submit_automl_job(
            project=gcp.project_id,
            location=gcp.location,
            dataset_path=dataset,
            modality=resolved_modality,
            target_column=target,
            staging_bucket=gcp.staging_bucket,
        )
    except Exception as e:
        click.echo(f"Error submitting job: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Job submitted: {job_name}")
    click.echo(f"  Training typically takes 5-30 minutes...")

    timeout_min = min(timeout, train_cfg.training_timeout_min)
    deadline = time.time() + timeout_min * 60

    try:
        result = poll_job(job_name, deadline=deadline)
    except TimeoutError:
        click.echo(
            f"  Training still running after {timeout_min} min. "
            f"Check status in Google Cloud Console.",
            err=True,
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"  Training failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"  Model saved to: {result.get('model_path', 'unknown')}")
    click.echo(f"  Evaluation accuracy: {result.get('accuracy', '?')}")
