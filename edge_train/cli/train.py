"""train command — local training by default, Vertex AI AutoML with --cloud."""

import sys
import time

import click

from edge_train.config import load_config


@click.command()
@click.option("--dataset", "-d", required=True, help="Path to training dataset CSV")
@click.option(
    "--type",
    "modality",
    type=click.Choice(["text", "image", "table"]),
    default=None,
    help="Override modality auto-detection",
)
@click.option("--target", default=None, help="Target column name (for CSV)")
@click.option(
    "--timeout", default=30, help="Max training wait time in minutes (cloud only)"
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory for trained model (default: ./model_output)",
)
@click.option("--epochs", default=None, type=int, help="Training epochs (local only)")
@click.option(
    "--cloud",
    is_flag=True,
    default=False,
    help="Use Vertex AI AutoML instead of local training",
)
def train(
    dataset: str,
    modality: str | None,
    target: str | None,
    timeout: int,
    output: str | None,
    epochs: int | None,
    cloud: bool,
):
    """Train a model — local by default, or Vertex AI AutoML with --cloud.

    Local training works entirely offline with no API keys.
    Currently supports text classification; image and table coming soon.
    """
    gcp, _, train_cfg, _ = load_config()

    from edge_train.datasets import infer_modality_from_path

    resolved_modality = modality or infer_modality_from_path(dataset)
    out_dir = output or train_cfg.output_dir
    num_epochs = epochs or train_cfg.local_epochs

    if cloud:
        _train_cloud(dataset, resolved_modality, target, timeout, gcp, train_cfg)
    else:
        _train_local(dataset, resolved_modality, target, out_dir, num_epochs)


def _train_cloud(dataset, modality, target, timeout, gcp, train_cfg):
    """Existing Vertex AI AutoML path."""
    from edge_train.cloud import Modality, submit_automl_job, poll_job

    if not gcp.is_valid():
        click.echo(
            "Error: GCP not configured. Set GCP_PROJECT and GCP_STAGING_BUCKET env vars.",
            err=True,
        )
        sys.exit(1)

    resolved = Modality(modality) if isinstance(modality, str) else Modality(modality)
    click.echo(f"  Modality: {resolved.value}")
    click.echo(f"  Dataset: {dataset}")
    click.echo(f"  GCP Project: {gcp.project_id}")
    click.echo("  Submitting training job to Vertex AI AutoML...")

    try:
        job_name = submit_automl_job(
            project=gcp.project_id,
            location=gcp.location,
            dataset_path=dataset,
            modality=resolved,
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


def _train_local(dataset, modality, target, output_dir, epochs):
    """Local training path — no cloud required."""
    click.echo(f"  Modality: {modality}")
    click.echo(f"  Dataset: {dataset}")
    click.echo("  Training locally...")

    if modality == "text":
        from edge_train.trainer import train_text_classifier

        try:
            model_path = train_text_classifier(
                dataset_path=dataset,
                target_column=target,
                output_dir=output_dir,
                epochs=epochs,
            )
        except Exception as e:
            click.echo(f"  Training failed: {e}", err=True)
            sys.exit(1)

        click.echo(f"  Model saved to: {model_path}")
        click.echo(
            f"  Next: edge-train validate --model {model_path} --output model.tflite"
        )
    else:
        click.echo(
            f"  Local training for '{modality}' is not yet supported. "
            f"Use --cloud for Vertex AI AutoML.",
            err=True,
        )
        sys.exit(1)
