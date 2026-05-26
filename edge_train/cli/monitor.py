"""monitor command — OpenTelemetry tracing via Arize Phoenix + retrain loop."""

import json
import sys
import webbrowser
from pathlib import Path

import click


def _derive_dashboard_url(endpoint: str) -> str:
    from edge_train.phoenix_util import derive_dashboard_url

    return derive_dashboard_url(endpoint)


@click.command()
@click.option("--dashboard", is_flag=True, help="Open Phoenix dashboard in browser")
@click.option("--status", is_flag=True, help="Print monitoring configuration status")
@click.option(
    "--retrain",
    is_flag=True,
    help="Check prediction log and retrain if accuracy is low",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Retrain accuracy threshold (default from config)",
)
@click.option(
    "--dataset",
    "-d",
    default=None,
    help="Path to original training CSV (for merging labeled data)",
)
@click.option(
    "--output", "-o", default=None, help="Output directory for retrained model"
)
def monitor(
    dashboard: bool,
    status: bool,
    retrain: bool,
    threshold: float | None,
    dataset: str | None,
    output: str | None,
):
    """Monitor model performance via Arize Phoenix and optionally retrain.

    Registers OpenTelemetry tracing with phoenix.otel.register() so
    spans are exported to the Phoenix collector.

    With --retrain, reads the prediction log (JSON lines), finds entries
    with ground_truth labels, and retrains if accuracy is below threshold.

    Requires PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT for tracing.
    """
    from edge_train.config import load_config

    _, arize, train_cfg, _ = load_config()

    if status:
        click.echo(f"  Phoenix Collector Endpoint: {arize.collector_endpoint}")
        click.echo(f"  Project Name:            {arize.project_name}")
        click.echo(
            f"  API Key:                 {'configured' if arize.api_key else 'not set'}"
        )

    if retrain:
        _check_and_retrain(
            arize=arize,
            train_cfg=train_cfg,
            threshold=threshold,
            dataset_path=dataset,
            output_dir=output,
        )
        return

    if not arize.is_valid():
        click.echo(
            "  Phoenix not configured. "
            "Set PHOENIX_API_KEY and PHOENIX_COLLECTOR_ENDPOINT."
        )
        return

    from edge_train.phoenix_util import check_phoenix_running, ensure_phoenix_ready

    probe = check_phoenix_running(arize)
    if status:
        if not probe.reachable:
            click.echo(f"  Status:                  not running ({probe.detail})")
        else:
            click.echo("  Status:                  reachable")

    phoenix_active, phoenix_err = ensure_phoenix_ready(arize)
    if not phoenix_active:
        click.echo(phoenix_err, err=True)
        return

    if status:
        click.echo("  Status:                  connected (OTEL registered)")
    else:
        click.echo("  Phoenix OTEL tracing registered.")
        click.echo(f"  Endpoint: {arize.collector_endpoint}")
        click.echo(f"  Project:  {arize.project_name}")

    if dashboard:
        dashboard_url = _derive_dashboard_url(arize.collector_endpoint)
        click.echo(f"  Opening Phoenix dashboard: {dashboard_url}")
        webbrowser.open(dashboard_url)


def _check_and_retrain(arize, train_cfg, threshold, dataset_path, output_dir):
    """Read prediction log, compute accuracy on labeled data, retrain if needed."""
    from edge_train.retrain import (
        compute_accuracy,
        labeled_entries,
        read_prediction_log,
    )

    log_path = Path(train_cfg.prediction_log_path)
    if not log_path.exists():
        click.echo(f"  No prediction log found at: {log_path}")
        return

    entries = read_prediction_log(log_path)
    labeled = labeled_entries(entries)
    if not labeled:
        click.echo(f"  No labeled entries in prediction log ({len(entries)} total).")
        click.echo(
            "  Add 'ground_truth' fields to prediction_log.jsonl entries to enable retraining."
        )
        return

    min_samples = train_cfg.retrain_min_samples
    if len(labeled) < min_samples:
        click.echo(f"  Only {len(labeled)} labeled entries (need {min_samples}).")
        return

    accuracy, correct, total = compute_accuracy(entries)
    accuracy_threshold = threshold or train_cfg.retrain_accuracy_threshold

    click.echo(f"  Labeled predictions: {total}")
    click.echo(f"  Current accuracy:    {accuracy:.2%}")
    click.echo(f"  Threshold:           {accuracy_threshold:.2%}")

    if accuracy >= accuracy_threshold:
        click.echo(f"  Accuracy is above threshold. No retrain needed.")
        return

    click.echo(f"  Accuracy below threshold — retraining...")

    if not dataset_path:
        click.echo(
            "  Error: --dataset is required for retraining. "
            "Provide the original training CSV to merge with new labeled data.",
            err=True,
        )
        sys.exit(1)

    _do_retrain(labeled, dataset_path, output_dir or train_cfg.output_dir, train_cfg)


def _do_retrain(labeled_entries, dataset_path, output_dir, train_cfg):
    """Merge labeled data with original dataset and retrain."""
    from edge_train.retrain import retrain_from_labeled

    click.echo(f"  Merging {len(labeled_entries)} labeled rows into training data...")

    model_path = retrain_from_labeled(
        labeled_entries,
        dataset_path,
        output_dir,
        train_cfg.local_epochs,
    )

    click.echo(f"  Retrained model saved to: {model_path}")
    click.echo(
        f"  Next: coralflow validate --model {model_path} --output {model_path}.tflite"
    )
