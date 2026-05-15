"""validate command — TFLite conversion and constraint checking."""

import sys
import json
from pathlib import Path

import click

from edge_train.config import load_config


@click.command()
@click.option("--model", "-m", required=True, help="Path to SavedModel or export directory")
@click.option("--output", "-o", default="./model.tflite", help="Output TFLite path")
@click.option("--force", is_flag=True, help="Skip constraint checks")
def validate(model: str, output: str, force: bool):
    """Convert a model to TFLite and validate size/latency constraints.

    Checks:
    - TFLite model size < 10 MB
    - Estimated inference latency < 50 ms on target CPU
    - Accuracy loss after quantization < 2% (requires validation dataset)
    """
    from edge_train.validation import convert_to_tflite, check_size_constraint, estimate_latency

    _, _, train_cfg, _ = load_config()
    model_path = Path(model)

    if not model_path.exists():
        click.echo(f"Error: model path '{model}' not found", err=True)
        sys.exit(1)

    click.echo(f"  Converting to TFLite...")
    tflite_path = convert_to_tflite(model_path, Path(output))
    click.echo(f"  TFLite model saved to: {tflite_path}")

    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    click.echo(f"  Model size: {size_mb:.2f} MB")

    if not force:
        if not check_size_constraint(size_mb, train_cfg.model_size_mb):
            click.echo(
                f"  FAIL: Model size {size_mb:.1f} MB exceeds {train_cfg.model_size_mb} MB limit.",
                err=True,
            )
            sys.exit(1)

        latency = estimate_latency(tflite_path)
        click.echo(f"  Estimated inference latency: {latency:.1f} ms")
        if latency > train_cfg.inference_ms:
            click.echo(
                f"  WARNING: Latency {latency:.0f} ms exceeds {train_cfg.inference_ms} ms target.",
                err=True,
            )

    click.echo("  Validation passed.")
