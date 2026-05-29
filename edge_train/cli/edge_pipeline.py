"""coralflow-edge-pipeline — train → validate → deploy in one command."""

from __future__ import annotations

import shutil
import subprocess
import sys

import click


def _coralflow_cmd() -> list[str]:
    exe = shutil.which("coralflow")
    if exe:
        return [exe]
    return [sys.executable, "-m", "edge_train.cli"]


def _run_step(label: str, args: list[str]) -> None:
    cmd = _coralflow_cmd() + args
    click.echo(f"\n  [{label}] {' '.join(args)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@click.command()
@click.argument("dataset")
@click.option(
    "--output",
    "-o",
    default="./model_output",
    show_default=True,
    help="SavedModel output directory",
)
@click.option(
    "--tflite",
    default="./model.tflite",
    show_default=True,
    help="TFLite output path",
)
@click.option(
    "--device",
    "-d",
    default=None,
    help="Device id from EDGE_DEVICES, or 'all' (default: EDGE_DEFAULT_DEVICE)",
)
@click.option("--epochs", type=int, default=None, help="Local training epochs")
@click.option(
    "--force", is_flag=True, help="Force retrain even if duplicate fingerprint"
)
@click.option("--skip-train", is_flag=True, help="Skip training step")
@click.option("--skip-validate", is_flag=True, help="Skip TFLite validation step")
@click.option("--skip-deploy", is_flag=True, help="Skip edge deploy step")
@click.option(
    "--host", default=None, help="Deploy directly to host (bypass EDGE_DEVICES)"
)
@click.option(
    "--port", default=8080, type=int, show_default=True, help="Port with --host"
)
def main(
    dataset: str,
    output: str,
    tflite: str,
    device: str | None,
    epochs: int | None,
    force: bool,
    skip_train: bool,
    skip_validate: bool,
    skip_deploy: bool,
    host: str | None,
    port: int,
) -> None:
    """Run the edge ML pipeline: train → validate → deploy.

    Example:

        coralflow-edge-pipeline ./data/urgent.csv

    Requires EDGE_DEVICES / EDGE_DEFAULT_DEVICE in .env (unless --host is set).
    """
    click.echo("  CoralFlow edge pipeline")
    click.echo(f"  Dataset: {dataset}")

    if not skip_train:
        train_args = ["train", "-d", dataset, "-o", output]
        if epochs is not None:
            train_args.extend(["--epochs", str(epochs)])
        if force:
            train_args.append("--force")
        _run_step("train", train_args)

    if not skip_validate:
        _run_step("validate", ["validate", "-m", output, "-o", tflite])

    if not skip_deploy:
        deploy_args = ["deploy", "-m", tflite]
        if host:
            deploy_args.extend(["--host", host, "--port", str(port)])
        elif device:
            deploy_args.extend(["-d", device])
        _run_step("deploy", deploy_args)

    click.echo("\n  Edge pipeline complete.")


if __name__ == "__main__":
    main()
