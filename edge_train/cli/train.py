"""train command — local training by default, Vertex AI AutoML with --cloud."""

import sys
import time
from datetime import datetime, timezone

import click

from edge_train.config import load_config


@click.command()
@click.option("--dataset", "-d", required=True, help="Path to training dataset CSV")
@click.option(
    "--type",
    "modality",
    type=click.Choice(["text", "image", "table", "video"]),
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
    help="Use Vertex AI cloud training (auto-routed: Gemini SFT / AutoML Tabular|Image|Video)",
)
@click.option(
    "--base-model",
    default=None,
    help="Gemini publisher model for cloud text fine-tuning (default: GCP_FINETUNE_MODEL)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Train even if the same dataset/config was trained before",
)
@click.option(
    "--detach",
    is_flag=True,
    default=False,
    help="Submit cloud job and exit without waiting (cloud only)",
)
@click.option(
    "--poll-every",
    type=int,
    default=None,
    help="Check cloud job status every N minutes until complete or timeout (cloud only)",
)
@click.option(
    "--purpose",
    "-p",
    default="",
    help="Human-readable project name saved in history (e.g. neu_cls_defect_classifier_v3)",
)
def train(
    dataset: str,
    modality: str | None,
    target: str | None,
    timeout: int,
    output: str | None,
    epochs: int | None,
    cloud: bool,
    base_model: str | None,
    force: bool,
    detach: bool,
    poll_every: int | None,
    purpose: str,
):
    """Train a model — local by default, or Vertex AI with --cloud.

    Local training works entirely offline with no API keys.
    Cloud training is auto-routed: Gemini fine-tuning (text), AutoML Tabular/Image/Video.
    """
    gcp, _, train_cfg, _ = load_config()

    from edge_train.datasets import infer_modality_from_path, resolve_dataset_path

    try:
        dataset_path, builtin_modality = resolve_dataset_path(dataset)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    resolved_modality = (
        modality or builtin_modality or infer_modality_from_path(dataset)
    )
    if resolved_modality == "unknown":
        click.echo(
            "Error: could not detect dataset modality. Use --type text|image|table|video.",
            err=True,
        )
        sys.exit(1)

    out_dir = output or train_cfg.output_dir
    num_epochs = epochs or train_cfg.local_epochs

    _print_training_startup_summary(cloud=cloud)

    if cloud:
        _train_cloud(
            dataset,
            dataset_path,
            resolved_modality,
            target,
            timeout,
            gcp,
            train_cfg,
            base_model=base_model,
            force=force,
            detach=detach,
            poll_every=poll_every,
            purpose=purpose,
        )
    else:
        _train_local(
            dataset,
            dataset_path,
            resolved_modality,
            target,
            out_dir,
            num_epochs,
            force=force,
            purpose=purpose,
        )


def _print_training_startup_summary(*, cloud: bool) -> None:
    from edge_train.training_history import TrainingHistory, format_startup_summary

    history = TrainingHistory.load()
    if cloud:
        history.sync_cloud_jobs()
    summary = format_startup_summary(history)
    if summary:
        click.echo(summary)
        click.echo("")


def _train_cloud(
    dataset_label,
    dataset_path,
    modality,
    target,
    timeout,
    gcp,
    train_cfg,
    base_model=None,
    force=False,
    detach=False,
    poll_every=None,
    purpose="",
):
    """Vertex AI cloud training with automatic service routing."""
    import sys

    from edge_train.cloud import (
        cloud_modality_supported,
        describe_finetune_base_model,
        plan_cloud_training,
        submit_automl_job,
    )
    from edge_train.cloud.training_wait import (
        DEFAULT_SCHEDULED_POLL_MIN,
        format_detach_message,
        format_post_submit_guidance,
        poll_job_scheduled,
        resolve_wait_strategy,
    )
    from edge_train.training_history import (
        TrainingHistory,
        TrainingRecord,
        format_duplicate_message,
        make_training_fingerprint,
    )

    supported, reason = cloud_modality_supported(modality)
    if not supported:
        click.echo(f"Error: {reason}", err=True)
        sys.exit(1)

    if not gcp.is_valid():
        click.echo(
            "Error: GCP not configured. Set GCP_PROJECT and GCP_STAGING_BUCKET env vars.",
            err=True,
        )
        sys.exit(1)

    try:
        plan = plan_cloud_training(dataset_path, modality)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"  Modality: {plan.modality}")
    click.echo(f"  Cloud method: {plan.label} ({plan.status})")
    click.echo(f"  Rationale: {plan.reason}")
    click.echo(f"  Dataset: {dataset_label}")
    if purpose:
        click.echo(f"  Project: {purpose}")
    if dataset_label != dataset_path:
        click.echo(f"  Resolved path: {dataset_path}")
    click.echo(f"  GCP Project: {gcp.project_id}")
    click.echo(f"  Staging bucket: {gcp.staging_bucket}")
    finetune_model = base_model or gcp.finetune_model
    if plan.method.value == "gemini_finetune":
        for line in describe_finetune_base_model(finetune_model, gcp.location):
            click.echo(f"  {line}")

    history = TrainingHistory.load()
    fingerprint = make_training_fingerprint(
        mode="cloud",
        dataset_path=dataset_path,
        modality=plan.modality,
        method=plan.method.value,
        target_column=target or "",
        base_model=finetune_model if plan.method.value == "gemini_finetune" else "",
    )
    action, existing = history.check_duplicate(fingerprint, force=force)
    if action == "skip_succeeded":
        click.echo(format_duplicate_message(action, existing))
        return
    job_name = ""
    if action == "resume_running" and existing and existing.job_name:
        click.echo(format_duplicate_message(action, existing))
        job_name = existing.job_name
    else:
        click.echo(f"  Submitting {plan.label} job to Vertex AI...")
        try:
            job_name = submit_automl_job(
                project=gcp.project_id,
                location=gcp.location,
                dataset_path=dataset_path,
                modality=plan.modality,
                target_column=target,
                staging_bucket=gcp.staging_bucket,
                finetune_model=finetune_model,
            )
        except Exception as e:
            click.echo(f"Error submitting job: {e}", err=True)
            sys.exit(1)

        history.add(
            TrainingRecord(
                fingerprint=fingerprint,
                dataset_label=dataset_label,
                dataset_path=dataset_path,
                modality=plan.modality,
                method=plan.method.value,
                mode="cloud",
                target_column=target or "",
                base_model=(
                    finetune_model if plan.method.value == "gemini_finetune" else ""
                ),
                purpose=purpose or "",
                job_name=job_name,
                status="running",
                project_id=gcp.project_id,
                location=gcp.location,
            )
        )

    resumed = action == "resume_running"

    for line in format_post_submit_guidance(plan, job_name, resumed=resumed):
        click.echo(line)

    try:
        wait_strategy = resolve_wait_strategy(
            detach=detach,
            poll_every=poll_every,
            interactive=sys.stdin.isatty(),
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if wait_strategy == "detach":
        click.echo(format_detach_message(plan))
        return

    interval_min = poll_every or DEFAULT_SCHEDULED_POLL_MIN
    click.echo(
        f"  Scheduled monitoring: checking every {interval_min} minutes "
        f"(up to {min(timeout, train_cfg.training_timeout_min)} min)."
    )

    timeout_min = min(timeout, train_cfg.training_timeout_min)
    deadline = time.time() + timeout_min * 60

    try:
        result = poll_job_scheduled(
            job_name,
            interval_min=interval_min,
            deadline=deadline,
        )
    except TimeoutError:
        history.update(fingerprint, status="timeout")
        click.echo(
            f"  Training still running after {timeout_min} min. "
            f"Check status in Google Cloud Console or re-run the same command to resume polling.",
            err=True,
        )
        sys.exit(1)
    except Exception as e:
        history.update(
            fingerprint,
            status="failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=str(e),
        )
        click.echo(f"  Training failed: {e}", err=True)
        sys.exit(1)

    history.update(
        fingerprint,
        status="succeeded",
        model_path=result.get("model_path", ""),
        completed_at=datetime.now(timezone.utc).isoformat(),
        error="",
        purpose=purpose or "",
    )
    click.echo(f"  Model saved to: {result.get('model_path', 'unknown')}")
    click.echo(f"  Evaluation accuracy: {result.get('accuracy', '?')}")

    from edge_train.agent import AgentState
    from edge_train.agent.context import sync_agent_context

    sync_agent_context(AgentState.load())

    from edge_train.deployments import format_phoenix_monitoring_hint

    model_path = result.get("model_path", "")
    click.echo(format_phoenix_monitoring_hint())
    if model_path.startswith("projects/"):
        click.echo(
            f"  Vertex deploy: coralflow deploy --cloud -m {model_path} --modality {plan.modality}"
        )
        click.echo(
            f"  After deploy: coralflow deploy --cloud -m {model_path} --modality {plan.modality} --simulate"
        )


def _train_local(
    dataset_label,
    dataset_path,
    modality,
    target,
    output_dir,
    epochs,
    force=False,
    purpose="",
):
    """Local training path — no cloud required."""
    from edge_train.training_history import (
        TrainingHistory,
        TrainingRecord,
        format_duplicate_message,
        make_training_fingerprint,
    )

    history = TrainingHistory.load()
    fingerprint = make_training_fingerprint(
        mode="local",
        dataset_path=dataset_path,
        modality=modality,
        method="local_keras",
        target_column=target or "",
        output_dir=output_dir,
        epochs=epochs,
    )
    action, existing = history.check_duplicate(fingerprint, force=force)
    if action == "skip_succeeded":
        click.echo(format_duplicate_message(action, existing))
        return

    click.echo(f"  Modality: {modality}")
    click.echo(f"  Dataset: {dataset_label}")
    if purpose:
        click.echo(f"  Project: {purpose}")
    if dataset_label != dataset_path:
        click.echo(f"  Resolved CSV: {dataset_path}")
    click.echo("  Training locally...")

    if modality == "text":
        from edge_train.trainer import train_text_classifier

        try:
            model_path = train_text_classifier(
                dataset_path=dataset_path,
                target_column=target,
                output_dir=output_dir,
                epochs=epochs,
            )
        except Exception as e:
            click.echo(f"  Training failed: {e}", err=True)
            history.update(
                fingerprint,
                status="failed",
                error=str(e),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            sys.exit(1)

        history.add(
            TrainingRecord(
                fingerprint=fingerprint,
                dataset_label=dataset_label,
                dataset_path=dataset_path,
                modality=modality,
                method="local_keras",
                mode="local",
                target_column=target or "",
                output_dir=output_dir,
                epochs=epochs,
                purpose=purpose or "",
                status="succeeded",
                model_path=model_path,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        )

        click.echo(f"  Model saved to: {model_path}")
        click.echo(
            f"  Next: coralflow validate --model {model_path} --output model.tflite"
        )
        click.echo(f"  Then: coralflow simulate --model {model_path}")

        from edge_train.agent import AgentState
        from edge_train.agent.context import sync_agent_context

        sync_agent_context(AgentState.load())
    else:
        click.echo(
            f"  Local training for '{modality}' is not yet supported. "
            f"Use --cloud for Vertex AI AutoML.",
            err=True,
        )
        sys.exit(1)
