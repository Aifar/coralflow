"""Post-submit guidance and optional scheduled polling for cloud training."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from edge_train.cloud.job_status import get_cloud_job_status
from edge_train.cloud.router import CloudTrainingMethod, CloudTrainingPlan

DEFAULT_SCHEDULED_POLL_MIN = 10

# (min_minutes, max_minutes, suggested_first_check_minutes)
_ETA_BY_METHOD: dict[CloudTrainingMethod, tuple[int, int, int]] = {
    CloudTrainingMethod.GEMINI_FINETUNE: (10, 60, 20),
    CloudTrainingMethod.AUTOML_TABULAR: (15, 90, 30),
    CloudTrainingMethod.AUTOML_IMAGE: (30, 120, 45),
    CloudTrainingMethod.AUTOML_VIDEO: (60, 180, 60),
}


def training_eta_minutes(plan: CloudTrainingPlan) -> tuple[int, int, int]:
    """Return (min_min, max_min, suggested_check_min) for a cloud training plan."""
    return _ETA_BY_METHOD.get(plan.method, (15, 90, 30))


def format_post_submit_guidance(
    plan: CloudTrainingPlan,
    job_name: str,
    *,
    resumed: bool = False,
) -> list[str]:
    """Human-readable lines after a cloud job is submitted or resumed."""
    min_min, max_min, check_min = training_eta_minutes(plan)
    check_at = datetime.now(timezone.utc) + timedelta(minutes=check_min)
    check_local = check_at.astimezone().strftime("%Y-%m-%d %H:%M")

    lines = [
        "",
        "  Cloud training is running on Vertex AI.",
    ]
    if resumed:
        lines.append(f"  Monitoring job: {job_name}")
    else:
        lines.append(f"  Job submitted: {job_name}")

    lines.extend(
        [
            f"  Estimated duration: {min_min}–{max_min} minutes ({plan.label}).",
            f"  Suggested first status check: ~{check_min} minutes from now ({check_local}).",
            "  Re-run the same `coralflow train --cloud ...` command later to resume status checks.",
            "",
        ]
    )
    return lines


def format_detach_message(plan: CloudTrainingPlan) -> str:
    _, _, check_min = training_eta_minutes(plan)
    return (
        f"  Exiting without waiting. Check again in ~{check_min} minutes "
        f"with the same command, or open the job in Google Cloud Console."
    )


def prompt_wait_strategy(
    *,
    default_scheduled_interval_min: int = DEFAULT_SCHEDULED_POLL_MIN,
) -> str:
    """Ask the user how to monitor the job (TTY only). Returns scheduled|detach."""
    import click

    click.echo("  How do you want to monitor this job?")
    click.echo(
        f"    [1] Check every {default_scheduled_interval_min} minutes until complete or timeout"
    )
    click.echo("    [2] Exit now — check status later with the same command")
    choice = click.prompt("  Choice", default="1", show_default=True)

    if choice in ("2", "detach", "exit", "quit", "no"):
        return "detach"
    return "scheduled"


def resolve_wait_strategy(
    *,
    detach: bool,
    poll_every: int | None,
    interactive: bool | None = None,
) -> str:
    """Pick monitoring mode from CLI flags or interactive prompt."""
    if detach:
        return "detach"
    if poll_every is not None:
        if poll_every < 1:
            raise ValueError("--poll-every must be at least 1 minute")
        return "scheduled"
    if interactive if interactive is not None else sys.stdin.isatty():
        return prompt_wait_strategy(
            default_scheduled_interval_min=poll_every or DEFAULT_SCHEDULED_POLL_MIN
        )
    return "scheduled"


def poll_job_scheduled(
    job_name: str,
    *,
    interval_min: int = DEFAULT_SCHEDULED_POLL_MIN,
    deadline: float,
    status_fn: Callable[[str], dict[str, str]] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    now_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Poll a cloud job every ``interval_min`` minutes until done or deadline."""
    status_fn = status_fn or get_cloud_job_status
    sleep_fn = sleep_fn or time.sleep
    now_fn = now_fn or time.time
    interval_sec = interval_min * 60
    check_num = 0

    while now_fn() < deadline:
        check_num += 1
        remote = status_fn(job_name)
        status = remote.get("status", "unknown")

        if status == "succeeded":
            return {
                "job_name": job_name,
                "model_path": remote.get("model_path", ""),
                "accuracy": 0.0,
            }
        if status == "failed":
            raise RuntimeError(remote.get("error") or "Training job failed")

        stamp = datetime.now().strftime("%H:%M:%S")
        print(
            f"  [{stamp}] Check #{check_num}: still training ({status})...",
            flush=True,
        )

        remaining = deadline - now_fn()
        if remaining <= 0:
            break

        sleep_for = min(interval_sec, remaining)
        print(
            f"  Next check in {interval_min} min (Ctrl+C to exit and check later).",
            flush=True,
        )
        sleep_fn(sleep_for)

    raise TimeoutError(
        f"Training job did not complete within deadline "
        f"(last check every {interval_min} min)."
    )
