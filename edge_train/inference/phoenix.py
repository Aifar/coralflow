"""Shared Phoenix OTEL setup for local and Vertex inference."""

from __future__ import annotations

import os
import sys

from edge_train.config import load_config


def should_skip_phoenix() -> bool:
    """True when the user opted out of Phoenix tracing for this process."""
    return os.environ.get("CORALFLOW_SKIP_PHOENIX", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def prepare_phoenix_for_inference(*, required: bool = False) -> tuple[bool, str]:
    """Register Phoenix when configured. Returns (active, error_message)."""
    if should_skip_phoenix():
        return False, ""

    from edge_train.config import load_config, phoenix_explicitly_configured
    from edge_train.phoenix_util import ensure_phoenix_ready

    _, arize, _, _ = load_config()
    if not phoenix_explicitly_configured(arize):
        if required:
            return False, (
                "Phoenix not configured. Set PHOENIX_COLLECTOR_ENDPOINT "
                "(and PHOENIX_API_KEY for Phoenix Cloud)."
            )
        return False, ""

    active, err = ensure_phoenix_ready(arize)
    if not active:
        return False, err
    return True, ""


def echo_phoenix_exit_error(err: str) -> None:
    import click

    click.echo(err, err=True)
    raise SystemExit(1)
