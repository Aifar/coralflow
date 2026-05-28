"""Shared Phoenix OTEL setup for local and Vertex inference."""

from __future__ import annotations

import os
import sys
from typing import Callable, NamedTuple

from edge_train.config import load_config


class PhoenixPrepareResult(NamedTuple):
    active: bool
    message: str
    abort: bool


def prepare_phoenix_for_inference(
    *,
    required: bool = False,
    interactive: bool | None = None,
    prompt_fn: Callable[[str], bool] | None = None,
    echo_fn: Callable[[str], None] | None = None,
) -> PhoenixPrepareResult:
    """Check Phoenix, optionally prompt to start local server, register OTEL.

    Returns PhoenixPrepareResult:
      - active=True when spans can be sent
      - abort=True only when the caller must stop (e.g. Phoenix not configured
        but required)
      - message set for warnings or instructions
    """
    from edge_train.phoenix_util import ensure_phoenix_ready_interactive

    if os.environ.get("CORALFLOW_PHOENIX_SKIP") == "1":
        return PhoenixPrepareResult(False, "", False)

    _, arize, _, _ = load_config()
    if not arize.is_valid():
        if required:
            return PhoenixPrepareResult(
                False,
                (
                    "Phoenix not configured. Set PHOENIX_COLLECTOR_ENDPOINT "
                    "(and PHOENIX_API_KEY for Phoenix Cloud)."
                ),
                True,
            )
        return PhoenixPrepareResult(False, "", False)

    if interactive is None:
        interactive = sys.stdin.isatty()

    active, err = ensure_phoenix_ready_interactive(
        arize,
        interactive=interactive,
        prompt_fn=prompt_fn,
        echo_fn=echo_fn,
    )
    if active:
        return PhoenixPrepareResult(True, "", False)

    if not err:
        return PhoenixPrepareResult(False, "", False)

    # Unreachable cloud collector — block when Phoenix is required.
    if required and not arize.collector_endpoint.startswith(
        ("http://localhost", "http://127.0.0.1")
    ):
        return PhoenixPrepareResult(False, err, True)

    # Local Phoenix declined or failed to start — warn and continue without spans.
    if "continue without" in err.lower() or "不会发送" in err:
        return PhoenixPrepareResult(False, err, False)

    if required:
        return PhoenixPrepareResult(False, err, True)
    return PhoenixPrepareResult(False, err, False)


def echo_phoenix_exit_error(err: str) -> None:
    import click

    click.echo(err, err=True)
    raise SystemExit(1)


def apply_phoenix_prepare(result: PhoenixPrepareResult) -> bool:
    """Handle prepare result for CLI commands. Returns phoenix_active."""
    import click

    if result.abort:
        echo_phoenix_exit_error(result.message)
    if result.message and not result.active:
        click.echo(f"Warning: {result.message}", err=True)
    return result.active
