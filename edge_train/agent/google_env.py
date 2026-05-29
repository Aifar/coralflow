"""Interactive Google Cloud and Phoenix env setup for the CoralFlow agent."""

from __future__ import annotations

import os
import shlex
import sys
from typing import Callable, Protocol

from edge_train.config import (
    _normalize_gcp_env,
    ensure_gcp_credentials,
    persist_env_values,
)

_gcp_skipped_at_startup = False
_phoenix_skipped_at_startup = False

PHOENIX_CLOUD_ENDPOINT = "https://app.phoenix.arize.com/v1/traces"
PHOENIX_LOCAL_ENDPOINT = "http://localhost:6006/v1/traces"
PHOENIX_DEFAULT_PROJECT = "edge-train"


class PromptFn(Protocol):
    def __call__(self, label: str, *, default: str = "") -> str: ...


GCP_ENV_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("GCP_PROJECT", "GCP project ID", ""),
    ("GCP_LOCATION", "GCP region", "us-central1"),
    ("GCP_STAGING_BUCKET", "GCS staging bucket (gs://...)", ""),
    ("GOOGLE_APPLICATION_CREDENTIALS", "Path to service account JSON", ""),
)

PHOENIX_CLOUD_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("PHOENIX_API_KEY", "Phoenix Cloud API key", ""),
    ("PHOENIX_COLLECTOR_ENDPOINT", "Phoenix collector URL", PHOENIX_CLOUD_ENDPOINT),
    ("PHOENIX_PROJECT_NAME", "Phoenix project name", PHOENIX_DEFAULT_PROJECT),
)


def missing_gcp_env_keys() -> list[str]:
    """Return GCP env var names that are missing or invalid."""
    missing: list[str] = []
    for key, _, _ in GCP_ENV_FIELDS:
        if key == "GOOGLE_APPLICATION_CREDENTIALS":
            ok, _ = ensure_gcp_credentials()
            if not ok:
                missing.append(key)
            continue
        if not os.environ.get(key, "").strip():
            missing.append(key)
    return missing


def gcp_env_ready() -> bool:
    return not missing_gcp_env_keys()


def phoenix_configured() -> bool:
    from edge_train.config import phoenix_explicitly_configured

    return phoenix_explicitly_configured()


def phoenix_env_ready() -> bool:
    return phoenix_configured()


def was_google_env_skipped_at_startup() -> bool:
    return _gcp_skipped_at_startup


def was_phoenix_skipped_at_startup() -> bool:
    return _phoenix_skipped_at_startup


# Backward-compatible aliases
missing_google_env_keys = missing_gcp_env_keys
google_env_ready = gcp_env_ready


def format_gcp_env_hint(missing: list[str] | None = None) -> str:
    keys = missing if missing is not None else missing_gcp_env_keys()
    lines = [
        "Google Cloud settings (only needed for cloud API calls):",
    ]
    for key, desc, default in GCP_ENV_FIELDS:
        current = os.environ.get(key, "").strip() or default or "(not set)"
        marker = " *" if key in keys else ""
        lines.append(f"  export {key}=...  # {desc}{marker}")
        if current != "(not set)":
            lines.append(f"    current: {current}")
    lines.append("")
    lines.append("Used by: `train --cloud`, `models`, `cost`, Vertex predict/deploy.")
    return "\n".join(lines)


def format_phoenix_env_hint() -> str:
    from edge_train.config import ArizeConfig

    arize = ArizeConfig()
    lines = [
        "Phoenix tracing settings (optional — for predict/monitor spans):",
        f"  PHOENIX_API_KEY — {'set' if arize.api_key else '(not set)'}",
        f"  PHOENIX_COLLECTOR_ENDPOINT — {arize.collector_endpoint or '(not set)'}",
        f"  PHOENIX_PROJECT_NAME — {arize.project_name or PHOENIX_DEFAULT_PROJECT}",
    ]
    return "\n".join(lines)


def format_google_env_hint(missing: list[str] | None = None) -> str:
    return format_gcp_env_hint(missing)


def _apply_env_value(key: str, value: str) -> None:
    value = value.strip()
    if not value:
        return
    if key == "GCP_STAGING_BUCKET" and not value.startswith("gs://"):
        value = f"gs://{value.lstrip('/')}"
    os.environ[key] = value
    _normalize_gcp_env()


def persist_gcp_env() -> None:
    updates = {
        key: os.environ[key]
        for key, _, _ in GCP_ENV_FIELDS
        if os.environ.get(key, "").strip()
    }
    if updates:
        persist_env_values(updates)


def persist_phoenix_env() -> None:
    keys = [key for key, _, _ in PHOENIX_CLOUD_FIELDS]
    updates = {key: os.environ[key] for key in keys if os.environ.get(key, "").strip()}
    if updates:
        persist_env_values(updates)


def _prompt_fields(
    prompt_fn: PromptFn,
    fields: tuple[tuple[str, str, str], ...],
    *,
    missing_only: bool,
    missing_keys: set[str],
) -> None:
    for key, desc, default in fields:
        if missing_only and key not in missing_keys:
            continue

        current = os.environ.get(key, "").strip() or default
        label = f"{key} ({desc})"
        if default:
            label += f" [default: {default}]"

        value = prompt_fn(label, default=current).strip()
        if value.lower() == "skip":
            return
        if value:
            _apply_env_value(key, value)
        elif default and not os.environ.get(key, "").strip():
            _apply_env_value(key, default)


def prompt_gcp_env_interactive(
    prompt_fn: PromptFn,
    *,
    missing_only: bool = True,
) -> None:
    """Prompt for each GCP setting."""
    missing = set(missing_gcp_env_keys())
    if missing_only and not missing:
        return
    _prompt_fields(
        prompt_fn, GCP_ENV_FIELDS, missing_only=missing_only, missing_keys=missing
    )
    persist_gcp_env()


def prompt_phoenix_cloud_interactive(prompt_fn: PromptFn) -> None:
    """Prompt for Phoenix Cloud (existing hosted service)."""
    _prompt_fields(
        prompt_fn,
        PHOENIX_CLOUD_FIELDS,
        missing_only=False,
        missing_keys=set(),
    )
    persist_phoenix_env()


def apply_phoenix_local_defaults() -> None:
    """Configure env for a locally running Phoenix instance."""
    persist_env_values(
        {
            "PHOENIX_COLLECTOR_ENDPOINT": PHOENIX_LOCAL_ENDPOINT,
            "PHOENIX_PROJECT_NAME": os.environ.get("PHOENIX_PROJECT_NAME", "").strip()
            or PHOENIX_DEFAULT_PROJECT,
        }
    )
    os.environ.pop("PHOENIX_API_KEY", None)


def _print_gcp_menu(echo: Callable[[str], None] | None) -> None:
    if not echo:
        return
    echo(
        "Google Cloud is optional — only required when calling Google Cloud APIs "
        "(e.g. `train --cloud`, Vertex deploy/predict, `models list`).\n"
        "Local training and edge deploy do not need these settings.\n"
        "\n"
        "  1 — Configure Google Cloud\n"
        "  2 — Skip\n"
    )


def _print_phoenix_menu(echo: Callable[[str], None] | None) -> None:
    if not echo:
        return
    echo(
        "Phoenix tracing is optional — used to record prediction/monitor spans.\n"
        "\n"
        "  1 — Configure existing Phoenix service (Cloud)\n"
        "  2 — Run Phoenix locally (`phoenix serve`)\n"
        "  3 — Skip\n"
    )


def _prompt_gcp_menu(prompt_fn: PromptFn, echo: Callable[[str], None] | None) -> bool:
    """Returns True if user chose to configure GCP."""
    _print_gcp_menu(echo)
    if gcp_env_ready() and echo:
        echo(format_gcp_env_hint([]))
    elif echo:
        echo(format_gcp_env_hint())

    choice = prompt_fn("Google Cloud choice [1/2]").strip().lower()
    return choice in ("1", "configure", "setup", "设置")


def _prompt_phoenix_menu(
    prompt_fn: PromptFn, echo: Callable[[str], None] | None
) -> str:
    """Returns 'cloud', 'local', or 'skip'."""
    _print_phoenix_menu(echo)
    if echo:
        echo(format_phoenix_env_hint())

    choice = prompt_fn("Phoenix choice [1/2/3]").strip().lower()
    if choice in ("1", "cloud", "existing", "service"):
        return "cloud"
    if choice in ("2", "local"):
        return "local"
    return "skip"


def ensure_google_env_at_startup(
    prompt_fn: PromptFn,
    *,
    echo: Callable[[str], None] | None = None,
    is_tty: bool | None = None,
) -> bool:
    """After LLM is ready, configure GCP then Phoenix. Returns True if GCP was skipped."""
    global _gcp_skipped_at_startup, _phoenix_skipped_at_startup

    if is_tty is None:
        is_tty = sys.stdout.isatty() and sys.stdin.isatty()

    if not is_tty:
        _gcp_skipped_at_startup = not gcp_env_ready()
        _phoenix_skipped_at_startup = not phoenix_env_ready()
        return _gcp_skipped_at_startup

    gcp_ready = gcp_env_ready()
    phoenix_ready = phoenix_env_ready()

    if gcp_ready and phoenix_ready:
        _gcp_skipped_at_startup = False
        _phoenix_skipped_at_startup = False
        return False

    # ── Phase 1: Google Cloud ───────────────────────────────────────
    if gcp_ready:
        _gcp_skipped_at_startup = False
    elif _prompt_gcp_menu(prompt_fn, echo):
        prompt_gcp_env_interactive(prompt_fn, missing_only=not gcp_env_ready())
        _gcp_skipped_at_startup = not gcp_env_ready()
    else:
        _gcp_skipped_at_startup = True

    # ── Phase 2: Phoenix (skip menu when already configured) ───────
    if phoenix_ready:
        _phoenix_skipped_at_startup = False
    else:
        phoenix_choice = _prompt_phoenix_menu(prompt_fn, echo)
        if phoenix_choice == "cloud":
            prompt_phoenix_cloud_interactive(prompt_fn)
            _phoenix_skipped_at_startup = not phoenix_env_ready()
        elif phoenix_choice == "local":
            apply_phoenix_local_defaults()
            if echo:
                echo(
                    "Local Phoenix defaults saved.\n"
                    "Start the server with: `pip install arize-phoenix && phoenix serve`\n"
                )
            _phoenix_skipped_at_startup = False
        else:
            _phoenix_skipped_at_startup = True

    return _gcp_skipped_at_startup


def require_google_env(
    prompt_fn: PromptFn | None = None,
    *,
    echo: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Ensure GCP env is configured before a cloud API operation."""
    global _gcp_skipped_at_startup

    if gcp_env_ready():
        _gcp_skipped_at_startup = False
        return True, ""

    hint = format_gcp_env_hint()

    if prompt_fn is None:
        return False, (
            "Google Cloud settings are not configured.\n\n"
            f"{hint}\n"
            "Set the variables above, then retry."
        )

    if echo:
        echo("Google Cloud settings are required for this operation.\n")
        echo(hint)

    if not _prompt_gcp_menu(prompt_fn, echo):
        return False, "Google Cloud setup skipped — operation cancelled."

    prompt_gcp_env_interactive(prompt_fn, missing_only=True)

    if gcp_env_ready():
        _gcp_skipped_at_startup = False
        return True, ""

    still_missing = missing_gcp_env_keys()
    return False, (
        "Google Cloud settings are still incomplete: "
        + ", ".join(still_missing)
        + "\n\n"
        + format_gcp_env_hint(still_missing)
    )


def _prompt_and_setup_phoenix(
    prompt_fn: PromptFn,
    echo: Callable[[str], None] | None,
) -> tuple[bool, str]:
    """Interactive Phoenix setup when not configured. Returns (use_phoenix, note)."""
    global _phoenix_skipped_at_startup

    if echo:
        echo("Phoenix tracing is not configured.\n")
        echo(format_phoenix_env_hint())

    choice = _prompt_phoenix_menu(prompt_fn, echo)
    if choice == "skip":
        _phoenix_skipped_at_startup = True
        return False, ""

    if choice == "cloud":
        prompt_phoenix_cloud_interactive(prompt_fn)
    else:
        apply_phoenix_local_defaults()
        if echo:
            echo(
                "Local Phoenix defaults saved.\n"
                "Start the server with: `pip install arize-phoenix && phoenix serve`\n"
            )

    from edge_train.config import ArizeConfig
    from edge_train.phoenix_util import ensure_phoenix_ready

    arize = ArizeConfig()
    if not arize.is_valid():
        _phoenix_skipped_at_startup = True
        return False, "Phoenix not configured — running without tracing."

    active, err = ensure_phoenix_ready(arize)
    if active:
        _phoenix_skipped_at_startup = False
        return True, ""
    return _prompt_phoenix_unreachable(prompt_fn, echo, err)


def _prompt_phoenix_unreachable(
    prompt_fn: PromptFn,
    echo: Callable[[str], None] | None,
    err: str,
) -> tuple[bool, str]:
    """Offer reconfigure or skip when Phoenix is configured but unreachable."""
    global _phoenix_skipped_at_startup

    if echo:
        echo(err)
        echo(
            "\n  1 — Reconfigure Phoenix\n"
            "  2 — Skip tracing (continue without uploading spans)\n"
        )

    choice = prompt_fn("Phoenix choice [1/2]", default="2").strip().lower()
    if choice in ("2", "skip", "s"):
        _phoenix_skipped_at_startup = True
        return False, ""

    return _prompt_and_setup_phoenix(prompt_fn, echo)


def require_phoenix_env(
    prompt_fn: PromptFn | None = None,
    *,
    echo: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Resolve Phoenix tracing before predict/monitor.

    Returns (use_phoenix, note). When use_phoenix is False, callers must not
    upload OTEL spans or Phoenix logs, but should continue the operation.
    """
    from edge_train.config import ArizeConfig
    from edge_train.phoenix_util import ensure_phoenix_ready

    if was_phoenix_skipped_at_startup():
        return False, ""

    if not phoenix_configured():
        if prompt_fn is None:
            return False, ""
        return _prompt_and_setup_phoenix(prompt_fn, echo)

    arize = ArizeConfig()
    active, err = ensure_phoenix_ready(arize)
    if active:
        return True, ""

    if prompt_fn is None:
        return False, err

    return _prompt_phoenix_unreachable(prompt_fn, echo, err)


def shell_command_needs_phoenix(cmd: str) -> bool:
    """True when a coralflow CLI subcommand may upload Phoenix spans."""
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    if not parts:
        return False

    sub = parts[0]
    flags = set(parts[1:])

    if sub == "predict":
        return True
    if sub == "monitor" and "--retrain" not in flags:
        return True
    return False


def shell_command_needs_google_env(cmd: str) -> bool:
    """True when a coralflow CLI subcommand needs GCP credentials."""
    try:
        parts = shlex.split(cmd)
    except ValueError:
        parts = cmd.split()
    if not parts:
        return False

    sub = parts[0]
    flags = set(parts[1:])

    if sub == "train" and ("--cloud" in flags):
        return True
    if sub == "models":
        return True
    if sub == "cost":
        return True
    if sub == "predict" and ("--endpoint" in flags):
        return True
    if sub == "deploy" and ("--cloud" in flags):
        return True
    return False


# Backward-compatible name used by prompt flow
prompt_google_env_interactive = prompt_gcp_env_interactive
persist_google_env = persist_gcp_env
