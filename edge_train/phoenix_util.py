"""Arize Phoenix connectivity — probe collector, prompt to start, register OTEL."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse


@dataclass
class PhoenixStatus:
    configured: bool
    reachable: bool
    is_local: bool
    collector_endpoint: str
    dashboard_url: str
    detail: str


def derive_dashboard_url(collector_endpoint: str) -> str:
    """Derive the Phoenix UI URL from the OTLP collector endpoint."""
    if "/v1/traces" in collector_endpoint:
        return collector_endpoint.rsplit("/v1/traces", 1)[0]
    parsed = urlparse(collector_endpoint)
    return f"{parsed.scheme}://{parsed.netloc}"


def is_local_collector(collector_endpoint: str) -> bool:
    host = (urlparse(collector_endpoint).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "0.0.0.0")


def _probe_urls(collector_endpoint: str) -> list[str]:
    """URLs to try when checking whether Phoenix is up."""
    parsed = urlparse(collector_endpoint)
    base = f"{parsed.scheme}://{parsed.netloc}"
    urls: list[str] = []

    if is_local_collector(collector_endpoint):
        urls.extend([f"{base}/healthz", f"{base}/"])
    else:
        urls.append(derive_dashboard_url(collector_endpoint).rstrip("/") + "/")
        urls.append(base.rstrip("/") + "/")

    seen: set[str] = set()
    ordered: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def _probe_url(url: str, timeout: float) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 500, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        # 401/404 still means something is listening
        if exc.code < 500:
            return True, f"HTTP {exc.code}"
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)


def check_phoenix_running(arize_config, timeout: float = 3.0) -> PhoenixStatus:
    """Return whether Phoenix collector/UI is reachable."""
    endpoint = arize_config.collector_endpoint or ""
    is_local = is_local_collector(endpoint)
    dashboard = (
        derive_dashboard_url(endpoint) if endpoint else "https://app.phoenix.arize.com"
    )

    if not arize_config.is_valid():
        return PhoenixStatus(
            configured=False,
            reachable=False,
            is_local=is_local,
            collector_endpoint=endpoint,
            dashboard_url=dashboard,
            detail="not configured",
        )

    last_err = ""
    for url in _probe_urls(endpoint):
        ok, detail = _probe_url(url, timeout)
        if ok:
            return PhoenixStatus(
                configured=True,
                reachable=True,
                is_local=is_local,
                collector_endpoint=endpoint,
                dashboard_url=dashboard,
                detail=detail,
            )
        last_err = detail

    return PhoenixStatus(
        configured=True,
        reachable=False,
        is_local=is_local,
        collector_endpoint=endpoint,
        dashboard_url=dashboard,
        detail=last_err or "unreachable",
    )


def format_phoenix_start_instructions(status: PhoenixStatus) -> str:
    """User-facing instructions when Phoenix is configured but not reachable."""
    lines = [
        "## Phoenix is not running",
        "",
        f"Collector: `{status.collector_endpoint}`",
        f"Probe: {status.detail}",
        "",
    ]

    if status.is_local:
        lines.extend(
            [
                "Start **local Phoenix** before predict/monitor:",
                "```bash",
                "pip install arize-phoenix",
                "phoenix serve",
                "```",
                "",
                "Ensure environment:",
                "```bash",
                "export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces",
                "export PHOENIX_PROJECT_NAME=edge-train",
                "```",
                "",
                "Verify: `curl -fsS http://localhost:6006/healthz` → `OK`",
                "",
                "Then re-run your command.",
            ]
        )
    else:
        lines.extend(
            [
                "Cannot reach **Arize Phoenix Cloud**. Check:",
                "- `PHOENIX_API_KEY` is set and valid",
                "- `PHOENIX_COLLECTOR_ENDPOINT` matches your Phoenix space",
                "- Network / VPN / firewall allows HTTPS to Arize",
                "",
                f"Dashboard: {status.dashboard_url}",
                "",
                "Example:",
                "```bash",
                "export PHOENIX_API_KEY=your-cloud-api-key",
                "export PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/v1/traces",
                "export PHOENIX_PROJECT_NAME=edge-train",
                "```",
                "",
                "After Phoenix Cloud is reachable, re-run your command.",
            ]
        )

    return "\n".join(lines)


def local_phoenix_port(collector_endpoint: str) -> int:
    parsed = urlparse(collector_endpoint)
    if parsed.port:
        return parsed.port
    return 6006


def wait_for_phoenix_running(
    arize_config,
    *,
    timeout: float = 90.0,
    interval: float = 2.0,
) -> bool:
    """Poll until Phoenix responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check_phoenix_running(arize_config).reachable:
            return True
        time.sleep(interval)
    return False


def start_local_phoenix_process(collector_endpoint: str) -> subprocess.Popen:
    """Start `phoenix serve` in the background for local collectors."""
    if not shutil.which("phoenix"):
        raise FileNotFoundError("phoenix")

    port = local_phoenix_port(collector_endpoint)
    return subprocess.Popen(
        ["phoenix", "serve", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def ensure_phoenix_ready_interactive(
    arize_config,
    *,
    interactive: bool = True,
    prompt_fn: Callable[[str], bool] | None = None,
    echo_fn: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Check Phoenix; when local and down, ask to start before registering OTEL."""
    if not arize_config.is_valid():
        return False, ""

    status = check_phoenix_running(arize_config)
    if status.reachable:
        return _register_phoenix(arize_config)

    instructions = format_phoenix_start_instructions(status)

    if status.is_local and interactive:
        if not prompt_fn:
            return False, instructions
        if echo_fn:
            echo_fn(instructions)

        ask = (
            "未检测到本地 Arize Phoenix 服务。是否现在启动 Phoenix？"
            "（将运行 phoenix serve）"
        )
        if prompt_fn(ask):
            if echo_fn:
                echo_fn(f"正在启动本地 Phoenix（{status.dashboard_url}）…")
            try:
                start_local_phoenix_process(status.collector_endpoint)
            except FileNotFoundError:
                return (
                    False,
                    instructions
                    + "\n\n未找到 `phoenix` 命令。请先运行：`pip install arize-phoenix`",
                )
            except Exception as exc:
                return False, f"{instructions}\n\n启动 Phoenix 失败：{exc}"

            if wait_for_phoenix_running(arize_config):
                return _register_phoenix(arize_config)

            return (
                False,
                "Phoenix 启动超时。请在另一终端运行 `phoenix serve` 后重试。"
                "本次将继续执行，但不会发送 OTEL span。",
            )

        return (
            False,
            "已跳过启动 Phoenix。本次将继续执行，但不会发送 OTEL span。",
        )

    return False, instructions


def _register_phoenix(arize_config) -> tuple[bool, str]:
    try:
        logging.getLogger(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter"
        ).setLevel(logging.CRITICAL)
        logging.getLogger("opentelemetry.trace").setLevel(logging.CRITICAL)

        from phoenix.otel import register

        register(
            endpoint=arize_config.collector_endpoint,
            project_name=arize_config.project_name,
            auto_instrument=False,
            verbose=False,
        )
        return True, ""
    except ImportError:
        return False, (
            "`arize-phoenix-otel` is not installed. Run: `pip install arize-phoenix-otel`"
        )
    except Exception as exc:
        return False, f"Phoenix registration failed: {exc}"


def ensure_phoenix_ready(arize_config) -> tuple[bool, str]:
    """If Phoenix is configured, require it to be up; register OTEL when ready.

    Returns:
        (active, error_message) — active is True when spans can be sent.
        When not configured, returns (False, "") and prediction may proceed without OTEL.
    """
    return ensure_phoenix_ready_interactive(
        arize_config,
        interactive=False,
    )
