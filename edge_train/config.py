"""Configuration for GCP, Arize, and edge device credentials."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_PKG_ROOT = Path(__file__).resolve().parent.parent


def env_file_path() -> Path:
    """Path to the .env file agent/CLI persists (always repo root).

    Loading still merges repo-root .env then cwd .env via _bootstrap_env().
    Writes always target the repo root so keys survive restarts from any cwd.
    """
    return _PKG_ROOT / ".env"


_ENV_LINE_RE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$"
)


def _format_env_value(value: str) -> str:
    if any(c in value for c in " \t#\"'\\"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _env_value_for_persist(key: str, value: str) -> str:
    """Prefer repo-relative credential paths when persisting to .env."""
    if key == "GOOGLE_APPLICATION_CREDENTIALS":
        p = Path(value.strip())
        if p.is_absolute():
            try:
                return str(p.relative_to(_PKG_ROOT))
            except ValueError:
                pass
    return value.strip()


def update_env_file(updates: dict[str, str], *, path: Path | None = None) -> Path:
    """Merge key=value pairs into .env, replacing existing keys without duplicates."""
    if not updates:
        return path or env_file_path()

    env_path = path or env_file_path()
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    seen_keys: set[str] = set()
    key_line_index: dict[str, int] = {}
    for idx, line in enumerate(lines):
        match = _ENV_LINE_RE.match(line)
        if not match:
            continue
        key = match.group("key")
        if key in seen_keys:
            lines[idx] = None  # type: ignore[assignment]
            continue
        seen_keys.add(key)
        key_line_index[key] = idx

    for key, raw_value in updates.items():
        value = _env_value_for_persist(key, raw_value)
        if not value:
            continue
        formatted = f"{key}={_format_env_value(value)}"
        if key in key_line_index:
            lines[key_line_index[key]] = formatted
        else:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(formatted)
            key_line_index[key] = len(lines) - 1
            seen_keys.add(key)

    cleaned = [line for line in lines if line is not None]
    text = "\n".join(cleaned)
    if text and not text.endswith("\n"):
        text += "\n"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(text, encoding="utf-8")
    return env_path


def persist_env_values(updates: dict[str, str]) -> Path | None:
    """Write non-empty env vars to .env and mirror them in os.environ."""
    filtered = {k: v.strip() for k, v in updates.items() if v and str(v).strip()}
    if not filtered:
        return None
    for key, value in filtered.items():
        os.environ[key] = value
    _normalize_gcp_env()
    return update_env_file(filtered)


def _bootstrap_env() -> None:
    """Load .env from package root, then optional cwd .env (local overrides)."""
    from dotenv import load_dotenv

    project_env = _PKG_ROOT / ".env"
    if project_env.exists():
        load_dotenv(project_env)

    cwd_env = Path(".env")
    if cwd_env.exists() and cwd_env.resolve() != project_env.resolve():
        load_dotenv(cwd_env, override=True)

    _normalize_gcp_env()


def _normalize_gcp_env() -> None:
    """Resolve credential paths and normalize GCS bucket URIs."""
    creds = (
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        .strip()
        .strip('"')
        .strip("'")
    )
    if creds:
        cred_path = Path(creds)
        if not cred_path.is_absolute():
            cred_path = (_PKG_ROOT / cred_path).resolve()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)

    bucket = os.environ.get("GCP_STAGING_BUCKET", "").strip()
    if bucket and not bucket.startswith("gs://"):
        os.environ["GCP_STAGING_BUCKET"] = f"gs://{bucket.lstrip('/')}"


def ensure_gcp_credentials() -> tuple[bool, str]:
    """Return (ok, error_message) before Vertex AI / GCS calls."""
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if creds:
        if not Path(creds).is_file():
            return False, (
                f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file:\n  {creds}"
            )
        return True, ""

    try:
        import google.auth

        google.auth.default()
        return True, ""
    except Exception as exc:
        env_hint = _PKG_ROOT / ".env"
        return False, (
            "GCP Application Default Credentials not found.\n"
            f"Add to {env_hint}:\n"
            "  GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json\n"
            f"Or run: gcloud auth application-default login\n"
            f"Detail: {exc}"
        )


def gcs_bucket_name(staging_bucket: str) -> str:
    """Extract bucket name from gs://bucket or gs://bucket/prefix."""
    bucket = staging_bucket.strip()
    if bucket.startswith("gs://"):
        bucket = bucket[5:]
    return bucket.split("/")[0].strip()


_bootstrap_env()


@dataclass
class GCPConfig:
    project_id: str = field(default_factory=lambda: os.environ.get("GCP_PROJECT", ""))
    location: str = field(
        default_factory=lambda: os.environ.get("GCP_LOCATION", "us-central1")
    )
    staging_bucket: str = field(
        default_factory=lambda: os.environ.get("GCP_STAGING_BUCKET", "")
    )
    finetune_model: str = field(
        default_factory=lambda: os.environ.get(
            "GCP_FINETUNE_MODEL", "gemini-2.0-flash-001"
        )
    )

    def is_valid(self) -> bool:
        return bool(self.project_id and self.staging_bucket)


@dataclass
class ArizeConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("PHOENIX_API_KEY", ""))
    collector_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://localhost:6006/v1/traces",
        )
    )
    project_name: str = field(
        default_factory=lambda: os.environ.get("PHOENIX_PROJECT_NAME", "edge-train")
    )

    def is_valid(self) -> bool:
        endpoint = self.collector_endpoint.strip()
        if not endpoint:
            return False
        from edge_train.phoenix_util import is_local_collector

        if is_local_collector(endpoint):
            return True
        return bool(self.api_key)


def phoenix_explicitly_configured(arize: ArizeConfig | None = None) -> bool:
    """True when Phoenix env vars were set (not only dataclass defaults)."""
    if not os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip():
        if not os.environ.get("PHOENIX_API_KEY", "").strip():
            return False
    if arize is None:
        arize = ArizeConfig()
    return arize.is_valid()


@dataclass
class TrainConfig:
    model_size_mb: float = 10.0
    inference_ms: int = 50
    accuracy_loss_pct: float = 2.0
    training_timeout_min: int = 30
    output_dir: str = "./model_output"
    local_epochs: int = 10
    retrain_accuracy_threshold: float = 0.85
    retrain_min_samples: int = 10
    prediction_log_path: str = field(
        default_factory=lambda: os.environ.get(
            "EDGE_PREDICTION_LOG_PATH", "./prediction_log.jsonl"
        )
    )


@dataclass
class LLMConfig:
    """LLM API configuration for the coralflow agent."""

    endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "CORALFLOW_LLM_ENDPOINT", "https://api.openai.com/v1"
        )
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("CORALFLOW_LLM_API_KEY", "")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("CORALFLOW_LLM_MODEL", "gpt-4o")
    )

    def is_valid(self) -> bool:
        return bool(self.api_key)


from edge_train.edge.config import EdgeConfig


def load_config() -> tuple[GCPConfig, ArizeConfig, TrainConfig, EdgeConfig]:
    return GCPConfig(), ArizeConfig(), TrainConfig(), EdgeConfig()
