"""Persist model deployments (edge devices and Vertex AI endpoints)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEPLOYMENTS_DIR = Path.home() / ".coralflow"
DEFAULT_DEPLOYMENTS_FILE = DEPLOYMENTS_DIR / "deployments.json"


def deployments_file_path() -> Path:
    override = os.environ.get("CORALFLOW_DEPLOYMENTS_PATH", "").strip()
    return Path(override) if override else DEFAULT_DEPLOYMENTS_FILE


@dataclass
class DeploymentRecord:
    model_path: str
    target: str  # "edge" | "vertex"
    modality: str = "text"
    method: str = ""
    endpoint_name: str = ""
    device_id: str = ""
    deployed_at: str = ""
    phoenix_project: str = ""

    def __post_init__(self) -> None:
        if not self.deployed_at:
            self.deployed_at = datetime.now(timezone.utc).isoformat()


@dataclass
class DeploymentRegistry:
    records: list[DeploymentRecord] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> DeploymentRegistry:
        target = path or deployments_file_path()
        if not target.exists():
            return cls()
        data = json.loads(target.read_text(encoding="utf-8"))
        records = [
            DeploymentRecord(
                **{
                    k: v
                    for k, v in item.items()
                    if k in DeploymentRecord.__dataclass_fields__
                }
            )
            for item in data.get("records", [])
        ]
        return cls(records=records)

    def save(self, path: Path | None = None) -> None:
        target = path or deployments_file_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [asdict(r) for r in self.records]}
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, record: DeploymentRecord) -> None:
        self.records = [
            r
            for r in self.records
            if not (r.model_path == record.model_path and r.target == record.target)
        ]
        self.records.insert(0, record)
        self.save()

    def latest_vertex(self, model_path: str | None = None) -> DeploymentRecord | None:
        for record in self.records:
            if record.target != "vertex":
                continue
            if model_path and record.model_path != model_path:
                continue
            return record
        return None

    def find_by_endpoint(self, endpoint_name: str) -> DeploymentRecord | None:
        for record in self.records:
            if record.endpoint_name == endpoint_name:
                return record
        return None


def format_phoenix_monitoring_hint(
    *,
    endpoint: str = "",
    model_path: str = "",
    modality: str = "",
) -> str:
    lines = [
        "",
        "  Arize Phoenix monitoring:",
        "    1. Set PHOENIX_COLLECTOR_ENDPOINT (+ PHOENIX_API_KEY for cloud)",
        "    2. coralflow monitor --status",
    ]
    if endpoint:
        mod = modality.lower().strip()
        if mod == "table":
            lines.append(
                f"    3. coralflow predict --endpoint {endpoint} --modality table --csv rows.csv"
            )
        elif mod == "image":
            lines.append(
                f"    3. coralflow predict --endpoint {endpoint} --modality image --image path.jpg"
            )
        elif mod == "video":
            lines.append(
                f"    3. coralflow predict --endpoint {endpoint} --modality video --gcs-uri gs://..."
            )
        else:
            lines.append(f'    3. coralflow predict --endpoint {endpoint} --text "..."')
    elif model_path:
        lines.append(f'    3. coralflow predict --model {model_path} --text "..."')
    else:
        lines.append("    3. coralflow predict --model <path> | --endpoint <vertex>")
    lines.append("    4. coralflow monitor --dashboard")
    lines.append("  Predictions log to prediction_log.jsonl and emit OTEL spans.")
    return "\n".join(lines)
