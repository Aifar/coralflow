"""Persist training runs and avoid duplicate jobs across sessions."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORY_DIR = Path.home() / ".coralflow"
DEFAULT_HISTORY_FILE = HISTORY_DIR / "training_history.json"

ACTIVE_STATUSES = frozenset({"running", "timeout"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})


def history_file_path() -> Path:
    override = os.environ.get("CORALFLOW_TRAINING_HISTORY_PATH", "").strip()
    return Path(override) if override else DEFAULT_HISTORY_FILE


@dataclass
class TrainingRecord:
    fingerprint: str
    dataset_label: str
    dataset_path: str
    modality: str
    method: str
    mode: str
    target_column: str = ""
    base_model: str = ""
    purpose: str = ""
    job_name: str = ""
    status: str = "running"
    model_path: str = ""
    output_dir: str = ""
    epochs: int | None = None
    project_id: str = ""
    location: str = ""
    submitted_at: str = ""
    completed_at: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        if self.model_path:
            self.model_path = str(self.model_path)
        if self.output_dir:
            self.output_dir = str(self.output_dir)
        if not self.submitted_at:
            self.submitted_at = _now_iso()


@dataclass
class TrainingHistory:
    records: list[TrainingRecord] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> TrainingHistory:
        target = path or history_file_path()
        if not target.exists():
            return cls()
        data = json.loads(target.read_text(encoding="utf-8"))
        records = [
            TrainingRecord(
                **{
                    k: v
                    for k, v in item.items()
                    if k in TrainingRecord.__dataclass_fields__
                }
            )
            for item in data.get("records", [])
        ]
        return cls(records=records)

    def save(self, path: Path | None = None) -> None:
        target = path or history_file_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [asdict(r) for r in self.records]}
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, record: TrainingRecord) -> None:
        self.records = [r for r in self.records if r.fingerprint != record.fingerprint]
        self.records.insert(0, record)
        self.save()

    def update(self, fingerprint: str, **fields: Any) -> TrainingRecord | None:
        for record in self.records:
            if record.fingerprint == fingerprint:
                for key, value in fields.items():
                    if key in TrainingRecord.__dataclass_fields__:
                        setattr(record, key, value)
                self.save()
                return record
        return None

    def find_by_fingerprint(self, fingerprint: str) -> TrainingRecord | None:
        for record in self.records:
            if record.fingerprint == fingerprint:
                return record
        return None

    def running_records(self) -> list[TrainingRecord]:
        return [r for r in self.records if r.status in ACTIVE_STATUSES]

    def sync_cloud_jobs(self) -> list[TrainingRecord]:
        """Refresh running cloud jobs from Vertex AI."""
        from edge_train.cloud.job_status import get_cloud_job_status
        from edge_train.config import ensure_gcp_credentials

        ok, _ = ensure_gcp_credentials()
        if not ok:
            return []

        changed: list[TrainingRecord] = []
        for record in self.running_records():
            if not record.job_name or record.mode != "cloud":
                continue
            try:
                remote = get_cloud_job_status(record.job_name)
            except Exception as exc:
                record.error = str(exc)
                changed.append(record)
                continue

            status = remote.get("status", "unknown")
            if status == "succeeded":
                record.status = "succeeded"
                record.model_path = remote.get("model_path") or record.model_path
                record.completed_at = _now_iso()
                record.error = ""
                changed.append(record)
            elif status == "failed":
                record.status = "failed"
                record.completed_at = _now_iso()
                record.error = remote.get("error") or "failed"
                changed.append(record)
            elif status == "running":
                record.status = "running"
                record.error = ""
                changed.append(record)

        if changed:
            self.save()
        return changed

    def check_duplicate(
        self,
        fingerprint: str,
        *,
        force: bool = False,
    ) -> tuple[str | None, TrainingRecord | None]:
        """Return action hint and existing record if training should not proceed."""
        if force:
            return None, None

        existing = self.find_by_fingerprint(fingerprint)
        if not existing:
            return None, None

        if existing.status == "succeeded":
            if existing.model_path or existing.output_dir:
                return "skip_succeeded", existing
            return None, None

        if existing.status in ACTIVE_STATUSES:
            return "resume_running", existing

        return None, None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_version(dataset_path: str) -> str:
    path = Path(dataset_path)
    if path.is_file():
        stat = path.stat()
        return f"{stat.st_size}:{int(stat.st_mtime)}"
    if path.is_dir():
        return f"dir:{path.resolve()}"
    return dataset_path


def make_training_fingerprint(
    *,
    mode: str,
    dataset_path: str,
    modality: str,
    method: str,
    target_column: str = "",
    base_model: str = "",
    output_dir: str = "",
    epochs: int | None = None,
) -> str:
    payload = {
        "mode": mode,
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_version": _dataset_version(dataset_path),
        "modality": modality,
        "method": method,
        "target_column": target_column or "",
        "base_model": base_model or "",
        "output_dir": str(output_dir) if output_dir else "",
        "epochs": epochs or 0,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def format_startup_summary(history: TrainingHistory | None = None) -> str:
    history = history or TrainingHistory.load()
    lines: list[str] = []

    running = history.running_records()
    if running:
        lines.append("Training in progress:")
        for record in running[:5]:
            target = f", target={record.target_column}" if record.target_column else ""
            lines.append(
                f"  • {record.dataset_label} ({record.method}{target}) — "
                f"{record.status} since {record.submitted_at[:19]}"
            )
            if record.job_name:
                lines.append(f"    job: {record.job_name}")
        if len(running) > 5:
            lines.append(f"  … and {len(running) - 5} more")

    recent = [r for r in history.records if r.status == "succeeded"][:3]
    if recent:
        if lines:
            lines.append("")
        lines.append("Recent completed training:")
        for record in recent:
            label = record.purpose or record.dataset_label
            lines.append(
                f"  • {label} ({record.method}) → "
                f"{record.model_path or record.output_dir or 'saved'}"
            )

    return "\n".join(lines)


def format_duplicate_message(action: str, record: TrainingRecord) -> str:
    if action == "skip_succeeded":
        where = record.model_path or record.output_dir or "(unknown path)"
        return (
            f"Duplicate training skipped: same dataset/config already trained.\n"
            f"  Dataset: {record.dataset_label}\n"
            f"  Method:  {record.method}\n"
            f"  Model:   {where}\n"
            f"  Completed: {record.completed_at or record.submitted_at}\n"
            f"Use --force to train again."
        )
    if action == "resume_running":
        return (
            f"Training already in progress for this dataset/config.\n"
            f"  Dataset: {record.dataset_label}\n"
            f"  Method:  {record.method}\n"
            f"  Job:     {record.job_name or '(local)'}\n"
            f"  Status:  {record.status} since {record.submitted_at[:19]}\n"
            f"Resuming poll instead of submitting a new job."
        )
    return ""
