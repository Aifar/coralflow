"""Configuration for GCP, Arize, and edge device credentials."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Auto-load .env from project root or current directory
_dotenv_path = Path(".env")
if not _dotenv_path.exists():
    _dotenv_path = Path(__file__).parent.parent / ".env"
if _dotenv_path.exists():
    from dotenv import load_dotenv

    load_dotenv(_dotenv_path)


@dataclass
class GCPConfig:
    project_id: str = field(default_factory=lambda: os.environ.get("GCP_PROJECT", ""))
    location: str = field(
        default_factory=lambda: os.environ.get("GCP_LOCATION", "us-central1")
    )
    staging_bucket: str = field(
        default_factory=lambda: os.environ.get("GCP_STAGING_BUCKET", "")
    )

    def is_valid(self) -> bool:
        return bool(self.project_id and self.staging_bucket)


@dataclass
class ArizeConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("PHOENIX_API_KEY", ""))
    collector_endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "https://app.phoenix.arize.com/v1/traces",
        )
    )
    project_name: str = field(
        default_factory=lambda: os.environ.get("PHOENIX_PROJECT_NAME", "edge-train")
    )

    def is_valid(self) -> bool:
        return bool(self.api_key and self.collector_endpoint)


@dataclass
class TrainConfig:
    model_size_mb: float = 10.0
    inference_ms: int = 50
    accuracy_loss_pct: float = 2.0
    training_timeout_min: int = 30
    output_dir: str = "./model_output"
    local_epochs: int = 10


from edge_train.edge.config import EdgeConfig


def load_config() -> tuple[GCPConfig, ArizeConfig, TrainConfig, EdgeConfig]:
    return GCPConfig(), ArizeConfig(), TrainConfig(), EdgeConfig()
