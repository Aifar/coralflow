"""Configuration for GCP, Arize, and edge device credentials."""

import os
from dataclasses import dataclass, field
from typing import Optional


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
    api_key: str = field(default_factory=lambda: os.environ.get("ARIZE_API_KEY", ""))
    space_key: str = field(
        default_factory=lambda: os.environ.get("ARIZE_SPACE_KEY", "")
    )
    endpoint: str = field(
        default_factory=lambda: os.environ.get(
            "ARIZE_ENDPOINT", "https://api.arize.com"
        )
    )

    def is_valid(self) -> bool:
        return bool(self.api_key and self.space_key)


@dataclass
class TrainConfig:
    model_size_mb: float = 10.0
    inference_ms: int = 50
    accuracy_loss_pct: float = 2.0
    training_timeout_min: int = 30


from edge_train.edge.config import EdgeConfig


def load_config() -> tuple[GCPConfig, ArizeConfig, TrainConfig, EdgeConfig]:
    return GCPConfig(), ArizeConfig(), TrainConfig(), EdgeConfig()
