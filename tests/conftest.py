"""Shared fixtures for edge-train tests."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_csv(temp_dir):
    path = temp_dir / "test.csv"
    path.write_text("text,label\nhello,greeting\nworld,greeting\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_text_csv(temp_dir):
    path = temp_dir / "train.csv"
    path.write_text(
        "text,label\nhello world,greeting\nhow are you,question\n"
        "goodbye,farewell\nsee you later,farewell\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture(autouse=True)
def _isolated_training_history(monkeypatch, tmp_path):
    """Keep training history and agent state out of the user's home during tests."""
    path = tmp_path / "training_history.json"
    monkeypatch.setenv("CORALFLOW_TRAINING_HISTORY_PATH", str(path))
    dep = tmp_path / "deployments.json"
    monkeypatch.setenv("CORALFLOW_DEPLOYMENTS_PATH", str(dep))
    monkeypatch.setenv(
        "EDGE_PREDICTION_LOG_PATH", str(tmp_path / "prediction_log.jsonl")
    )
    monkeypatch.setattr("edge_train.agent.STATE_FILE", tmp_path / "agent_state.json")
    for key in (
        "PHOENIX_API_KEY",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "PHOENIX_PROJECT_NAME",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def isolated_training_history(temp_dir, monkeypatch):
    path = temp_dir / "training_history.json"
    monkeypatch.setenv("CORALFLOW_TRAINING_HISTORY_PATH", str(path))
    return path


@pytest.fixture
def clear_env():
    """Remove edge-train env vars before each test."""
    for key in list(os.environ):
        if key.startswith(("GCP_", "ARIZE_", "PHOENIX_", "EDGE_", "CORALFLOW_")):
            os.environ.pop(key, None)
    yield
