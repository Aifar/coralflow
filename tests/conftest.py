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
def clear_env():
    """Remove edge-train env vars before each test."""
    for key in list(os.environ):
        if key.startswith("GCP_") or key.startswith("ARIZE_"):
            os.environ.pop(key, None)
    yield
