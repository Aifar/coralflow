"""Tests for edge_train.config."""

import os

from edge_train.config import GCPConfig, ArizeConfig, TrainConfig, load_config


class TestGCPConfig:
    def test_valid(self):
        cfg = GCPConfig(
            project_id="my-project",
            location="us-central1",
            staging_bucket="gs://bucket",
        )
        assert cfg.is_valid()

    def test_invalid_when_empty(self):
        cfg = GCPConfig(project_id="", location="")
        assert not cfg.is_valid()


class TestArizeConfig:
    def test_valid(self):
        cfg = ArizeConfig(
            api_key="ak_xxx", space_key="sk_xxx", endpoint="https://api.arize.com"
        )
        assert cfg.is_valid()

    def test_invalid_when_missing_keys(self):
        cfg = ArizeConfig(api_key="", space_key="", endpoint="")
        assert not cfg.is_valid()


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.model_size_mb == 10.0
        assert cfg.inference_ms == 50
        assert cfg.accuracy_loss_pct == 2.0
        assert cfg.training_timeout_min == 30


class TestLoadConfig:
    def test_load_config_returns_quad(self, clear_env):
        gcp, arize, train, edge = load_config()
        assert isinstance(gcp, GCPConfig)
        assert isinstance(arize, ArizeConfig)
        assert isinstance(train, TrainConfig)
        from edge_train.edge.config import EdgeConfig

        assert isinstance(edge, EdgeConfig)

    def test_load_config_from_env(self, clear_env):
        os.environ["GCP_PROJECT"] = "test-project"
        os.environ["GCP_LOCATION"] = "europe-west4"
        os.environ["GCP_STAGING_BUCKET"] = "gs://my-bucket"
        os.environ["ARIZE_API_KEY"] = "test-key"
        os.environ["ARIZE_SPACE_KEY"] = "test-space"

        gcp, arize, _, _ = load_config()
        assert gcp.project_id == "test-project"
        assert gcp.location == "europe-west4"
        assert gcp.staging_bucket == "gs://my-bucket"
        assert arize.api_key == "test-key"
        assert arize.space_key == "test-space"
