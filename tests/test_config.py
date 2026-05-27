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
            api_key="phx_xxx",
            collector_endpoint="https://app.phoenix.arize.com/v1/traces",
            project_name="my-project",
        )
        assert cfg.is_valid()

    def test_invalid_when_missing_keys(self):
        cfg = ArizeConfig(api_key="", collector_endpoint="", project_name="")
        assert not cfg.is_valid()

    def test_defaults(self, clear_env):
        cfg = ArizeConfig()
        assert cfg.collector_endpoint == "http://localhost:6006/v1/traces"
        assert cfg.project_name == "edge-train"
        assert cfg.api_key == ""
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
        os.environ["PHOENIX_API_KEY"] = "test-key"
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://example.com/v1/traces"
        os.environ["PHOENIX_PROJECT_NAME"] = "my-edge-project"

        gcp, arize, _, _ = load_config()
        assert gcp.project_id == "test-project"
        assert gcp.location == "europe-west4"
        assert gcp.staging_bucket == "gs://my-bucket"
        assert arize.api_key == "test-key"
        assert arize.collector_endpoint == "https://example.com/v1/traces"
        assert arize.project_name == "my-edge-project"

    def test_staging_bucket_normalized(self, clear_env, monkeypatch):
        from edge_train import config as cfg

        monkeypatch.setenv("GCP_STAGING_BUCKET", "my-bucket")
        cfg._normalize_gcp_env()
        assert os.environ["GCP_STAGING_BUCKET"] == "gs://my-bucket"

    def test_relative_credentials_resolved(self, clear_env, monkeypatch, tmp_path):
        from edge_train import config as cfg

        key = tmp_path / "sa.json"
        key.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "key/sa.json")
        monkeypatch.setattr(cfg, "_PKG_ROOT", tmp_path)
        (tmp_path / "key").mkdir()
        key2 = tmp_path / "key" / "sa.json"
        key2.write_text("{}", encoding="utf-8")
        cfg._normalize_gcp_env()
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(key2.resolve())

    def test_ensure_gcp_credentials_missing_file(self, clear_env, monkeypatch):
        from edge_train.config import ensure_gcp_credentials

        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/no/such/key.json")
        ok, err = ensure_gcp_credentials()
        assert not ok
        assert "missing file" in err.lower()

    def test_gcs_bucket_name(self):
        from edge_train.config import gcs_bucket_name

        assert gcs_bucket_name("gs://coralflow") == "coralflow"
        assert gcs_bucket_name("coralflow") == "coralflow"
        assert gcs_bucket_name("gs://coralflow/prefix") == "coralflow"
