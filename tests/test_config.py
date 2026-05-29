"""Tests for edge_train.config."""

import os
from unittest.mock import patch

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
        assert cfg.is_valid()

    def test_local_without_api_key(self):
        cfg = ArizeConfig(
            api_key="",
            collector_endpoint="http://localhost:6006/v1/traces",
        )
        assert cfg.is_valid()

    def test_cloud_requires_api_key(self):
        cfg = ArizeConfig(
            api_key="",
            collector_endpoint="https://app.phoenix.arize.com/v1/traces",
        )
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


class TestEnvFile:
    def test_env_file_path_always_repo_root(self, monkeypatch, tmp_path):
        from edge_train import config as cfg

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".env").write_text(
            "CORALFLOW_LLM_API_KEY=from-repo\n", encoding="utf-8"
        )
        sub = repo / "sub"
        sub.mkdir()
        (sub / ".env").write_text("CORALFLOW_LLM_API_KEY=from-cwd\n", encoding="utf-8")
        monkeypatch.setattr(cfg, "_PKG_ROOT", repo)
        monkeypatch.chdir(sub)

        assert cfg.env_file_path() == repo / ".env"

    def test_persist_llm_from_subdir_writes_repo_env(
        self, clear_env, monkeypatch, tmp_path
    ):
        from edge_train import config as cfg
        from edge_train.agent.llm import LLMConfig, persist_llm_config

        repo = tmp_path / "repo"
        repo.mkdir()
        sub = repo / "sub"
        sub.mkdir()
        (sub / ".env").write_text("CORALFLOW_LLM_API_KEY=from-cwd\n", encoding="utf-8")
        monkeypatch.setattr(cfg, "_PKG_ROOT", repo)
        monkeypatch.chdir(sub)

        persist_llm_config(
            LLMConfig(
                api_key="sk-persisted",
                endpoint="https://api.example/v1",
                model="gpt-4o-mini",
            )
        )

        repo_text = (repo / ".env").read_text(encoding="utf-8")
        cwd_text = (sub / ".env").read_text(encoding="utf-8")
        assert "CORALFLOW_LLM_API_KEY=sk-persisted" in repo_text
        assert "CORALFLOW_LLM_API_KEY=from-cwd" in cwd_text

    def test_update_env_file_replaces_existing_key(self, tmp_path):
        from edge_train.config import update_env_file

        env_path = tmp_path / ".env"
        env_path.write_text(
            "GCP_PROJECT=old-project\nPHOENIX_PROJECT_NAME=edge-train\n",
            encoding="utf-8",
        )
        update_env_file({"GCP_PROJECT": "new-project"}, path=env_path)
        text = env_path.read_text(encoding="utf-8")
        assert "GCP_PROJECT=new-project" in text
        assert "GCP_PROJECT=old-project" not in text
        assert text.count("GCP_PROJECT=") == 1

    def test_update_env_file_removes_duplicate_keys(self, tmp_path):
        from edge_train.config import update_env_file

        env_path = tmp_path / ".env"
        env_path.write_text(
            "GCP_PROJECT=first\nGCP_PROJECT=duplicate\n",
            encoding="utf-8",
        )
        update_env_file({"GCP_PROJECT": "merged"}, path=env_path)
        text = env_path.read_text(encoding="utf-8")
        assert text.count("GCP_PROJECT=") == 1
        assert "GCP_PROJECT=merged" in text

    def test_update_env_file_appends_new_key(self, tmp_path):
        from edge_train.config import update_env_file

        env_path = tmp_path / ".env"
        env_path.write_text("# comment\n", encoding="utf-8")
        update_env_file({"CORALFLOW_LLM_API_KEY": "sk-test"}, path=env_path)
        text = env_path.read_text(encoding="utf-8")
        assert "# comment" in text
        assert "CORALFLOW_LLM_API_KEY=sk-test" in text

    def test_persist_env_values_writes_and_sets_environ(
        self, clear_env, tmp_path, monkeypatch
    ):
        from edge_train import config as cfg
        from edge_train.config import persist_env_values

        env_path = tmp_path / ".env"
        monkeypatch.setattr(cfg, "env_file_path", lambda: env_path)
        persist_env_values({"GCP_PROJECT": "proj", "GCP_LOCATION": "us-central1"})
        assert os.environ["GCP_PROJECT"] == "proj"
        text = env_path.read_text(encoding="utf-8")
        assert "GCP_PROJECT=proj" in text
        assert "GCP_LOCATION=us-central1" in text


class TestGoogleEnv:
    def test_missing_gcp_keys_when_empty(self, clear_env, monkeypatch):
        from edge_train.agent.google_env import missing_gcp_env_keys

        monkeypatch.setattr(
            "edge_train.agent.google_env.ensure_gcp_credentials",
            lambda: (False, "no creds"),
        )
        missing = missing_gcp_env_keys()
        assert "GCP_PROJECT" in missing
        assert "GCP_STAGING_BUCKET" in missing
        assert "GCP_LOCATION" in missing
        assert "GOOGLE_APPLICATION_CREDENTIALS" in missing
        assert "PHOENIX_PROJECT_NAME" not in missing

    def test_gcp_ready_when_all_set(self, clear_env, monkeypatch, tmp_path):
        from edge_train.agent.google_env import gcp_env_ready

        key = tmp_path / "sa.json"
        key.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("GCP_PROJECT", "proj")
        monkeypatch.setenv("GCP_LOCATION", "us-central1")
        monkeypatch.setenv("GCP_STAGING_BUCKET", "gs://bucket")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
        monkeypatch.setattr(
            "edge_train.agent.google_env.ensure_gcp_credentials",
            lambda: (True, ""),
        )
        assert gcp_env_ready()

    def test_phoenix_ready_when_cloud_configured(self, clear_env, monkeypatch):
        from edge_train.agent.google_env import phoenix_env_ready

        monkeypatch.setenv("PHOENIX_API_KEY", "phx_test")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/v1/traces"
        )
        assert phoenix_env_ready()

    def test_apply_phoenix_local_defaults(self, clear_env, tmp_path, monkeypatch):
        from edge_train import config as cfg
        from edge_train.agent.google_env import (
            PHOENIX_LOCAL_ENDPOINT,
            apply_phoenix_local_defaults,
        )

        env_path = tmp_path / ".env"
        monkeypatch.setattr(cfg, "env_file_path", lambda: env_path)
        monkeypatch.setenv("PHOENIX_API_KEY", "phx_old")
        apply_phoenix_local_defaults()
        assert os.environ["PHOENIX_COLLECTOR_ENDPOINT"] == PHOENIX_LOCAL_ENDPOINT
        assert os.environ["PHOENIX_PROJECT_NAME"] == "edge-train"
        assert "PHOENIX_API_KEY" not in os.environ
        text = env_path.read_text(encoding="utf-8")
        assert f"PHOENIX_COLLECTOR_ENDPOINT={PHOENIX_LOCAL_ENDPOINT}" in text

    def test_startup_skips_when_gcp_and_phoenix_ready(
        self, clear_env, monkeypatch, tmp_path
    ):
        from edge_train.agent.google_env import (
            ensure_google_env_at_startup,
            was_google_env_skipped_at_startup,
            was_phoenix_skipped_at_startup,
        )

        monkeypatch.setenv("GCP_PROJECT", "my-project")
        monkeypatch.setenv("GCP_LOCATION", "us-central1")
        monkeypatch.setenv("GCP_STAGING_BUCKET", "gs://bucket")
        key = tmp_path / "sa.json"
        key.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
        monkeypatch.setattr(
            "edge_train.agent.google_env.ensure_gcp_credentials",
            lambda: (True, ""),
        )
        monkeypatch.setenv("PHOENIX_API_KEY", "phx_test")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/v1/traces"
        )

        def _fail_prompt(*args, **kwargs):
            raise AssertionError("should not prompt when env is already configured")

        ensure_google_env_at_startup(
            _fail_prompt,
            echo=lambda msg: None,
            is_tty=True,
        )
        assert not was_google_env_skipped_at_startup()
        assert not was_phoenix_skipped_at_startup()

    def test_startup_skips_gcp_only_when_gcp_ready(
        self, clear_env, monkeypatch, tmp_path
    ):
        from edge_train.agent.google_env import (
            PHOENIX_LOCAL_ENDPOINT,
            ensure_google_env_at_startup,
        )

        monkeypatch.setenv("GCP_PROJECT", "my-project")
        monkeypatch.setenv("GCP_LOCATION", "us-central1")
        monkeypatch.setenv("GCP_STAGING_BUCKET", "gs://bucket")
        key = tmp_path / "sa.json"
        key.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key))
        monkeypatch.setattr(
            "edge_train.agent.google_env.ensure_gcp_credentials",
            lambda: (True, ""),
        )

        answers = iter(["2"])
        ensure_google_env_at_startup(
            lambda label, *, default="": next(answers),
            echo=lambda msg: None,
            is_tty=True,
        )
        assert os.environ["PHOENIX_COLLECTOR_ENDPOINT"] == PHOENIX_LOCAL_ENDPOINT

    def test_startup_gcp_skip_then_phoenix_local(self, clear_env, monkeypatch):
        from edge_train.agent.google_env import (
            PHOENIX_LOCAL_ENDPOINT,
            ensure_google_env_at_startup,
        )

        answers = iter(["2", "2"])
        ensure_google_env_at_startup(
            lambda label, *, default="": next(answers),
            echo=lambda msg: None,
            is_tty=True,
        )
        assert os.environ["PHOENIX_COLLECTOR_ENDPOINT"] == PHOENIX_LOCAL_ENDPOINT

    def test_startup_gcp_configure_then_phoenix_skip(
        self, clear_env, monkeypatch, tmp_path
    ):
        from edge_train import config as cfg
        from edge_train.agent.google_env import ensure_google_env_at_startup

        env_path = tmp_path / ".env"
        monkeypatch.setattr(cfg, "env_file_path", lambda: env_path)
        monkeypatch.setattr(
            "edge_train.agent.google_env.ensure_gcp_credentials",
            lambda: (True, ""),
        )
        answers = iter(
            [
                "1",
                "my-project",
                "us-central1",
                "gs://bucket",
                str(tmp_path / "sa.json"),
                "3",
            ]
        )
        (tmp_path / "sa.json").write_text("{}", encoding="utf-8")
        ensure_google_env_at_startup(
            lambda label, *, default="": next(answers, default),
            echo=lambda msg: None,
            is_tty=True,
        )
        assert os.environ["GCP_PROJECT"] == "my-project"
        assert "GCP_PROJECT=my-project" in env_path.read_text(encoding="utf-8")

    def test_shell_command_needs_google_env(self):
        from edge_train.agent.google_env import shell_command_needs_google_env

        assert shell_command_needs_google_env("train --cloud -d data.csv")
        assert shell_command_needs_google_env("models list")
        assert not shell_command_needs_google_env("train -d data.csv")
        assert shell_command_needs_google_env(
            "predict --endpoint projects/p/l/e/1 --text hi"
        )

    def test_prompt_gcp_applies_values(self, clear_env, monkeypatch, tmp_path):
        from edge_train import config as cfg
        from edge_train.agent.google_env import (
            missing_gcp_env_keys,
            prompt_gcp_env_interactive,
        )

        env_path = tmp_path / ".env"
        monkeypatch.setattr(cfg, "env_file_path", lambda: env_path)
        monkeypatch.setattr(
            "edge_train.agent.google_env.ensure_gcp_credentials",
            lambda: (True, ""),
        )
        answers = iter(
            [
                "my-project",
                "europe-west4",
                "gs://my-bucket",
                "/tmp/key.json",
            ]
        )
        prompt_gcp_env_interactive(
            lambda label, *, default="": next(answers, default),
            missing_only=False,
        )
        assert os.environ["GCP_PROJECT"] == "my-project"
        assert os.environ["GCP_LOCATION"] == "europe-west4"
        assert os.environ["GCP_STAGING_BUCKET"] == "gs://my-bucket"
        assert missing_gcp_env_keys() == []
        text = env_path.read_text(encoding="utf-8")
        assert "GCP_PROJECT=my-project" in text
        assert text.count("GCP_PROJECT=") == 1

    def test_require_phoenix_skip_when_startup_skipped(self, clear_env, monkeypatch):
        from edge_train.agent import google_env as ge
        from edge_train.agent.google_env import require_phoenix_env

        monkeypatch.setattr(ge, "_phoenix_skipped_at_startup", True)
        use_phoenix, note = require_phoenix_env(None)
        assert not use_phoenix
        assert note == ""

    def test_require_phoenix_skip_without_config(self, clear_env):
        from edge_train.agent.google_env import require_phoenix_env

        use_phoenix, note = require_phoenix_env(None)
        assert not use_phoenix
        assert note == ""

    @patch("edge_train.phoenix_util.ensure_phoenix_ready")
    def test_require_phoenix_active_when_ready(
        self, mock_ensure, clear_env, monkeypatch
    ):
        from edge_train.agent.google_env import require_phoenix_env

        monkeypatch.setenv("PHOENIX_API_KEY", "phx_test")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/v1/traces"
        )
        mock_ensure.return_value = (True, "")
        use_phoenix, note = require_phoenix_env(None)
        assert use_phoenix
        assert note == ""

    @patch("edge_train.phoenix_util.ensure_phoenix_ready")
    def test_require_phoenix_user_skips_unreachable(
        self, mock_ensure, clear_env, monkeypatch
    ):
        from edge_train.agent import google_env as ge
        from edge_train.agent.google_env import require_phoenix_env

        monkeypatch.setenv("PHOENIX_API_KEY", "phx_test")
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com/v1/traces"
        )
        mock_ensure.return_value = (False, "Phoenix is not running")
        use_phoenix, note = require_phoenix_env(
            lambda label, *, default="": "2",
            echo=lambda msg: None,
        )
        assert not use_phoenix
        assert note == ""
        assert ge.was_phoenix_skipped_at_startup()

    def test_shell_command_needs_phoenix(self):
        from edge_train.agent.google_env import shell_command_needs_phoenix

        assert shell_command_needs_phoenix("predict --model ./m --text hi")
        assert shell_command_needs_phoenix("monitor --status")
        assert not shell_command_needs_phoenix("monitor --retrain")

    def test_persist_llm_config(self, clear_env, tmp_path, monkeypatch):
        from edge_train import config as cfg
        from edge_train.agent.llm import LLMConfig, persist_llm_config

        env_path = tmp_path / ".env"
        env_path.write_text("CORALFLOW_LLM_MODEL=old-model\n", encoding="utf-8")
        monkeypatch.setattr(cfg, "env_file_path", lambda: env_path)
        persist_llm_config(
            LLMConfig(
                api_key="sk-test",
                endpoint="https://api.example/v1",
                model="gpt-4o-mini",
            )
        )
        text = env_path.read_text(encoding="utf-8")
        assert "CORALFLOW_LLM_API_KEY=sk-test" in text
        assert "CORALFLOW_LLM_MODEL=gpt-4o-mini" in text
        assert text.count("CORALFLOW_LLM_MODEL=") == 1
