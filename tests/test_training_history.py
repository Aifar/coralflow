"""Tests for training history persistence and duplicate detection."""

import json
from pathlib import Path

from edge_train.training_history import (
    TrainingHistory,
    TrainingRecord,
    format_duplicate_message,
    make_training_fingerprint,
)


class TestTrainingFingerprint:
    def test_same_config_same_fingerprint(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        fp1 = make_training_fingerprint(
            mode="cloud",
            dataset_path=str(csv),
            modality="table",
            method="automl_tabular",
            target_column="Churn",
        )
        fp2 = make_training_fingerprint(
            mode="cloud",
            dataset_path=str(csv),
            modality="table",
            method="automl_tabular",
            target_column="Churn",
        )
        assert fp1 == fp2

    def test_different_target_different_fingerprint(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n", encoding="utf-8")
        fp1 = make_training_fingerprint(
            mode="cloud",
            dataset_path=str(csv),
            modality="table",
            method="automl_tabular",
            target_column="Churn",
        )
        fp2 = make_training_fingerprint(
            mode="cloud",
            dataset_path=str(csv),
            modality="table",
            method="automl_tabular",
            target_column="label",
        )
        assert fp1 != fp2


class TestTrainingHistory:
    def test_save_and_load(self, tmp_path, monkeypatch):
        history_path = tmp_path / "training_history.json"
        monkeypatch.setenv("CORALFLOW_TRAINING_HISTORY_PATH", str(history_path))

        history = TrainingHistory()
        history.add(
            TrainingRecord(
                fingerprint="abc123",
                dataset_label="telco_churn.csv",
                dataset_path="/tmp/telco_churn.csv",
                modality="table",
                method="automl_tabular",
                mode="cloud",
                target_column="Churn",
                job_name="projects/p/locations/us/trainingPipelines/1",
                status="running",
            )
        )

        loaded = TrainingHistory.load()
        assert len(loaded.records) == 1
        assert loaded.records[0].job_name.endswith("/trainingPipelines/1")

    def test_skip_succeeded_duplicate(self, tmp_path, monkeypatch):
        history_path = tmp_path / "training_history.json"
        monkeypatch.setenv("CORALFLOW_TRAINING_HISTORY_PATH", str(history_path))

        record = TrainingRecord(
            fingerprint="done1",
            dataset_label="urgent",
            dataset_path="/data/urgent.csv",
            modality="text",
            method="gemini_finetune",
            mode="cloud",
            status="succeeded",
            model_path="projects/p/tunedModels/1",
            completed_at="2026-01-01T00:00:00+00:00",
        )
        history = TrainingHistory(records=[record])
        history.save()

        action, existing = history.check_duplicate("done1")
        assert action == "skip_succeeded"
        assert existing.model_path.endswith("/tunedModels/1")
        assert "Duplicate training skipped" in format_duplicate_message(
            action, existing
        )

    def test_resume_running_duplicate(self, tmp_path, monkeypatch):
        history_path = tmp_path / "training_history.json"
        monkeypatch.setenv("CORALFLOW_TRAINING_HISTORY_PATH", str(history_path))

        record = TrainingRecord(
            fingerprint="run1",
            dataset_label="telco_churn.csv",
            dataset_path="/data/telco_churn.csv",
            modality="table",
            method="automl_tabular",
            mode="cloud",
            job_name="projects/p/locations/us/trainingPipelines/9",
            status="running",
        )
        history = TrainingHistory(records=[record])
        history.save()

        action, existing = history.check_duplicate("run1")
        assert action == "resume_running"
        assert "already in progress" in format_duplicate_message(action, existing)

    def test_force_bypasses_duplicate(self, tmp_path, monkeypatch):
        history_path = tmp_path / "training_history.json"
        monkeypatch.setenv("CORALFLOW_TRAINING_HISTORY_PATH", str(history_path))

        history = TrainingHistory(
            records=[
                TrainingRecord(
                    fingerprint="done1",
                    dataset_label="urgent",
                    dataset_path="/data/urgent.csv",
                    modality="text",
                    method="gemini_finetune",
                    mode="cloud",
                    status="succeeded",
                    model_path="projects/p/tunedModels/1",
                )
            ]
        )
        history.save()

        action, _ = history.check_duplicate("done1", force=True)
        assert action is None

    def test_sync_cloud_jobs_updates_status(self, tmp_path, monkeypatch, mocker):
        history_path = tmp_path / "training_history.json"
        monkeypatch.setenv("CORALFLOW_TRAINING_HISTORY_PATH", str(history_path))
        mocker.patch(
            "edge_train.config.ensure_gcp_credentials", return_value=(True, "")
        )
        mocker.patch(
            "edge_train.cloud.job_status.get_cloud_job_status",
            return_value={
                "status": "succeeded",
                "model_path": "projects/p/models/1",
                "error": "",
            },
        )

        history = TrainingHistory(
            records=[
                TrainingRecord(
                    fingerprint="run1",
                    dataset_label="telco",
                    dataset_path="/data/telco.csv",
                    modality="table",
                    method="automl_tabular",
                    mode="cloud",
                    job_name="projects/p/locations/us/trainingPipelines/9",
                    status="running",
                )
            ]
        )
        history.save()

        history.sync_cloud_jobs()
        assert history.records[0].status == "succeeded"
        assert history.records[0].model_path == "projects/p/models/1"
        saved = json.loads(history_path.read_text(encoding="utf-8"))
        assert saved["records"][0]["status"] == "succeeded"
