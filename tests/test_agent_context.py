"""Tests for focused agent project context persistence."""

import json

from edge_train.agent import AgentState
from edge_train.agent.context import (
    format_project_context,
    sync_agent_context,
    update_agent_context,
)
from edge_train.training_history import TrainingHistory, TrainingRecord


class TestAgentContext:
    def test_update_and_format(self, tmp_path):
        state = AgentState()
        update_agent_context(
            state,
            training_purpose="neu_cls_defect_classifier_v3",
            dataset_path="gs://coralflow/neu-cls",
            modality="image",
            training_status="succeeded",
            model_path="projects/p/locations/us/models/1",
            deployment_status="deployed",
            endpoint_name="projects/p/locations/us/endpoints/9",
            data_collection="360 条预测日志，0 条已标注",
            last_step="train",
        )

        text = format_project_context(state)
        assert "neu_cls_defect_classifier_v3" in text
        assert "gs://coralflow/neu-cls" in text
        assert "succeeded" in text
        assert "endpoints/9" in text

    def test_sync_from_training_history(self, tmp_path, monkeypatch):
        history_file = tmp_path / "training_history.json"
        monkeypatch.setenv("CORALFLOW_TRAINING_HISTORY_PATH", str(history_file))

        history = TrainingHistory()
        history.add(
            TrainingRecord(
                fingerprint="abc123",
                dataset_label="neu-cls",
                dataset_path="gs://coralflow/neu-cls",
                modality="image",
                method="automl_image",
                mode="cloud",
                purpose="neu_cls_defect_classifier_v3",
                status="succeeded",
                model_path="projects/p/locations/us/models/1",
            )
        )

        state = sync_agent_context(AgentState())
        assert state.training_purpose == "neu_cls_defect_classifier_v3"
        assert state.dataset_path == "gs://coralflow/neu-cls"
        assert state.training_status == "succeeded"
        assert state.model_path.endswith("/models/1")

    def test_prediction_log_stats(self, tmp_path, monkeypatch):
        log_path = tmp_path / "prediction_log.jsonl"
        log_path.write_text(
            json.dumps({"prediction": "a", "ground_truth": "a"}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("EDGE_PREDICTION_LOG_PATH", str(log_path))

        state = sync_agent_context(AgentState())
        assert "1 条预测日志" in state.data_collection
