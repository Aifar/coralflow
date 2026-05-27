"""Tests for cloud training router."""

from pathlib import Path

import pytest


class TestCloudRouter:
    def test_text_routes_to_gemini_finetune(self):
        from edge_train.cloud.router import CloudTrainingMethod, plan_cloud_training

        plan = plan_cloud_training("urgent.csv", "text")
        assert plan.method == CloudTrainingMethod.GEMINI_FINETUNE
        assert plan.modality == "text"
        assert plan.status == "active"

    def test_table_routes_to_automl_tabular(self):
        from edge_train.cloud.router import CloudTrainingMethod, plan_cloud_training

        plan = plan_cloud_training("data.csv", "table")
        assert plan.method == CloudTrainingMethod.AUTOML_TABULAR
        assert "Structured tabular" in plan.reason

    def test_image_routes_to_automl_image(self):
        from edge_train.cloud.router import CloudTrainingMethod, plan_cloud_training

        plan = plan_cloud_training("photos/", "image")
        assert plan.method == CloudTrainingMethod.AUTOML_IMAGE

    def test_video_routes_to_automl_video(self):
        from edge_train.cloud.router import CloudTrainingMethod, plan_cloud_training

        plan = plan_cloud_training("clips/", "video")
        assert plan.method == CloudTrainingMethod.AUTOML_VIDEO

    def test_csv_text_column_inferred_as_finetune(self, tmp_path):
        from edge_train.cloud.router import CloudTrainingMethod, plan_cloud_training

        csv = tmp_path / "data.csv"
        csv.write_text("text,label\nhello,urgent\n", encoding="utf-8")
        plan = plan_cloud_training(str(csv))
        assert plan.method == CloudTrainingMethod.GEMINI_FINETUNE

    def test_csv_numeric_inferred_as_tabular(self, tmp_path):
        from edge_train.cloud.router import CloudTrainingMethod, plan_cloud_training

        csv = tmp_path / "data.csv"
        csv.write_text(
            "age,height,score,label\n25,180,95,A\n30,170,88,B\n",
            encoding="utf-8",
        )
        plan = plan_cloud_training(str(csv))
        assert plan.method == CloudTrainingMethod.AUTOML_TABULAR

    def test_video_directory_inference(self, tmp_path):
        from edge_train.cloud.router import CloudTrainingMethod, plan_cloud_training

        d = tmp_path / "videos"
        d.mkdir()
        (d / "a.mp4").write_bytes(b"fake")
        plan = plan_cloud_training(str(d))
        assert plan.method == CloudTrainingMethod.AUTOML_VIDEO

    def test_sound_not_supported(self):
        from edge_train.cloud.router import plan_cloud_training

        with pytest.raises(ValueError, match="audio"):
            plan_cloud_training("audio.wav", "sound")

    def test_cloud_modality_supported_text(self):
        from edge_train.cloud.router import cloud_modality_supported

        ok, _ = cloud_modality_supported("text")
        assert ok

    def test_cloud_modality_supported_sound(self):
        from edge_train.cloud.router import cloud_modality_supported

        ok, msg = cloud_modality_supported("sound")
        assert not ok
        assert "audio" in msg.lower()
