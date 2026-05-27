"""Tests for edge_train.cloud GCS upload and fine-tuning helpers."""

import json
from pathlib import Path


class TestUploadToGcs:
    def test_uses_configured_staging_bucket(self, mocker, tmp_path):
        csv = tmp_path / "urgent.csv"
        csv.write_text("text,label\nhello,urgent\n", encoding="utf-8")

        mock_client = mocker.MagicMock()
        mock_bucket = mocker.MagicMock()
        mock_blob = mocker.MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mocker.patch("google.cloud.storage.Client", return_value=mock_client)

        from edge_train.cloud import _upload_to_gcs

        uri = _upload_to_gcs(str(csv), "my-project", "gs://coralflow")
        mock_client.bucket.assert_called_once_with("coralflow")
        mock_blob.upload_from_filename.assert_called_once_with(str(csv))
        assert uri == "gs://coralflow/datasets/urgent.csv"

    def test_upload_with_custom_prefix(self, mocker, tmp_path):
        data = tmp_path / "train.jsonl"
        data.write_text("{}\n", encoding="utf-8")

        mock_client = mocker.MagicMock()
        mock_bucket = mocker.MagicMock()
        mock_blob = mocker.MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mocker.patch("google.cloud.storage.Client", return_value=mock_client)

        from edge_train.cloud import _upload_to_gcs

        uri = _upload_to_gcs(
            str(data), "my-project", "gs://coralflow", blob_prefix="finetune"
        )
        mock_bucket.blob.assert_called_once_with("finetune/train.jsonl")
        assert uri == "gs://coralflow/finetune/train.jsonl"

    def test_missing_staging_bucket_raises(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("text,label\na,b\n", encoding="utf-8")

        from edge_train.cloud import _upload_to_gcs

        try:
            _upload_to_gcs(str(csv), "my-project", "")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "GCP_STAGING_BUCKET" in str(exc)


class TestCloudModalitySupport:
    def test_text_supported_via_finetune(self):
        from edge_train.cloud import cloud_modality_supported

        ok, _ = cloud_modality_supported("text")
        assert ok

    def test_table_automl_supported(self):
        from edge_train.cloud import cloud_modality_supported

        ok, _ = cloud_modality_supported("table")
        assert ok

    def test_video_automl_supported(self):
        from edge_train.cloud import cloud_modality_supported

        ok, _ = cloud_modality_supported("video")
        assert ok


class TestFinetuneConversion:
    def test_convert_csv_to_jsonl(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text(
            "text,urgency\n" '"服务器挂了",紧急\n' '"今晚吃什么",一般\n',
            encoding="utf-8",
        )

        from edge_train.cloud.finetune import convert_csv_to_jsonl

        jsonl_path, classes = convert_csv_to_jsonl(str(csv))
        assert classes == ["一般", "紧急"]
        lines = Path(jsonl_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert record["contents"][0]["parts"][0]["text"] == "服务器挂了"
        assert record["contents"][1]["parts"][0]["text"] == "紧急"
        assert "紧急" in record["systemInstruction"]["parts"][0]["text"]

    def test_submit_finetune_job_uses_sft(self, mocker, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("text,label\nhello,urgent\n", encoding="utf-8")

        mocker.patch(
            "edge_train.cloud.finetune.convert_csv_to_jsonl",
            return_value=("/tmp/train.jsonl", ["urgent"]),
        )
        mocker.patch(
            "edge_train.cloud._upload_to_gcs",
            return_value="gs://coralflow/finetune/train.jsonl",
        )
        mock_train = mocker.patch("vertexai.tuning.sft.train")
        mock_job = mocker.MagicMock()
        mock_job.resource_name = "projects/p/locations/us-central1/tuningJobs/123"
        mock_train.return_value = mock_job
        mocker.patch("vertexai.init")
        mocker.patch(
            "edge_train.cloud.publisher_models.resolve_finetune_base_model",
            return_value="gemini-2.0-flash-001",
        )

        from edge_train.cloud.finetune import submit_finetune_job

        job_name = submit_finetune_job(
            "p",
            "us-central1",
            str(csv),
            "gs://coralflow",
            base_model="gemini-2.0-flash-001",
        )
        assert job_name.endswith("/tuningJobs/123")
        mock_train.assert_called_once()
        assert (
            mock_train.call_args.kwargs["train_dataset"]
            == "gs://coralflow/finetune/train.jsonl"
        )

    def test_is_finetune_job(self):
        from edge_train.cloud.finetune import is_finetune_job

        assert is_finetune_job("projects/p/locations/us/tuningJobs/1")
        assert not is_finetune_job("projects/p/locations/us/trainingPipelines/1")


class TestCreateDataset:
    def test_tabular_create_omits_import_schema_uri(self, mocker):
        mock_create = mocker.patch(
            "google.cloud.aiplatform.TabularDataset.create",
            return_value=mocker.MagicMock(),
        )
        from edge_train.cloud import Modality, _create_or_get_dataset

        _create_or_get_dataset(
            "gs://coralflow/datasets/churn.csv",
            Modality.TABLE,
            "my-project",
            "us-central1",
            "gs://coralflow",
        )
        kwargs = mock_create.call_args.kwargs
        assert "import_schema_uri" not in kwargs
        assert kwargs["gcs_source"] == "gs://coralflow/datasets/churn.csv"

    def test_image_create_includes_import_schema_uri(self, mocker):
        mock_create = mocker.patch(
            "google.cloud.aiplatform.ImageDataset.create",
            return_value=mocker.MagicMock(),
        )
        from edge_train.cloud import Modality, _create_or_get_dataset

        _create_or_get_dataset(
            "gs://coralflow/images/",
            Modality.IMAGE,
            "my-project",
            "us-central1",
            "gs://coralflow",
        )
        kwargs = mock_create.call_args.kwargs
        assert "import_schema_uri" in kwargs
        assert "image_classification_schema.yaml" in kwargs["import_schema_uri"]


class TestLaunchTrainingPipeline:
    def test_tabular_job_uses_optimization_prediction_type(self, mocker):
        mock_job = mocker.MagicMock()
        mock_job.resource_name = "projects/p/locations/us/trainingPipelines/1"
        mock_tabular_cls = mocker.patch(
            "google.cloud.aiplatform.AutoMLTabularTrainingJob",
            return_value=mock_job,
        )
        mock_dataset = mocker.MagicMock()
        from edge_train.cloud import Modality, _launch_training_pipeline

        name = _launch_training_pipeline(mock_dataset, Modality.TABLE, "Churn")
        mock_tabular_cls.assert_called_once_with(
            display_name=mocker.ANY,
            optimization_prediction_type="classification",
        )
        mock_job.run.assert_called_once_with(
            dataset=mock_dataset,
            target_column="Churn",
            sync=False,
        )
        assert name.endswith("/trainingPipelines/1")

    def test_image_job_uses_classification_prediction_type(self, mocker):
        mock_job = mocker.MagicMock()
        mock_job.resource_name = "projects/p/locations/us/trainingPipelines/2"
        mock_image_cls = mocker.patch(
            "google.cloud.aiplatform.AutoMLImageTrainingJob",
            return_value=mock_job,
        )
        mock_dataset = mocker.MagicMock()
        from edge_train.cloud import Modality, _launch_training_pipeline

        _launch_training_pipeline(mock_dataset, Modality.IMAGE, None)
        mock_image_cls.assert_called_once_with(
            display_name=mocker.ANY,
            prediction_type="classification",
        )
        mock_job.run.assert_called_once_with(dataset=mock_dataset, sync=False)
