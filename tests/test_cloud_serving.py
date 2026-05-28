"""Tests for Vertex AI serving helpers."""

import base64
import json

import pytest

from edge_train.cloud.serving import (
    VertexImagePredictor,
    VertexTabularPredictor,
    VertexTextPredictor,
    VertexVideoPredictor,
    deploy_model_to_vertex,
    is_vertex_endpoint,
    is_vertex_resource,
    model_supports_dedicated_deployment,
    parse_automl_classification,
    resolve_vertex_predictor,
)


class TestVertexResourceDetection:
    def test_model_resource(self):
        name = "projects/p/locations/us-central1/models/123"
        assert is_vertex_resource(name)
        assert not is_vertex_endpoint(name)

    def test_endpoint_resource(self):
        name = "projects/p/locations/us-central1/endpoints/456"
        assert is_vertex_resource(name)
        assert is_vertex_endpoint(name)


class TestParseAutoMLClassification:
    def test_display_names_and_confidences(self):
        pred = {
            "displayNames": ["cat", "dog"],
            "confidences": [0.9, 0.1],
        }
        label, conf, probs = parse_automl_classification(pred)
        assert label == "cat"
        assert conf == pytest.approx(0.9)
        assert probs["dog"] == pytest.approx(0.1)

    def test_predicted_class(self):
        pred = {"predicted_class": "churn", "confidence": 0.88}
        label, conf, probs = parse_automl_classification(pred)
        assert label == "churn"
        assert conf == pytest.approx(0.88)

    def test_scalar_fallback(self):
        label, conf, probs = parse_automl_classification("positive")
        assert label == "positive"
        assert conf == 1.0


class TestDeployModelToVertex:
    def test_deploy_calls_aiplatform(self, mocker):
        mock_init = mocker.patch("google.cloud.aiplatform.init")
        mock_model_cls = mocker.patch("google.cloud.aiplatform.Model")
        endpoint = mocker.MagicMock()
        endpoint.resource_name = "projects/p/locations/us/endpoints/1"
        endpoint.list_models.return_value = [mocker.MagicMock(id="dm-1")]
        mock_model = mock_model_cls.return_value
        mock_model.display_name = "test-model"
        mock_model.gca_resource = mocker.MagicMock(
            supported_deployment_resources_types=["DEDICATED_RESOURCES"]
        )
        mock_model.deploy.return_value = endpoint
        mocker.patch(
            "edge_train.config.ensure_gcp_credentials", return_value=(True, "")
        )

        result = deploy_model_to_vertex(
            "projects/p/locations/us/models/1",
            project="p",
            location="us",
        )
        assert result.endpoint_name.endswith("/endpoints/1")
        mock_init.assert_called_once()
        mock_model.deploy.assert_called_once()
        assert "machine_type" in mock_model.deploy.call_args.kwargs

    def test_automl_deploy_omits_machine_type(self, mocker):
        mock_init = mocker.patch("google.cloud.aiplatform.init")
        mock_model_cls = mocker.patch("google.cloud.aiplatform.Model")
        endpoint = mocker.MagicMock()
        endpoint.resource_name = "projects/p/locations/us/endpoints/2"
        endpoint.list_models.return_value = [mocker.MagicMock(id="dm-2")]
        mock_model = mock_model_cls.return_value
        mock_model.display_name = "automl-image"
        mock_model.gca_resource = mocker.MagicMock(
            supported_deployment_resources_types=[]
        )
        mock_model.deploy.return_value = endpoint
        mocker.patch(
            "edge_train.config.ensure_gcp_credentials", return_value=(True, "")
        )

        deploy_model_to_vertex(
            "projects/p/locations/us/models/2",
            project="p",
            location="us",
        )
        kwargs = mock_model.deploy.call_args.kwargs
        assert "machine_type" not in kwargs
        assert kwargs.get("sync") is True


class TestVertexTextPredictor:
    def test_predict_parses_label(self, mocker):
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch("vertexai.init")
        mock_model_cls = mocker.patch("vertexai.generative_models.GenerativeModel")
        response = mocker.MagicMock()
        response.text = "urgent\n"
        mock_model_cls.return_value.generate_content.return_value = response
        mocker.patch(
            "edge_train.config.ensure_gcp_credentials", return_value=(True, "")
        )

        predictor = VertexTextPredictor(
            "projects/p/locations/us-central1/endpoints/1",
            project="p",
            location="us-central1",
        )
        label, conf = predictor.predict("need this today")
        assert label == "urgent"
        assert conf == 1.0

    def test_rejects_non_endpoint(self):
        with pytest.raises(ValueError, match="endpoint"):
            VertexTextPredictor(
                "projects/p/locations/us-central1/models/1",
                project="p",
                location="us-central1",
            )


class TestVertexAutoMLPredictors:
    def _mock_endpoint(self, mocker, predictions):
        mocker.patch(
            "edge_train.config.ensure_gcp_credentials", return_value=(True, "")
        )
        mocker.patch("google.cloud.aiplatform.init")
        endpoint = mocker.MagicMock()
        endpoint.predict.return_value = mocker.MagicMock(predictions=predictions)
        mocker.patch(
            "google.cloud.aiplatform.Endpoint",
            return_value=endpoint,
        )
        return endpoint

    def test_tabular_predict(self, mocker):
        endpoint = self._mock_endpoint(
            mocker,
            [{"displayNames": ["yes", "no"], "confidences": [0.8, 0.2]}],
        )
        predictor = VertexTabularPredictor(
            "projects/p/locations/us-central1/endpoints/1",
            project="p",
            location="us-central1",
        )
        label, conf = predictor.predict({"age": 30, "plan": "monthly"})
        assert label == "yes"
        assert conf == pytest.approx(0.8)
        endpoint.predict.assert_called_once()
        instances = endpoint.predict.call_args.kwargs["instances"]
        assert instances[0]["age"] == 30

    def test_image_predict_local_file(self, mocker, tmp_path):
        endpoint = self._mock_endpoint(
            mocker,
            [{"classes": ["cat"], "scores": [0.99]}],
        )
        img = tmp_path / "cat.jpg"
        img.write_bytes(b"\xff\xd8\xff fake jpeg")
        predictor = VertexImagePredictor(
            "projects/p/locations/us-central1/endpoints/2",
            project="p",
            location="us-central1",
        )
        label, conf = predictor.predict(str(img))
        assert label == "cat"
        instances = endpoint.predict.call_args.kwargs["instances"]
        assert "content" in instances[0]
        assert base64.b64decode(instances[0]["content"])

    def test_image_predict_gcs(self, mocker):
        endpoint = self._mock_endpoint(
            mocker,
            [{"classes": ["dog"], "scores": [0.7]}],
        )
        mock_storage = mocker.patch("google.cloud.storage.Client")
        bucket = mocker.MagicMock()
        blob = mocker.MagicMock()
        blob.download_as_bytes.return_value = b"\x89PNG fake image bytes"
        bucket.blob.return_value = blob
        mock_storage.return_value.get_bucket.return_value = bucket

        predictor = VertexImagePredictor(
            "projects/p/locations/us-central1/endpoints/2",
            project="p",
            location="us-central1",
        )
        label, _ = predictor.predict("gs://bucket/dog.png")
        assert label == "dog"
        mock_storage.return_value.get_bucket.assert_called_once_with("bucket")
        bucket.blob.assert_called_once_with("dog.png")
        instances = endpoint.predict.call_args.kwargs["instances"]
        assert "content" in instances[0]
        assert base64.b64decode(instances[0]["content"])
        assert instances[0]["mimeType"] == "image/png"

    def test_video_requires_gcs(self, mocker):
        self._mock_endpoint(mocker, [{"classes": ["action"], "scores": [0.6]}])
        predictor = VertexVideoPredictor(
            "projects/p/locations/us-central1/endpoints/3",
            project="p",
            location="us-central1",
        )
        with pytest.raises(ValueError, match="GCS URI"):
            predictor.predict("/local/video.mp4")

    def test_resolve_vertex_predictor_table(self, mocker):
        mocker.patch(
            "edge_train.config.ensure_gcp_credentials", return_value=(True, "")
        )
        mocker.patch("google.cloud.aiplatform.init")
        mocker.patch("google.cloud.aiplatform.Endpoint")
        predictor = resolve_vertex_predictor(
            "projects/p/locations/us-central1/endpoints/1",
            project="p",
            location="us-central1",
            modality="table",
        )
        assert isinstance(predictor, VertexTabularPredictor)


class TestDeploymentRegistryLookup:
    def test_find_by_endpoint(self, tmp_path, monkeypatch):
        from edge_train.deployments import DeploymentRecord, DeploymentRegistry

        path = tmp_path / "deployments.json"
        monkeypatch.setenv("CORALFLOW_DEPLOYMENTS_PATH", str(path))
        reg = DeploymentRegistry()
        reg.add(
            DeploymentRecord(
                model_path="projects/p/locations/us/models/1",
                target="vertex",
                endpoint_name="projects/p/locations/us/endpoints/9",
                modality="table",
            )
        )
        found = DeploymentRegistry.load().find_by_endpoint(
            "projects/p/locations/us/endpoints/9"
        )
        assert found is not None
        assert found.modality == "table"

    def test_phoenix_hint_table(self):
        from edge_train.deployments import format_phoenix_monitoring_hint

        hint = format_phoenix_monitoring_hint(
            endpoint="projects/p/locations/us/endpoints/9",
            modality="table",
        )
        assert "--modality table" in hint
