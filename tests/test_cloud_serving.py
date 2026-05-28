"""Tests for Vertex AI serving helpers."""

import pytest

from edge_train.cloud.serving import (
    VertexTextPredictor,
    deploy_model_to_vertex,
    is_vertex_endpoint,
    is_vertex_resource,
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


class TestDeployModelToVertex:
    def test_deploy_calls_aiplatform(self, mocker):
        mock_init = mocker.patch("google.cloud.aiplatform.init")
        mock_model_cls = mocker.patch("google.cloud.aiplatform.Model")
        endpoint = mocker.MagicMock()
        endpoint.resource_name = "projects/p/locations/us/endpoints/1"
        endpoint.list_models.return_value = [mocker.MagicMock(id="dm-1")]
        mock_model = mock_model_cls.return_value
        mock_model.display_name = "test-model"
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


class TestVertexTextPredictor:
    def test_predict_parses_label(self, mocker):
        mocker.patch("vertexai.init")
        mock_model_cls = mocker.patch("vertexai.generative_models.GenerativeModel")
        response = mocker.MagicMock()
        response.text = "urgent\n"
        mock_model_cls.return_value.generate_content.return_value = response
        mocker.patch(
            "edge_train.config.ensure_gcp_credentials", return_value=(True, "")
        )

        predictor = VertexTextPredictor(
            "projects/p/locations/us/endpoints/1",
            project="p",
            location="us",
        )
        label, conf = predictor.predict("need this today")
        assert label == "urgent"
        assert conf == 1.0

    def test_rejects_non_endpoint(self):
        with pytest.raises(ValueError, match="endpoint"):
            VertexTextPredictor(
                "projects/p/locations/us/models/1",
                project="p",
                location="us",
            )
