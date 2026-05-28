"""Tests for batch evaluation helpers."""

import json

import pytest

from edge_train.evaluate import (
    format_inference_environment,
    parse_labeled_sample,
    run_vertex_evaluation,
)
from edge_train.inference.phoenix import PhoenixPrepareResult


class TestParseLabeledSample:
    def test_vertex_image_jsonl(self):
        row = {
            "imageGcsUri": "gs://bucket/test/img_0.png",
            "classificationAnnotation": {"displayName": "crazing"},
        }
        uri, label = parse_labeled_sample(row)
        assert uri == "gs://bucket/test/img_0.png"
        assert label == "crazing"

    def test_simple_label(self):
        row = {"gcsUri": "gs://b/x.png", "label": "scratches"}
        uri, label = parse_labeled_sample(row)
        assert uri == "gs://b/x.png"
        assert label == "scratches"


class TestFormatInferenceEnvironment:
    def test_vertex_banner(self):
        text = format_inference_environment(
            source="vertex",
            modality="image",
            endpoint="projects/p/locations/us-central1/endpoints/1",
            project="p",
            location="us-central1",
        )
        assert "Vertex AI" in text
        assert "us-central1" in text
        assert "image" in text


class TestRunVertexEvaluation:
    def test_batch_eval_logs_and_accuracy(self, mocker, tmp_path):
        mocker.patch(
            "edge_train.config.ensure_gcp_credentials", return_value=(True, "")
        )
        mock_arize = mocker.MagicMock()
        mock_arize.is_valid.return_value = True
        mock_arize.collector_endpoint = "https://app.phoenix.arize.com/v1/traces"
        mock_arize.project_name = "edge-train"
        mocker.patch(
            "edge_train.config.load_config",
            return_value=(
                mocker.MagicMock(),
                mock_arize,
                mocker.MagicMock(),
                mocker.MagicMock(),
            ),
        )
        mocker.patch(
            "edge_train.inference.phoenix.prepare_phoenix_for_inference",
            return_value=PhoenixPrepareResult(True, "", False),
        )
        mocker.patch("google.cloud.aiplatform.init")

        mock_classifier = mocker.MagicMock()
        mock_classifier.predict_batch.return_value = [
            ("crazing", 0.9),
            ("inclusion", 0.8),
        ]
        mock_classifier.predict_proba.side_effect = [
            {"crazing": 0.9, "inclusion": 0.1},
            {"inclusion": 0.8, "crazing": 0.2},
        ]
        mock_classifier.format_input.side_effect = lambda x: str(x)

        mocker.patch(
            "edge_train.cli.predict._load_vertex_predictor",
            return_value=mock_classifier,
        )
        mocker.patch(
            "edge_train.cli.predict._resolve_endpoint_modality",
            return_value="image",
        )

        jsonl = tmp_path / "test.jsonl"
        rows = [
            {
                "imageGcsUri": "gs://b/a.png",
                "classificationAnnotation": {"displayName": "crazing"},
            },
            {
                "imageGcsUri": "gs://b/b.png",
                "classificationAnnotation": {"displayName": "inclusion"},
            },
        ]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        log_path = tmp_path / "log.jsonl"
        result = run_vertex_evaluation(
            endpoint="projects/p/locations/us/endpoints/1",
            modality="image",
            dataset_path=str(jsonl),
            project="p",
            location="us",
            log_path=str(log_path),
            batch_size=8,
        )

        assert result.total == 2
        assert result.labeled == 2
        assert result.correct == 2
        assert result.accuracy == pytest.approx(1.0)
        assert log_path.exists()
        logged = [json.loads(line) for line in log_path.read_text().splitlines()]
        assert len(logged) == 2
        assert logged[0]["ground_truth"] == "crazing"
