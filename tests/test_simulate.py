"""Tests for Phoenix smoke-test simulate runs."""

import pytest

from edge_train.simulate import (
    DEFAULT_TEXT_SAMPLES,
    format_simulate_command,
    get_simulation_samples,
    run_simulation,
)


class TestFormatSimulateCommand:
    def test_vertex_text(self):
        cmd = format_simulate_command(
            endpoint="projects/p/locations/us-central1/endpoints/1"
        )
        assert "coralflow simulate" in cmd
        assert "--endpoint" in cmd
        assert "endpoints/1" in cmd

    def test_vertex_table(self):
        cmd = format_simulate_command(
            endpoint="projects/p/locations/us-central1/endpoints/1",
            modality="table",
        )
        assert "--modality table" in cmd

    def test_local_model(self):
        cmd = format_simulate_command(model="./model_output")
        assert "--model ./model_output" in cmd


class TestGetSimulationSamples:
    def test_text_defaults(self):
        samples = get_simulation_samples("text", count=3)
        assert len(samples) == 3
        assert samples[0] in DEFAULT_TEXT_SAMPLES

    def test_text_from_dataset(self, sample_text_csv):
        samples = get_simulation_samples("text", count=2, dataset=sample_text_csv)
        assert len(samples) == 2
        assert all(isinstance(s, str) for s in samples)

    def test_table_defaults(self):
        samples = get_simulation_samples("table", count=2)
        assert len(samples) == 2
        assert isinstance(samples[0], dict)

    def test_image_requires_input(self):
        with pytest.raises(ValueError, match="Image simulate"):
            get_simulation_samples("image", count=1)


class TestRunSimulation:
    def test_local_text_simulate(self, mocker, monkeypatch, tmp_path, sample_text_csv):
        from edge_train.trainer import train_text_classifier

        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://example.com/v1/traces"
        )
        monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
        monkeypatch.setenv(
            "EDGE_PREDICTION_LOG_PATH", str(tmp_path / "prediction_log.jsonl")
        )

        model_path = train_text_classifier(
            sample_text_csv, output_dir=str(tmp_path / "model"), epochs=1
        )
        mocker.patch(
            "edge_train.inference.phoenix.prepare_phoenix_for_inference",
            return_value=(True, ""),
        )

        result = run_simulation(model=model_path, count=2)
        assert result.count == 2
        assert result.phoenix_active is True
        assert (tmp_path / "prediction_log.jsonl").exists()

    def test_vertex_simulate(self, mocker, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "PHOENIX_COLLECTOR_ENDPOINT", "https://example.com/v1/traces"
        )
        monkeypatch.setenv("PHOENIX_API_KEY", "test-key")
        monkeypatch.setenv(
            "EDGE_PREDICTION_LOG_PATH", str(tmp_path / "prediction_log.jsonl")
        )

        mock_predictor = mocker.MagicMock()
        mock_predictor.predict.return_value = ("urgent", 0.9)
        mock_predictor.predict_proba.return_value = {"urgent": 0.9}
        mock_predictor.format_input.side_effect = lambda x: str(x)
        mocker.patch(
            "edge_train.cli.predict._load_vertex_predictor",
            return_value=mock_predictor,
        )
        mocker.patch(
            "edge_train.inference.phoenix.prepare_phoenix_for_inference",
            return_value=(True, ""),
        )

        result = run_simulation(
            endpoint="projects/p/locations/us-central1/endpoints/1",
            modality="text",
            count=3,
        )
        assert result.count == 3
        assert mock_predictor.predict.call_count == 3
