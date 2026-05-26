"""Tests for demo retrain-loop and shared retrain helpers."""

import json

import pytest
from click.testing import CliRunner

from edge_train.cli import main
from edge_train.demo.challenges import URGENT_DRIFT_CASES, write_urgent_challenge_csv
from edge_train.retrain import (
    compute_accuracy,
    evaluate_cases,
    filter_csv_by_labels,
    write_prediction_log,
)


class TestRetrainHelpers:
    def test_compute_accuracy(self):
        entries = [
            {"predicted_label": "a", "ground_truth": "a"},
            {"predicted_label": "b", "ground_truth": "a"},
        ]
        acc, correct, total = compute_accuracy(entries)
        assert total == 2
        assert correct == 1
        assert acc == 0.5

    def test_filter_csv_by_labels(self, sample_text_csv, tmp_path):
        out = tmp_path / "partial.csv"
        n = filter_csv_by_labels(sample_text_csv, str(out), ("farewell",))
        assert n == 2
        assert "farewell" in out.read_text(encoding="utf-8")


class TestDemoRetrainLoop:
    @pytest.mark.slow
    def test_retrain_loop_improves_accuracy(self, tmp_path):
        """End-to-end: weak baseline → low acc → retrain → higher acc on challenges."""
        work = tmp_path / "demo"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "demo",
                "retrain-loop",
                "-w",
                str(work),
                "--epochs",
                "2",
                "--retrain-epochs",
                "3",
                "--hard",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "Baseline:" in result.output
        assert "Retrained:" in result.output
        assert "retraining" in result.output.lower()
        assert "improved" in result.output.lower() or "✓" in result.output

        log_path = work / "prediction_log.jsonl"
        assert log_path.exists()
        lines = [json.loads(ln) for ln in log_path.read_text().strip().splitlines()]
        assert len(lines) == len(URGENT_DRIFT_CASES)
        assert all("ground_truth" in e for e in lines)

    def test_challenge_csv_written(self, tmp_path):
        path = write_urgent_challenge_csv(tmp_path / "c.csv")
        assert path.exists()
        assert (
            path.read_text(encoding="utf-8").count("\n") == len(URGENT_DRIFT_CASES) + 1
        )

    def test_write_prediction_log_includes_ground_truth(
        self, sample_text_csv, tmp_path
    ):
        from edge_train.trainer import train_text_classifier
        from edge_train.inference import TextClassifier

        model_dir = tmp_path / "m"
        train_text_classifier(sample_text_csv, output_dir=str(model_dir), epochs=1)
        clf = TextClassifier(str(model_dir))
        cases = [("buy viagra now", "spam"), ("team meeting tomorrow", "ham")]
        log = tmp_path / "log.jsonl"
        write_prediction_log(log, cases, clf)
        entries = [json.loads(ln) for ln in log.read_text().strip().splitlines()]
        assert len(entries) == 2
        assert entries[0]["ground_truth"] in ("spam", "ham")
