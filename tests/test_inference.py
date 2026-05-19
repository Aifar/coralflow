"""Tests for local inference."""

import json
from pathlib import Path

from edge_train.inference import TextClassifier, log_prediction


class TestTextClassifier:
    def test_predict_returns_label_and_confidence(self, tmp_path):
        from edge_train.trainer import train_text_classifier

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text,label\nhello world,greeting\nhow are you,question\n")
        model_path = train_text_classifier(
            str(csv_path), output_dir=str(tmp_path / "model"), epochs=2
        )

        clf = TextClassifier(str(model_path))
        label, conf = clf.predict("hello world")
        assert isinstance(label, str)
        assert label in clf.class_names
        assert 0.0 <= conf <= 1.0

    def test_predict_proba_returns_all_classes(self, tmp_path):
        from edge_train.trainer import train_text_classifier

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text,label\nhello world,greeting\nhow are you,question\n")
        model_path = train_text_classifier(
            str(csv_path), output_dir=str(tmp_path / "model"), epochs=2
        )

        clf = TextClassifier(str(model_path))
        probs = clf.predict_proba("hello world")
        assert isinstance(probs, dict)
        assert set(probs.keys()) == set(clf.class_names)
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_batch(self, tmp_path):
        from edge_train.trainer import train_text_classifier

        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "text,label\nhello,greeting\nhow are you,question\ngoodbye,farewell\n"
        )
        model_path = train_text_classifier(
            str(csv_path), output_dir=str(tmp_path / "model"), epochs=2
        )

        clf = TextClassifier(str(model_path))
        results = clf.predict_batch(["hello", "goodbye"])
        assert len(results) == 2
        for label, conf in results:
            assert isinstance(label, str)
            assert 0.0 <= conf <= 1.0

    def test_num_classes(self, tmp_path):
        from edge_train.trainer import train_text_classifier

        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "text,label\nhello,greeting\nhow are you,question\ngoodbye,farewell\n"
        )
        model_path = train_text_classifier(
            str(csv_path), output_dir=str(tmp_path / "model"), epochs=2
        )

        clf = TextClassifier(str(model_path))
        assert clf.num_classes == 3
        assert clf.class_names == ["farewell", "greeting", "question"]

    def test_missing_model_dir_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            TextClassifier(str(tmp_path / "nonexistent"))

    def test_missing_meta_raises(self, tmp_path):
        import pytest

        (tmp_path / "model").mkdir()
        (tmp_path / "model" / "saved_model.pb").write_text("fake")
        with pytest.raises(FileNotFoundError, match="model_meta.json"):
            TextClassifier(str(tmp_path / "model"))


class TestLogPrediction:
    def test_appends_to_log_file(self, tmp_path):
        log_path = str(tmp_path / "preds.jsonl")
        log_prediction(
            log_path,
            text="hello",
            predicted_label="greeting",
            confidence=0.95,
            all_probs={"greeting": 0.95, "question": 0.05},
        )
        log_prediction(
            log_path,
            text="world",
            predicted_label="question",
            confidence=0.60,
            all_probs={"greeting": 0.40, "question": 0.60},
            ground_truth="greeting",
        )

        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 2

        e1 = json.loads(lines[0])
        assert e1["text"] == "hello"
        assert e1["predicted_label"] == "greeting"
        assert "ground_truth" not in e1

        e2 = json.loads(lines[1])
        assert e2["text"] == "world"
        assert e2["ground_truth"] == "greeting"

    def test_creates_parent_directory(self, tmp_path):
        log_path = str(tmp_path / "sub" / "preds.jsonl")
        log_prediction(
            log_path,
            text="test",
            predicted_label="a",
            confidence=1.0,
            all_probs={"a": 1.0},
        )
        assert Path(log_path).exists()
