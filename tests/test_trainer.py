"""Tests for local trainer."""

from pathlib import Path

from edge_train.trainer import train_text_classifier, _detect_columns, _resolve_columns


class TestDetectColumns:
    def test_auto_detect(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text,label\nhello,greeting\nworld,farewell\n")
        text_col, label_col = _detect_columns(str(csv_path))
        assert text_col == "text"
        assert label_col == "label"

    def test_different_header_names(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("message,category\nhi,urgent\nbye,normal\n")
        text_col, label_col = _detect_columns(str(csv_path))
        assert text_col == "message"
        assert label_col == "category"


class TestResolveColumns:
    def test_explicit_target(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("msg,cat\nhello,a\nworld,b\n")
        text_col, label_col = _resolve_columns(str(csv_path), "cat")
        assert label_col == "cat"
        assert text_col == "msg"

    def test_auto_fallback(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text,label\nhello,greeting\n")
        text_col, label_col = _resolve_columns(str(csv_path), None)
        assert text_col == "text"
        assert label_col == "label"


class TestTrainTextClassifier:
    def test_returns_saved_model(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text,label\nhello world,greeting\nhow are you,question\n")

        out = train_text_classifier(
            str(csv_path), output_dir=str(tmp_path / "model"), epochs=2
        )
        assert out.exists()
        assert (out / "saved_model.pb").exists()
        assert (out / "model_meta.json").exists()

    def test_meta_has_classes_and_vocab(self, tmp_path):
        import json

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text,label\nhello world,greeting\nhow are you,question\n")

        out = train_text_classifier(
            str(csv_path), output_dir=str(tmp_path / "model"), epochs=2
        )
        meta = json.loads((out / "model_meta.json").read_text())
        assert "classes" in meta
        assert "vocabulary" in meta
        assert len(meta["classes"]) == 2
        assert "[UNK]" in meta["vocabulary"]

    def test_explicit_target_column(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("msg,cat\nhello world,a\nhow are you,b\n")

        out = train_text_classifier(
            str(csv_path),
            target_column="cat",
            output_dir=str(tmp_path / "model"),
            epochs=2,
        )
        assert out.exists()
        assert (out / "saved_model.pb").exists()

    def test_model_is_tflite_convertible(self, tmp_path):
        import tensorflow as tf

        csv_path = tmp_path / "data.csv"
        csv_path.write_text(
            "text,label\nhello world,greeting\nhow are you,question\ngoodbye,farewell\n"
        )

        out = train_text_classifier(
            str(csv_path), output_dir=str(tmp_path / "model"), epochs=2
        )
        converter = tf.lite.TFLiteConverter.from_saved_model(str(out))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite = converter.convert()
        assert len(tflite) > 0
        assert len(tflite) < 10 * 1024 * 1024

    def test_number_of_classes(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text,label\nhello,greeting\nhow are you,question\n")

        out = train_text_classifier(
            str(csv_path), output_dir=str(tmp_path / "model"), epochs=2
        )
        import json

        meta = json.loads((out / "model_meta.json").read_text())
        assert len(meta["classes"]) == 2
