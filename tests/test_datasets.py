"""Tests for edge_train.datasets."""

from pathlib import Path

from edge_train.datasets import (
    get_builtin,
    infer_modality,
    infer_modality_from_path,
)


class TestBuiltinDatasets:
    def test_get_builtin_returns_dict(self):
        datasets = get_builtin()
        assert isinstance(datasets, dict)
        assert "urgent" in datasets
        assert "expense" in datasets

    def test_urgent_dataset_structure(self):
        ds = get_builtin()["urgent"]
        assert ds["modality"] == "text"
        assert ds["samples"] == 400
        assert len(ds["classes"]) == 4
        assert "紧急" in ds["classes"]
        assert "csv_content" in ds

    def test_expense_dataset_structure(self):
        ds = get_builtin()["expense"]
        assert ds["modality"] == "text"
        assert ds["samples"] == 500
        assert len(ds["classes"]) == 5
        assert "餐饮" in ds["classes"]

    def test_csv_content_has_header(self):
        csv_content = get_builtin()["urgent"]["csv_content"]
        lines = csv_content.split("\n")
        assert lines[0] == "text,urgency"

    def test_csv_content_parsable(self):
        import csv
        import io

        csv_content = get_builtin()["expense"]["csv_content"]
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
        assert len(rows) == 500
        assert all("text" in row and "category" in row for row in rows)

    def test_list_builtin_equals_get(self):
        from edge_train.datasets import list_builtin

        assert list_builtin() == get_builtin()


class TestInferModality:
    def test_csv_text_keyword(self, temp_dir):
        """CSV with 'text' column → text modality."""
        path = temp_dir / "data.csv"
        path.write_text("text,label\nhello,world\n", encoding="utf-8")
        assert infer_modality(path) == "text"

    def test_csv_numeric(self, temp_dir):
        """Mostly numeric columns → table modality."""
        path = temp_dir / "data.csv"
        path.write_text("age,height,score,label\n25,180,95,A\n30,170,88,B\n", encoding="utf-8")
        assert infer_modality(path) == "table"

    def test_image_extensions(self, temp_dir):
        assert infer_modality(temp_dir / "photo.jpg") == "image"
        assert infer_modality(temp_dir / "photo.jpeg") == "image"
        assert infer_modality(temp_dir / "photo.png") == "image"
        assert infer_modality(temp_dir / "photo.webp") == "image"

    def test_sound_extensions(self, temp_dir):
        assert infer_modality(temp_dir / "audio.wav") == "sound"
        assert infer_modality(temp_dir / "audio.mp3") == "sound"
        assert infer_modality(temp_dir / "audio.flac") == "sound"

    def test_unknown_extension(self, temp_dir):
        assert infer_modality(temp_dir / "data.parquet") == "unknown"

    def test_infer_modality_from_path_calls_through(self, temp_dir):
        p = temp_dir / "test.csv"
        p.write_text("content,rating\nnice,5\n", encoding="utf-8")
        assert infer_modality_from_path(str(p)) == "text"
