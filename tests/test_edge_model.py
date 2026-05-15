"""Tests for model packaging."""

import json
import hashlib
import pytest
from edge_train.edge.model import ModelPackage, ModelManifest


class TestModelManifest:
    def test_default_framework(self):
        m = ModelManifest(
            version="1.0",
            sha256="abc",
            modality="text",
            model_size_bytes=100,
            timestamp="2026-01-01T00:00:00Z",
        )
        assert m.framework == "tflite"


class TestModelPackage:
    def test_creates_manifest(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake tflite content")

        pkg = ModelPackage(tflite, version="0.2.0", modality="text")
        m = pkg.manifest
        assert m.version == "0.2.0"
        assert m.modality == "text"
        assert m.model_size_bytes == len(b"fake tflite content")
        assert isinstance(m.sha256, str) and len(m.sha256) == 64

    def test_correct_sha256(self, tmp_path):
        content = b"hello tflite model bytes"
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(content)

        pkg = ModelPackage(tflite, version="1.0", modality="text")
        expected = hashlib.sha256(content).hexdigest()
        assert pkg.manifest.sha256 == expected

    def test_model_bytes(self, tmp_path):
        content = b"some model data"
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(content)
        pkg = ModelPackage(tflite, version="1.0", modality="text")
        assert pkg.model_bytes == content

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ModelPackage("/nonexistent/model.tflite", version="1.0")

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "model.txt"
        f.write_bytes(b"data")
        with pytest.raises(ValueError, match="Not a TFLite"):
            ModelPackage(f, version="1.0")

    def test_write_package(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"fake model")
        pkg = ModelPackage(tflite, version="0.2.0", modality="text")

        out_dir = tmp_path / "package"
        result = pkg.write_package(out_dir)
        assert result == out_dir
        assert (out_dir / "model.tflite").exists()
        assert (out_dir / "manifest.json").exists()

        manifest_data = json.loads((out_dir / "manifest.json").read_text())
        assert manifest_data["version"] == "0.2.0"
        assert manifest_data["modality"] == "text"
        assert manifest_data["framework"] == "tflite"

    def test_manifest_timestamp_format(self, tmp_path):
        tflite = tmp_path / "model.tflite"
        tflite.write_bytes(b"data")
        pkg = ModelPackage(tflite, version="1.0", modality="text")
        ts = pkg.manifest.timestamp
        assert ts.endswith("Z")
        assert "T" in ts
