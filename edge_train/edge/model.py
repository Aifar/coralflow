"""Model packaging — wrap a .tflite file with a versioned manifest."""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ModelManifest:
    """Metadata accompanying a deployed TFLite model."""

    version: str
    sha256: str
    modality: str
    model_size_bytes: int
    timestamp: str
    framework: str = "tflite"
    metadata: dict = field(default_factory=dict)


class ModelPackage:
    """Wraps a .tflite file and provides a deployable package with manifest."""

    def __init__(
        self, tflite_path: str | Path, version: str, modality: str = "text"
    ) -> None:
        self._path = Path(tflite_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Model not found: {self._path}")
        if self._path.suffix.lower() != ".tflite":
            raise ValueError(f"Not a TFLite model: {self._path}")
        self._version = version
        self._modality = modality
        self._manifest: ModelManifest | None = None

    @property
    def manifest(self) -> ModelManifest:
        if self._manifest is None:
            self._manifest = self._build_manifest()
        return self._manifest

    @property
    def model_bytes(self) -> bytes:
        return self._path.read_bytes()

    def write_package(self, output_dir: str | Path) -> Path:
        """Write model + manifest.json into output_dir, return output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_dest = out / self._path.name
        model_dest.write_bytes(self.model_bytes)

        manifest_dest = out / "manifest.json"
        manifest_dest.write_text(
            json.dumps(asdict(self.manifest), indent=2, ensure_ascii=False)
        )
        return out

    def _build_manifest(self) -> ModelManifest:
        sha256 = hashlib.sha256()
        with self._path.open("rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        stat = self._path.stat()
        return ModelManifest(
            version=self._version,
            sha256=sha256.hexdigest(),
            modality=self._modality,
            model_size_bytes=stat.st_size,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
