"""Agent state, dataset scanning, and training recommendations."""

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path.home() / ".coralflow"
STATE_FILE = STATE_DIR / "agent_state.json"


@dataclass
class AgentState:
    """Persistent agent state for resumability across sessions."""

    dataset_path: str = ""
    model_path: str | None = None
    task_type: str = ""
    deployment_target: str | None = None
    last_step: str = ""
    created_at: str = ""
    conversation_summary: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def save(self, path: str | None = None) -> None:
        """Persist state to JSON file."""
        target = Path(path or STATE_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | None = None) -> "AgentState":
        """Load state from JSON file. Returns default if file doesn't exist."""
        target = Path(path or STATE_FILE)
        if not target.exists():
            return cls()
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DatasetScanner:
    """Discover datasets from filesystem and built-in catalog."""

    @staticmethod
    def scan(base_dirs: list[str] | None = None) -> list[dict]:
        """Scan for CSV datasets and return metadata for each."""
        if base_dirs is None:
            base_dirs = ["./data", ".", str(Path.home() / ".coralflow")]

        found: dict[str, dict] = {}

        # Filesystem scan
        for base in base_dirs:
            base_path = Path(base)
            if not base_path.exists():
                continue
            for csv_file in base_path.rglob("*.csv"):
                try:
                    info = DatasetScanner._analyze_csv(csv_file)
                    if info:
                        key = str(csv_file.resolve())
                        info["name"] = csv_file.stem
                        info["path"] = key
                        info["source"] = "local"
                        found[key] = info
                except Exception:
                    pass

        # Built-in datasets
        try:
            from edge_train.datasets import list_builtin

            for name, meta in list_builtin().items():
                info = {
                    "name": name,
                    "path": f"builtin:{name}",
                    "modality": meta.get("modality", "text"),
                    "rows": meta.get("samples", 0),
                    "classes": meta.get("classes", []),
                    "source": "built-in",
                    "description": meta.get("description", ""),
                }
                found[f"builtin:{name}"] = info
        except ImportError:
            pass

        return sorted(found.values(), key=lambda x: x["name"])

    @staticmethod
    def _analyze_csv(path: Path) -> dict | None:
        """Quickly inspect a CSV file for dataset metadata."""
        try:
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = list(reader)

            if not rows or not headers:
                return None

            # Detect columns
            text_col, label_col = _detect_text_label_columns(headers)

            classes = (
                sorted(set(r[label_col] for r in rows))
                if label_col and label_col in rows[0]
                else []
            )

            # Infer modality
            from edge_train.datasets import infer_modality

            modality = infer_modality(path)

            return {
                "modality": modality,
                "rows": len(rows),
                "classes": classes,
                "headers": headers,
                "text_col": text_col,
                "label_col": label_col,
            }
        except Exception:
            return None


def _detect_text_label_columns(headers: list[str]) -> tuple[str | None, str | None]:
    """Auto-detect text and label columns from CSV headers."""
    text_keywords = {
        "text",
        "message",
        "content",
        "sentence",
        "review",
        "comment",
        "description",
    }
    label_keywords = {
        "label",
        "class",
        "category",
        "target",
        "intent",
        "urgency",
        "ground_truth",
    }

    text_col = None
    label_col = None

    for h in headers:
        lower = h.lower().strip()
        if lower in text_keywords and text_col is None:
            text_col = h
        elif lower in label_keywords and label_col is None:
            label_col = h

    # Fallback: first non-label column as text, first label-like as label
    if text_col is None:
        for h in headers:
            if h != label_col:
                text_col = h
                break
    if label_col is None:
        for h in headers:
            if h != text_col:
                label_col = h
                break

    return text_col, label_col


class Recommender:
    """Recommend training approach based on dataset characteristics."""

    @staticmethod
    def recommend(dataset_info: dict) -> dict:
        """Return a recommendation dict with method, epochs, and reason."""
        rows = dataset_info.get("rows", 0)
        modality = dataset_info.get("modality", "text")
        num_classes = len(dataset_info.get("classes", []))

        if modality in ("image", "table", "sound"):
            return {
                "method": "cloud",
                "epochs": None,
                "reason": f"{modality.capitalize()} modality requires cloud training (Vertex AI AutoML).",
            }

        if rows >= 10000:
            return {
                "method": "cloud",
                "epochs": None,
                "reason": f"Large dataset ({rows:,} rows) — cloud training recommended for speed.",
            }

        # Local training recommendation
        if rows < 200:
            epochs = 20
        elif rows < 1000:
            epochs = 15
        elif rows < 5000:
            epochs = 10
        else:
            epochs = 8

        return {
            "method": "local",
            "epochs": epochs,
            "reason": f"{rows:,} rows, {num_classes} classes — small dataset fits local CPU training.",
        }


def scan_models(dirs: list[str] | None = None) -> list[dict]:
    """Scan for trained SavedModel directories."""
    if dirs is None:
        dirs = ["./model_output", "./models"]

    found = []
    for base in dirs:
        base_path = Path(base)
        if not base_path.exists():
            continue
        for pb in base_path.rglob("saved_model.pb"):
            model_dir = pb.parent
            meta_path = model_dir / "model_meta.json"
            classes = []
            created_at = ""
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    classes = meta.get("classes", [])
                except Exception:
                    pass
            try:
                created_at = datetime.fromtimestamp(
                    pb.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                pass
            found.append(
                {
                    "name": (
                        model_dir.parent.name + "/" + model_dir.name
                        if model_dir.parent.name not in ("model_output", "models", ".")
                        else model_dir.name
                    ),
                    "path": str(model_dir.resolve()),
                    "classes": classes,
                    "created_at": created_at,
                }
            )

    return sorted(found, key=lambda x: x["name"])
