"""Route cloud training jobs to the best Vertex AI service for each dataset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CloudTrainingMethod(str, Enum):
    GEMINI_FINETUNE = "gemini_finetune"
    AUTOML_TABULAR = "automl_tabular"
    AUTOML_IMAGE = "automl_image"
    AUTOML_VIDEO = "automl_video"


METHOD_LABELS = {
    CloudTrainingMethod.GEMINI_FINETUNE: "Gemini Fine-Tuning",
    CloudTrainingMethod.AUTOML_TABULAR: "AutoML Tabular",
    CloudTrainingMethod.AUTOML_IMAGE: "AutoML Image",
    CloudTrainingMethod.AUTOML_VIDEO: "AutoML Video",
}

METHOD_REASONS = {
    CloudTrainingMethod.GEMINI_FINETUNE: (
        "Text classification — AutoML Text is deprecated; Gemini SFT is the "
        "supported Vertex path for free-text inputs."
    ),
    CloudTrainingMethod.AUTOML_TABULAR: (
        "Structured tabular data — AutoML Tabular remains best-in-class for "
        "numeric/categorical features (churn, credit risk, pricing)."
    ),
    CloudTrainingMethod.AUTOML_IMAGE: (
        "Image data — AutoML Image for high-accuracy vision classification "
        "and object detection workloads."
    ),
    CloudTrainingMethod.AUTOML_VIDEO: (
        "Video data — AutoML Video for classification and action recognition."
    ),
}


@dataclass(frozen=True)
class CloudTrainingPlan:
    method: CloudTrainingMethod
    modality: str
    reason: str
    status: str = "active"

    @property
    def label(self) -> str:
        return METHOD_LABELS[self.method]


def plan_cloud_training(
    dataset_path: str,
    modality: str | None = None,
) -> CloudTrainingPlan:
    """Choose Gemini fine-tuning vs AutoML Tabular/Image/Video for a dataset."""
    resolved = (modality or _infer_cloud_modality(dataset_path)).lower().strip()

    if resolved == "sound":
        raise ValueError(
            "Cloud training for audio is not supported yet. "
            "Use local training or convert to a supported modality."
        )
    if resolved == "unknown":
        raise ValueError(
            "Could not determine dataset modality for cloud training. "
            "Use --type text|table|image|video."
        )

    method = _method_for_modality(resolved)
    return CloudTrainingPlan(
        method=method,
        modality=resolved,
        reason=METHOD_REASONS[method],
    )


def cloud_modality_supported(modality: str) -> tuple[bool, str]:
    """Return whether cloud training is available for this modality."""
    mod = modality.lower().strip()
    if mod == "sound":
        return False, (
            "Cloud training for audio is not supported yet. "
            "Use local training or convert to a supported modality."
        )
    try:
        _method_for_modality(mod)
    except ValueError as exc:
        return False, str(exc)
    return True, ""


def _method_for_modality(modality: str) -> CloudTrainingMethod:
    mapping = {
        "text": CloudTrainingMethod.GEMINI_FINETUNE,
        "table": CloudTrainingMethod.AUTOML_TABULAR,
        "image": CloudTrainingMethod.AUTOML_IMAGE,
        "video": CloudTrainingMethod.AUTOML_VIDEO,
    }
    method = mapping.get(modality)
    if method is None:
        raise ValueError(
            f"Unsupported cloud modality '{modality}'. "
            "Use text, table, image, or video."
        )
    return method


def _infer_cloud_modality(dataset_path: str) -> str:
    from edge_train.datasets import infer_modality_from_path

    path = Path(dataset_path)
    if path.is_dir():
        return _infer_modality_from_directory(path)

    return infer_modality_from_path(dataset_path)


def _infer_modality_from_directory(path: Path) -> str:
    video_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

    videos = sum(1 for p in path.rglob("*") if p.suffix.lower() in video_ext)
    images = sum(1 for p in path.rglob("*") if p.suffix.lower() in image_ext)

    if videos and videos >= images:
        return "video"
    if images:
        return "image"
    return "unknown"
