"""Query Vertex AI publisher (foundation) models vs project custom models."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Models in Model Garden that are not suitable Gemini SFT base models.
_FINETUNE_EXCLUDE_RE = re.compile(
    r"(embedding|embed|tts|image|live|computer-use|video|audio|vision)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PublisherModelInfo:
    """A Google publisher model from Model Garden."""

    publisher: str
    model_id: str
    version: str
    qualified_name: str
    resource_name: str

    @property
    def tuning_source_model(self) -> str:
        """Short model ID accepted by vertexai.tuning.sft.train(source_model=...)."""
        return self.model_id


def parse_model_garden_entry(entry: str, location: str) -> PublisherModelInfo:
    """Parse ``google/gemini-2.0-flash-001@default`` into structured metadata."""
    publisher, rest = entry.split("/", 1)
    if "@" in rest:
        model_id, version = rest.rsplit("@", 1)
    else:
        model_id, version = rest, "default"

    resource_name = (
        f"projects/google/locations/{location}/publishers/{publisher}/models/{model_id}"
    )
    return PublisherModelInfo(
        publisher=publisher,
        model_id=model_id,
        version=version,
        qualified_name=f"{publisher}/{model_id}@{version}",
        resource_name=resource_name,
    )


def is_likely_finetune_base_model(model_id: str) -> bool:
    """Heuristic: text Gemini models suitable for supervised fine-tuning."""
    if not model_id.lower().startswith("gemini"):
        return False
    return _FINETUNE_EXCLUDE_RE.search(model_id) is None


def list_publisher_models(
    project: str,
    location: str = "us-central1",
    *,
    model_filter: str | None = "gemini",
    finetune_capable_only: bool = True,
) -> list[PublisherModelInfo]:
    """List Google publisher models from Model Garden (not your custom models).

    ``ModelServiceClient.list_models`` only returns models saved in your project's
    Model Registry. Foundation models live under ``publishers/google`` and must be
    queried via Model Garden instead.
    """
    import vertexai
    from vertexai import model_garden

    vertexai.init(project=project, location=location)
    entries = model_garden.list_models(model_filter=model_filter)

    models = [parse_model_garden_entry(entry, location) for entry in entries]
    if finetune_capable_only:
        models = [m for m in models if is_likely_finetune_base_model(m.model_id)]
    return sorted(models, key=lambda m: m.model_id)


def list_custom_models(project: str, location: str = "us-central1") -> list[str]:
    """List custom models in the current project's Model Registry."""
    from google.cloud.aiplatform_v1 import ModelServiceClient

    client = ModelServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    parent = f"projects/{project}/locations/{location}"
    return [model.name for model in client.list_models(parent=parent)]


def resolve_finetune_base_model(
    model_id: str,
    project: str,
    location: str = "us-central1",
    *,
    validate: bool = True,
) -> str:
    """Return the short model ID for SFT, optionally validating against Model Garden."""
    raw = model_id.strip()
    if raw.startswith("projects/"):
        # projects/google/locations/.../models/gemini-2.0-flash-001[@version]
        match = re.search(r"/models/([^/@]+)", raw)
        if not match:
            raise ValueError(f"Cannot parse publisher model path: {raw}")
        raw = match.group(1)

    if "/" in raw:
        # google/gemini-2.0-flash-001@default
        parsed = parse_model_garden_entry(raw, location)
        raw = parsed.model_id

    if not validate:
        return raw

    try:
        available = {m.model_id for m in list_publisher_models(project, location)}
    except Exception:
        return raw

    if available and raw not in available:
        sample = ", ".join(sorted(available)[:5])
        raise ValueError(
            f"Unknown fine-tune base model '{raw}'. "
            f"Run `coralflow models list` to see publisher models. "
            f"Examples: {sample}"
        )
    return raw


def normalize_finetune_model_id(model_id: str) -> str:
    """Return the short publisher model ID from env, path, or qualified name."""
    raw = model_id.strip()
    if raw.startswith("projects/"):
        match = re.search(r"/models/([^/@]+)", raw)
        if match:
            return match.group(1)
        raise ValueError(f"Cannot parse publisher model path: {raw}")
    if "/" in raw:
        return parse_model_garden_entry(raw, "us-central1").model_id
    return raw


def publisher_model_resource_name(model_id: str, location: str = "us-central1") -> str:
    """Full Vertex resource path for a Google publisher model."""
    short_id = normalize_finetune_model_id(model_id)
    return f"projects/google/locations/{location}/publishers/google/models/{short_id}"


def describe_finetune_base_model(
    model_id: str,
    location: str = "us-central1",
    *,
    markdown: bool = False,
) -> list[str]:
    """User-facing lines explaining which foundation model SFT will fine-tune."""
    short_id = normalize_finetune_model_id(model_id)
    resource = publisher_model_resource_name(short_id, location)
    if markdown:
        return [
            f"**Fine-tune base model:** `{short_id}`",
            "_(Google Gemini publisher model — your labels adapt this foundation model)_",
            f"Publisher path: `{resource}`",
            "Change model: `GCP_FINETUNE_MODEL=<model-id>` or run `coralflow models list`",
        ]
    return [
        f"Fine-tune base model: {short_id}",
        "(Google Gemini publisher model — your labels adapt this foundation model)",
        f"Publisher path: {resource}",
        "Change model: GCP_FINETUNE_MODEL=<model-id>  (see: coralflow models list)",
    ]
