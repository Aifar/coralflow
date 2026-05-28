"""Deploy Vertex AI models to endpoints and run online inference."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VERTEX_RESOURCE_RE = re.compile(
    r"^projects/[^/]+/locations/[^/]+/(models|endpoints|publishers)/"
)


def is_vertex_resource(name: str) -> bool:
    return bool(_VERTEX_RESOURCE_RE.match(name.strip()))


def is_vertex_endpoint(name: str) -> bool:
    return "/endpoints/" in name


@dataclass
class VertexDeployResult:
    model_path: str
    endpoint_name: str
    deployed_model_id: str = ""


def model_supports_dedicated_deployment(model) -> bool:
    """Return True when the Vertex model accepts machine_type / dedicated resources."""
    try:
        gca = model.gca_resource
        types = list(getattr(gca, "supported_deployment_resources_types", []) or [])
        if not types:
            return False
        return any("DEDICATED" in str(t).upper() for t in types)
    except Exception:
        return True


def deploy_model_to_vertex(
    model_name: str,
    *,
    project: str,
    location: str,
    display_name: str | None = None,
    machine_type: str = "n1-standard-2",
    min_replica_count: int = 1,
    max_replica_count: int = 1,
) -> VertexDeployResult:
    """Deploy a Vertex model resource to a new online prediction endpoint."""
    from edge_train.config import ensure_gcp_credentials

    ok, err = ensure_gcp_credentials()
    if not ok:
        raise RuntimeError(err)

    import google.cloud.aiplatform as aip

    aip.init(project=project, location=location)
    model = aip.Model(model_name)
    deploy_kwargs: dict[str, Any] = {
        "deployed_model_display_name": display_name
        or f"coralflow-{model.display_name}",
        "sync": True,
    }
    # AutoML Image/Tabular/Video models require automatic_resources only.
    if model_supports_dedicated_deployment(model):
        deploy_kwargs["machine_type"] = machine_type
        deploy_kwargs["min_replica_count"] = min_replica_count
        deploy_kwargs["max_replica_count"] = max_replica_count
    endpoint = model.deploy(**deploy_kwargs)
    deployed = endpoint.list_models()[0] if endpoint.list_models() else None
    return VertexDeployResult(
        model_path=model_name,
        endpoint_name=endpoint.resource_name,
        deployed_model_id=getattr(deployed, "id", "") if deployed else "",
    )


def parse_automl_classification(pred: Any) -> tuple[str, float, dict[str, float]]:
    """Normalize Vertex AutoML classification responses to label, confidence, probs."""
    if isinstance(pred, list) and pred:
        pred = pred[0]
    if not isinstance(pred, dict):
        label = str(pred)
        return label, 1.0, {label: 1.0}

    for key in ("predicted_class", "predictedClass", "class", "displayName"):
        if key in pred and pred[key] is not None:
            label = str(pred[key])
            conf = float(
                pred.get("confidence")
                or pred.get("score")
                or pred.get("maxConfidence")
                or 1.0
            )
            return label, conf, {label: conf}

    classes = (
        pred.get("displayNames") or pred.get("classes") or pred.get("classNames") or []
    )
    scores = (
        pred.get("confidences") or pred.get("scores") or pred.get("classScores") or []
    )
    if classes and scores:
        pairs = list(zip(classes, scores))
        label, conf = max(pairs, key=lambda x: float(x[1]))
        probs = {str(c): float(s) for c, s in zip(classes, scores)}
        return str(label), float(conf), probs

    if "structValue" in pred or "listValue" in pred:
        return parse_automl_classification(_unwrap_proto_value(pred))

    label = json.dumps(pred, ensure_ascii=False, sort_keys=True)
    return label, 1.0, {label: 1.0}


def _unwrap_proto_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "numberValue" in value:
        return value["numberValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "structValue" in value:
        fields = value["structValue"].get("fields", {})
        return {k: _unwrap_proto_value(v) for k, v in fields.items()}
    if "listValue" in value:
        return [_unwrap_proto_value(v) for v in value["listValue"].get("values", [])]
    return value


def _init_vertex(project: str, location: str) -> None:
    from edge_train.config import ensure_gcp_credentials

    ok, err = ensure_gcp_credentials()
    if not ok:
        raise RuntimeError(err)
    import google.cloud.aiplatform as aip

    aip.init(project=project, location=location)


class VertexAutoMLPredictor:
    """Run classification against a Vertex AutoML endpoint (tabular/image/video)."""

    modality: str = "table"

    def __init__(self, endpoint_name: str, *, project: str, location: str):
        if not is_vertex_endpoint(endpoint_name):
            raise ValueError(
                f"Not a Vertex endpoint resource name: {endpoint_name}\n"
                "Expected: projects/.../locations/.../endpoints/..."
            )
        _init_vertex(project, location)
        import google.cloud.aiplatform as aip

        self._endpoint = aip.Endpoint(endpoint_name)
        self._endpoint_name = endpoint_name

    def build_instance(self, payload: Any) -> dict[str, Any]:
        raise NotImplementedError

    def format_input(self, payload: Any) -> str:
        if isinstance(payload, dict):
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return str(payload)

    def _predict_instances(
        self, instances: list[dict[str, Any]]
    ) -> list[tuple[str, float, dict[str, float]]]:
        response = self._endpoint.predict(instances=instances)
        return [parse_automl_classification(p) for p in response.predictions]

    def predict(self, payload: Any) -> tuple[str, float]:
        label, conf, _ = self._predict_instances([self.build_instance(payload)])[0]
        return label, conf

    def predict_proba(self, payload: Any) -> dict[str, float]:
        _, _, probs = self._predict_instances([self.build_instance(payload)])[0]
        return probs

    def predict_batch(self, payloads: list[Any]) -> list[tuple[str, float]]:
        if not payloads:
            return []
        instances = [self.build_instance(p) for p in payloads]
        return [(label, conf) for label, conf, _ in self._predict_instances(instances)]


class VertexTabularPredictor(VertexAutoMLPredictor):
    modality = "table"

    def build_instance(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("Tabular predict expects a feature dict or CSV row")
        return dict(payload)


class VertexImagePredictor(VertexAutoMLPredictor):
    modality = "image"

    def build_instance(self, payload: str) -> dict[str, Any]:
        source = str(payload).strip()
        if source.startswith("gs://"):
            mime, _ = mimetypes.guess_type(source)
            return {"gcsUri": source, "mimeType": mime or "image/jpeg"}
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {source}")
        mime, _ = mimetypes.guess_type(str(path))
        content = base64.b64encode(path.read_bytes()).decode("utf-8")
        return {"content": content, "mimeType": mime or "image/jpeg"}


class VertexVideoPredictor(VertexAutoMLPredictor):
    modality = "video"

    def build_instance(self, payload: str) -> dict[str, Any]:
        source = str(payload).strip()
        if not source.startswith("gs://"):
            raise ValueError(
                "Video endpoint predict requires a GCS URI (gs://bucket/path.mp4). "
                "Upload the file to GCS first or use --gcs-uri."
            )
        mime, _ = mimetypes.guess_type(source)
        return {"gcsUri": source, "mimeType": mime or "video/mp4"}


class VertexTextPredictor:
    """Run text classification against a Gemini fine-tuned Vertex endpoint."""

    modality = "text"

    def __init__(self, endpoint_name: str, *, project: str, location: str):
        if not is_vertex_endpoint(endpoint_name):
            raise ValueError(
                f"Not a Vertex endpoint resource name: {endpoint_name}\n"
                "Expected: projects/.../locations/.../endpoints/..."
            )
        _init_vertex(project, location)
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=project, location=location)
        self._endpoint_name = endpoint_name
        self._model = GenerativeModel(endpoint_name)

    def format_input(self, text: str) -> str:
        return text

    def predict(self, text: str) -> tuple[str, float]:
        label, conf, probs = self._predict_raw(text)
        return label, conf

    def predict_proba(self, text: str) -> dict[str, float]:
        label, conf, probs = self._predict_raw(text)
        if probs:
            return probs
        return {label: conf}

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        return [self.predict(t) for t in texts]

    def _predict_raw(self, text: str) -> tuple[str, float, dict[str, float]]:
        response = self._model.generate_content(
            text,
            generation_config={"temperature": 0.0, "max_output_tokens": 64},
        )
        raw = (response.text or "").strip()
        label = raw.splitlines()[0].strip().strip('"').strip("'")
        probs: dict[str, float] = {}
        return label, 1.0 if label else 0.0, probs


def resolve_vertex_predictor(
    endpoint_name: str,
    *,
    project: str,
    location: str,
    modality: str = "text",
):
    """Return the correct Vertex predictor for endpoint modality."""
    mod = modality.lower().strip()
    if mod == "text":
        return VertexTextPredictor(endpoint_name, project=project, location=location)
    if mod == "table":
        return VertexTabularPredictor(endpoint_name, project=project, location=location)
    if mod == "image":
        return VertexImagePredictor(endpoint_name, project=project, location=location)
    if mod == "video":
        return VertexVideoPredictor(endpoint_name, project=project, location=location)
    raise ValueError(
        f"Unsupported endpoint modality: {modality}. "
        "Use text, table, image, or video."
    )
