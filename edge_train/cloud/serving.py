"""Deploy Vertex AI models to endpoints and run online inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    endpoint = model.deploy(
        deployed_model_display_name=display_name or f"coralflow-{model.display_name}",
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        sync=True,
    )
    deployed = endpoint.list_models()[0] if endpoint.list_models() else None
    return VertexDeployResult(
        model_path=model_name,
        endpoint_name=endpoint.resource_name,
        deployed_model_id=getattr(deployed, "id", "") if deployed else "",
    )


class VertexTextPredictor:
    """Run text classification against a Gemini fine-tuned Vertex endpoint."""

    def __init__(self, endpoint_name: str, *, project: str, location: str):
        if not is_vertex_endpoint(endpoint_name):
            raise ValueError(
                f"Not a Vertex endpoint resource name: {endpoint_name}\n"
                "Expected: projects/.../locations/.../endpoints/..."
            )
        from edge_train.config import ensure_gcp_credentials

        ok, err = ensure_gcp_credentials()
        if not ok:
            raise RuntimeError(err)

        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=project, location=location)
        self._endpoint_name = endpoint_name
        self._model = GenerativeModel(endpoint_name)

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
