"""Query Vertex AI cloud training job status."""

from __future__ import annotations

from typing import Any

_RUNNING = {
    "PIPELINE_STATE_PENDING",
    "PIPELINE_STATE_RUNNING",
    "PIPELINE_STATE_QUEUED",
}
_SUCCEEDED = {"PIPELINE_STATE_SUCCEEDED"}
_FAILED = {"PIPELINE_STATE_FAILED", "PIPELINE_STATE_CANCELLED"}


def _state_name(state: Any) -> str:
    return state.name if hasattr(state, "name") else str(state)


def get_cloud_job_status(job_name: str) -> dict[str, str]:
    """Return ``status`` (running|succeeded|failed|unknown), ``model_path``, ``error``."""
    from edge_train.cloud.finetune import is_finetune_job

    if is_finetune_job(job_name):
        return _finetune_job_status(job_name)
    if "/trainingPipelines/" in job_name:
        return _automl_pipeline_status(job_name)
    return {"status": "unknown", "model_path": "", "error": ""}


def _finetune_job_status(job_name: str) -> dict[str, str]:
    from vertexai.tuning import sft

    job = sft.SupervisedTuningJob(job_name)
    job.refresh()
    if job.has_ended:
        if job.has_succeeded:
            return {
                "status": "succeeded",
                "model_path": job.tuned_model_name
                or job.tuned_model_endpoint_name
                or "",
                "error": "",
            }
        return {
            "status": "failed",
            "model_path": "",
            "error": str(job.error or job.state),
        }
    return {"status": "running", "model_path": "", "error": ""}


def _automl_pipeline_status(job_name: str) -> dict[str, str]:
    from google.cloud.aiplatform.training_jobs import _TrainingJob

    job = _TrainingJob._get_and_return_subclass(job_name)
    state_name = _state_name(job.state)
    if state_name in _SUCCEEDED:
        model_path = ""
        model_to_upload = getattr(job._gca_resource, "model_to_upload", None)
        if model_to_upload and getattr(model_to_upload, "name", None):
            model_path = model_to_upload.name
        return {"status": "succeeded", "model_path": model_path, "error": ""}
    if state_name in _FAILED:
        err = getattr(job, "error", None)
        return {
            "status": "failed",
            "model_path": "",
            "error": str(err or state_name),
        }
    if state_name in _RUNNING or state_name.startswith("PIPELINE_STATE_"):
        return {"status": "running", "model_path": "", "error": ""}
    return {"status": "unknown", "model_path": "", "error": state_name}
