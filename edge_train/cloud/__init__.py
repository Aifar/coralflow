"""Vertex AI cloud training — routed to Gemini SFT or AutoML services."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import google.cloud.aiplatform as aip

from edge_train.cloud.publisher_models import (
    PublisherModelInfo,
    describe_finetune_base_model,
    list_custom_models,
    list_publisher_models,
    resolve_finetune_base_model,
)
from edge_train.cloud.serving import (
    VertexTextPredictor,
    deploy_model_to_vertex,
    is_vertex_endpoint,
    is_vertex_resource,
)
from edge_train.cloud.router import (
    CloudTrainingMethod,
    CloudTrainingPlan,
    cloud_modality_supported,
    plan_cloud_training,
)

__all__ = [
    "Modality",
    "CloudTrainingMethod",
    "CloudTrainingPlan",
    "PublisherModelInfo",
    "TrainingResult",
    "cloud_modality_supported",
    "describe_finetune_base_model",
    "list_custom_models",
    "list_publisher_models",
    "plan_cloud_training",
    "poll_job",
    "resolve_finetune_base_model",
    "submit_automl_job",
    "deploy_model_to_vertex",
    "is_vertex_resource",
    "is_vertex_endpoint",
    "VertexTextPredictor",
    "_upload_to_gcs",
]


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    VIDEO = "video"


def _modality_from_plan(plan: CloudTrainingPlan) -> Modality:
    return Modality(plan.modality)


def _format_cloud_error(exc: Exception) -> str:
    """Surface API error details instead of opaque 500 messages."""
    from google.api_core import exceptions as gcp_exc

    if isinstance(exc, gcp_exc.GoogleAPIError):
        return str(exc.message if hasattr(exc, "message") else exc)
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, gcp_exc.GoogleAPIError):
        return str(cause.message if hasattr(cause, "message") else cause)
    return str(exc)


@dataclass
class TrainingResult:
    model_path: str = ""
    accuracy: float = 0.0
    job_name: str = ""


def submit_automl_job(
    project: str,
    location: str,
    dataset_path: str,
    modality: Modality | str | None = None,
    target_column: str | None = None,
    staging_bucket: str = "",
    finetune_model: str = "gemini-2.0-flash-001",
) -> str:
    """Submit a routed cloud training job. Returns job resource name for polling."""
    from edge_train.config import ensure_gcp_credentials

    ok, err = ensure_gcp_credentials()
    if not ok:
        raise RuntimeError(err)

    modality_str = modality.value if isinstance(modality, Modality) else modality
    plan = plan_cloud_training(dataset_path, modality_str)

    try:
        if plan.method == CloudTrainingMethod.GEMINI_FINETUNE:
            from edge_train.cloud.finetune import submit_finetune_job

            return submit_finetune_job(
                project=project,
                location=location,
                dataset_path=dataset_path,
                staging_bucket=staging_bucket,
                target_column=target_column,
                base_model=finetune_model,
            )

        automl_modality = _modality_from_plan(plan)
        aip.init(project=project, location=location, staging_bucket=staging_bucket)
        dataset = _create_or_get_dataset(
            dataset_path, automl_modality, project, location, staging_bucket
        )
        return _launch_training_pipeline(dataset, automl_modality, target_column)
    except Exception as exc:
        raise RuntimeError(_format_cloud_error(exc)) from exc


def poll_job(job_name: str, deadline: float) -> dict[str, Any]:
    """Poll a cloud training job until completion or timeout."""
    from edge_train.cloud.finetune import is_finetune_job, poll_finetune_job

    if is_finetune_job(job_name):
        return poll_finetune_job(job_name, deadline)

    if "/trainingPipelines/" in job_name:
        return _poll_automl_training_pipeline(job_name, deadline)

    while time.time() < deadline:
        job = aip.PipelineJob.get(resource_name=job_name)
        state = job.state

        if state == "PIPELINE_STATE_SUCCEEDED":
            return _extract_result(job)
        if state in ("PIPELINE_STATE_FAILED", "PIPELINE_STATE_CANCELLED"):
            raise RuntimeError(f"Training job failed: {state}")

        time.sleep(30)

    raise TimeoutError(f"Training job did not complete within deadline")


def _poll_automl_training_pipeline(job_name: str, deadline: float) -> dict[str, Any]:
    """Poll an AutoML training pipeline (tabular/image/video)."""
    from google.cloud.aiplatform.training_jobs import _TrainingJob

    succeeded = {"PIPELINE_STATE_SUCCEEDED"}
    failed = {"PIPELINE_STATE_FAILED", "PIPELINE_STATE_CANCELLED"}

    while time.time() < deadline:
        job = _TrainingJob._get_and_return_subclass(job_name)
        state = job.state
        state_name = state.name if hasattr(state, "name") else str(state)

        if state_name in succeeded:
            model_path = ""
            model_to_upload = getattr(job._gca_resource, "model_to_upload", None)
            if model_to_upload and getattr(model_to_upload, "name", None):
                model_path = model_to_upload.name
            return {
                "job_name": job_name,
                "model_path": model_path,
                "accuracy": 0.0,
            }
        if state_name in failed:
            raise RuntimeError(f"Training job failed: {state_name}")

        time.sleep(30)

    raise TimeoutError(f"Training job did not complete within deadline")


def _create_or_get_dataset(
    path: str,
    modality: Modality,
    project: str,
    location: str,
    staging_bucket: str,
) -> aip.ImageDataset | aip.TabularDataset | aip.VideoDataset:
    """Create a Vertex AI dataset and import data from the given path."""
    mapping = {
        Modality.IMAGE: ("image_classification", aip.ImageDataset),
        Modality.TABLE: ("tabular", aip.TabularDataset),
        Modality.VIDEO: ("video_classification", aip.VideoDataset),
    }
    schema_type, dataset_cls = mapping[modality]
    import_uri = (
        path
        if path.startswith("gs://")
        else _upload_to_gcs(path, project, staging_bucket)
    )

    create_kwargs: dict[str, Any] = {
        "display_name": f"edge-train-{int(time.time())}",
        "gcs_source": import_uri,
        "project": project,
        "location": location,
    }
    # TabularDataset.create() does not accept import_schema_uri (SDK 1.152+).
    if modality != Modality.TABLE:
        create_kwargs["import_schema_uri"] = (
            f"gs://google-cloud-aiplatform/schema/dataset/metadata/{schema_type}_schema.yaml"
        )

    dataset = dataset_cls.create(**create_kwargs)
    return dataset


def _launch_training_pipeline(
    dataset: aip.ImageDataset | aip.TabularDataset | aip.VideoDataset,
    modality: Modality,
    target_column: str | None,
) -> str:
    """Launch an AutoML training pipeline on the given dataset."""
    display_name = f"edge-train-pipeline-{int(time.time())}"

    if modality == Modality.VIDEO:
        job = aip.AutoMLVideoTrainingJob(
            display_name=display_name,
            prediction_type="classification",
        )
        job.run(dataset=dataset, sync=False)
        return job.resource_name

    if modality == Modality.IMAGE:
        job = aip.AutoMLImageTrainingJob(
            display_name=display_name,
            prediction_type="classification",
        )
        job.run(dataset=dataset, sync=False)
        return job.resource_name

    job = aip.AutoMLTabularTrainingJob(
        display_name=display_name,
        optimization_prediction_type="classification",
    )
    job.run(
        dataset=dataset,
        target_column=target_column or "label",
        sync=False,
    )
    return job.resource_name


def _extract_result(job: aip.PipelineJob) -> dict[str, Any]:
    gca_resource = getattr(job, "_gca_resource", None)
    model_path = (
        gca_resource.endpoint.model
        if gca_resource and hasattr(gca_resource, "endpoint")
        else ""
    )
    return {
        "job_name": job.resource_name,
        "model_path": model_path,
        "accuracy": 0.0,
    }


def _upload_to_gcs(
    local_path: str,
    project: str,
    staging_bucket: str,
    blob_prefix: str = "datasets",
) -> str:
    """Upload a local file to the configured GCS staging bucket."""
    from google.cloud import storage

    from edge_train.config import gcs_bucket_name

    if not staging_bucket:
        raise ValueError(
            "GCP_STAGING_BUCKET is not set. Add gs://your-bucket to .env or run "
            "gsutil mb gs://your-bucket"
        )

    bucket_name = gcs_bucket_name(staging_bucket)
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob_name = f"{blob_prefix}/{local_path.split('/')[-1]}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"
