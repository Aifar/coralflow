"""Vertex AI AutoML integration."""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import google.cloud.aiplatform as aip


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


@dataclass
class TrainingResult:
    model_path: str = ""
    accuracy: float = 0.0
    job_name: str = ""


def submit_automl_job(
    project: str,
    location: str,
    dataset_path: str,
    modality: Modality,
    target_column: str | None = None,
    staging_bucket: str = "",
) -> str:
    """Submit a Vertex AI AutoML training job.

    Returns the job resource name for polling.
    """
    from edge_train.config import ensure_gcp_credentials

    ok, err = ensure_gcp_credentials()
    if not ok:
        raise RuntimeError(err)

    aip.init(project=project, location=location, staging_bucket=staging_bucket)

    dataset = _create_or_get_dataset(dataset_path, modality, project, location)
    job_name = _launch_training_pipeline(dataset, modality, target_column)
    return job_name


def poll_job(job_name: str, deadline: float) -> dict[str, Any]:
    """Poll a training job until completion or timeout."""
    while time.time() < deadline:
        job = aip.PipelineJob.get(resource_name=job_name)
        state = job.state

        if state == "PIPELINE_STATE_SUCCEEDED":
            return _extract_result(job)
        if state in ("PIPELINE_STATE_FAILED", "PIPELINE_STATE_CANCELLED"):
            raise RuntimeError(f"Training job failed: {state}")

        time.sleep(30)

    raise TimeoutError(f"Training job did not complete within deadline")


def _create_or_get_dataset(
    path: str,
    modality: Modality,
    project: str,
    location: str,
) -> aip.TextDataset | aip.ImageDataset | aip.TabularDataset:
    """Create a Vertex AI dataset and import data from the given path."""
    mapping = {
        Modality.TEXT: ("text_classification", aip.TextDataset),
        Modality.IMAGE: ("image_classification", aip.ImageDataset),
        Modality.TABLE: ("tabular", aip.TabularDataset),
    }
    schema_type, dataset_cls = mapping[modality]
    import_uri = (
        path if path.startswith("gs://") else _upload_to_gcs(path, project, location)
    )

    dataset = dataset_cls.create(
        display_name=f"edge-train-{int(time.time())}",
        gcs_source=import_uri,
        import_schema_uri=f"gs://google-cloud-aiplatform/schema/dataset/metadata/{schema_type}_schema.yaml",
        project=project,
        location=location,
    )
    return dataset


def _launch_training_pipeline(
    dataset: aip.TextDataset | aip.ImageDataset | aip.TabularDataset,
    modality: Modality,
    target_column: str | None,
) -> str:
    """Launch an AutoML training pipeline on the given dataset."""
    prediction_type = {
        Modality.TEXT: "text_classification",
        Modality.IMAGE: "image_classification",
        Modality.TABLE: "tabular",
    }[modality]

    training_cls = {
        Modality.TEXT: aip.AutoMLTextTrainingJob,
        Modality.IMAGE: aip.AutoMLImageTrainingJob,
        Modality.TABLE: aip.AutoMLTabularTrainingJob,
    }[modality]

    job = training_cls(
        display_name=f"edge-train-pipeline-{int(time.time())}",
        prediction_type=prediction_type,
    )
    job.run(dataset=dataset, target_column=target_column or "label")
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


def _upload_to_gcs(local_path: str, project: str, location: str) -> str:
    """Upload local dataset to GCS staging bucket for Vertex AI."""
    from google.cloud import storage

    bucket_name = f"{project}-edge-train-staging"
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob_name = f"datasets/{local_path.split('/')[-1]}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"
