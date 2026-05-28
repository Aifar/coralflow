"""Tests for cloud training wait guidance and scheduled polling."""

import time

import pytest

from edge_train.cloud.router import CloudTrainingMethod, CloudTrainingPlan
from edge_train.cloud.training_wait import (
    format_post_submit_guidance,
    poll_job_scheduled,
    resolve_wait_strategy,
    training_eta_minutes,
)


def _plan(method: CloudTrainingMethod) -> CloudTrainingPlan:
    return CloudTrainingPlan(method=method, modality="text", reason="test")


class TestTrainingEta:
    def test_gemini_eta(self):
        min_m, max_m, check_m = training_eta_minutes(
            _plan(CloudTrainingMethod.GEMINI_FINETUNE)
        )
        assert min_m < max_m
        assert min_m <= check_m <= max_m

    def test_post_submit_guidance_includes_duration(self):
        lines = format_post_submit_guidance(
            _plan(CloudTrainingMethod.AUTOML_TABULAR),
            "projects/p/locations/us/trainingPipelines/1",
        )
        text = "\n".join(lines)
        assert "Estimated duration" in text
        assert "Suggested first status check" in text
        assert "trainingPipelines/1" in text


class TestResolveWaitStrategy:
    def test_detach_flag(self):
        assert (
            resolve_wait_strategy(detach=True, poll_every=None, interactive=False)
            == "detach"
        )

    def test_poll_every_flag(self):
        assert (
            resolve_wait_strategy(detach=False, poll_every=30, interactive=False)
            == "scheduled"
        )

    def test_non_interactive_defaults_to_scheduled(self):
        assert (
            resolve_wait_strategy(detach=False, poll_every=None, interactive=False)
            == "scheduled"
        )

    def test_poll_every_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            resolve_wait_strategy(detach=False, poll_every=0, interactive=False)


class TestPollJobScheduled:
    def test_succeeds_on_first_check(self, mocker):
        mocker.patch(
            "edge_train.cloud.training_wait.get_cloud_job_status",
            return_value={"status": "succeeded", "model_path": "projects/p/models/1"},
        )
        result = poll_job_scheduled(
            "projects/p/jobs/1",
            interval_min=30,
            deadline=time.time() + 60,
            sleep_fn=lambda _: None,
        )
        assert result["model_path"] == "projects/p/models/1"

    def test_waits_between_checks(self, mocker):
        statuses = [
            {"status": "running", "model_path": ""},
            {"status": "succeeded", "model_path": "projects/p/models/2"},
        ]
        mocker.patch(
            "edge_train.cloud.training_wait.get_cloud_job_status",
            side_effect=statuses,
        )
        sleeps: list[float] = []
        result = poll_job_scheduled(
            "projects/p/jobs/1",
            interval_min=10,
            deadline=time.time() + 3600,
            sleep_fn=lambda sec: sleeps.append(sec),
        )
        assert result["model_path"] == "projects/p/models/2"
        assert len(sleeps) == 1
        assert sleeps[0] == 10 * 60

    def test_raises_on_failure(self, mocker):
        mocker.patch(
            "edge_train.cloud.training_wait.get_cloud_job_status",
            return_value={"status": "failed", "error": "boom"},
        )
        with pytest.raises(RuntimeError, match="boom"):
            poll_job_scheduled(
                "projects/p/jobs/1",
                interval_min=30,
                deadline=time.time() + 60,
                sleep_fn=lambda _: None,
            )
