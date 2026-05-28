"""Tests for deployment registry and Phoenix hints."""

from edge_train.deployments import (
    DeploymentRecord,
    DeploymentRegistry,
    format_phoenix_monitoring_hint,
)


class TestDeploymentRegistry:
    def test_save_and_load(self, tmp_path, monkeypatch):
        path = tmp_path / "deployments.json"
        monkeypatch.setenv("CORALFLOW_DEPLOYMENTS_PATH", str(path))

        reg = DeploymentRegistry()
        reg.add(
            DeploymentRecord(
                model_path="projects/p/locations/us/models/1",
                target="vertex",
                endpoint_name="projects/p/locations/us/endpoints/9",
                modality="text",
            )
        )

        loaded = DeploymentRegistry.load()
        assert loaded.records[0].endpoint_name.endswith("/endpoints/9")

    def test_find_by_endpoint(self, tmp_path, monkeypatch):
        path = tmp_path / "deployments.json"
        monkeypatch.setenv("CORALFLOW_DEPLOYMENTS_PATH", str(path))

        reg = DeploymentRegistry()
        reg.add(
            DeploymentRecord(
                model_path="projects/p/locations/us/models/1",
                target="vertex",
                endpoint_name="projects/p/locations/us/endpoints/9",
                modality="image",
            )
        )

        found = DeploymentRegistry.load().find_by_endpoint(
            "projects/p/locations/us/endpoints/9"
        )
        assert found is not None
        assert found.modality == "image"

    def test_phoenix_hint_vertex(self):
        hint = format_phoenix_monitoring_hint(
            endpoint="projects/p/locations/us/endpoints/9"
        )
        assert "predict --endpoint" in hint or "simulate --endpoint" in hint
        assert "monitor --dashboard" in hint
