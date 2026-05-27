"""Tests for publisher model listing helpers."""

from edge_train.cloud.publisher_models import (
    is_likely_finetune_base_model,
    parse_model_garden_entry,
    resolve_finetune_base_model,
)


class TestParseModelGardenEntry:
    def test_parses_qualified_name(self):
        info = parse_model_garden_entry(
            "google/gemini-2.0-flash-001@default", "us-central1"
        )
        assert info.model_id == "gemini-2.0-flash-001"
        assert info.publisher == "google"
        assert info.version == "default"
        assert info.tuning_source_model == "gemini-2.0-flash-001"
        assert info.resource_name.endswith("/models/gemini-2.0-flash-001")

    def test_parses_without_version(self):
        info = parse_model_garden_entry("google/gemini-2.5-flash", "us-central1")
        assert info.model_id == "gemini-2.5-flash"
        assert info.version == "default"


class TestFinetuneHeuristic:
    def test_excludes_embedding(self):
        assert not is_likely_finetune_base_model("gemini-embedding-001")

    def test_includes_flash(self):
        assert is_likely_finetune_base_model("gemini-2.0-flash-001")


class TestDescribeFinetuneBaseModel:
    def test_plain_lines_include_model_id(self):
        from edge_train.cloud.publisher_models import describe_finetune_base_model

        lines = describe_finetune_base_model("gemini-2.0-flash-001", "us-central1")
        text = "\n".join(lines)
        assert "gemini-2.0-flash-001" in text
        assert "GCP_FINETUNE_MODEL" in text
        assert "coralflow models list" in text

    def test_markdown_lines(self):
        from edge_train.cloud.publisher_models import describe_finetune_base_model

        lines = describe_finetune_base_model(
            "gemini-2.5-flash", "us-central1", markdown=True
        )
        assert any("**Fine-tune base model:**" in line for line in lines)


class TestResolveFinetuneBaseModel:
    def test_short_id_passthrough_without_validation(self):
        assert (
            resolve_finetune_base_model("gemini-2.0-flash-001", "p", validate=False)
            == "gemini-2.0-flash-001"
        )

    def test_full_resource_path(self):
        path = (
            "projects/google/locations/us-central1/publishers/google/"
            "models/gemini-2.0-flash-001"
        )
        assert resolve_finetune_base_model(path, "p", validate=False) == (
            "gemini-2.0-flash-001"
        )

    def test_qualified_name(self):
        assert (
            resolve_finetune_base_model(
                "google/gemini-2.5-flash@default", "p", validate=False
            )
            == "gemini-2.5-flash"
        )

    def test_validate_against_model_garden(self, mocker):
        from edge_train.cloud import publisher_models as pm

        mocker.patch.object(
            pm,
            "list_publisher_models",
            return_value=[
                parse_model_garden_entry(
                    "google/gemini-2.0-flash-001@default", "us-central1"
                )
            ],
        )
        assert resolve_finetune_base_model("gemini-2.0-flash-001", "p") == (
            "gemini-2.0-flash-001"
        )

    def test_unknown_model_raises(self, mocker):
        from edge_train.cloud import publisher_models as pm

        mocker.patch.object(
            pm,
            "list_publisher_models",
            return_value=[
                parse_model_garden_entry(
                    "google/gemini-2.0-flash-001@default", "us-central1"
                )
            ],
        )
        try:
            resolve_finetune_base_model("not-a-real-model", "p")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "Unknown fine-tune base model" in str(exc)
