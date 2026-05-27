"""Tests for fine-tune base model prompts in agent assess_resources."""


class TestAssessResourcesFinetunePrompt:
    def test_text_cloud_option_shows_base_model(self, monkeypatch):
        monkeypatch.setenv("GCP_FINETUNE_MODEL", "gemini-2.5-flash")
        from edge_train.agent.tools import _exec_assess_resources

        result = _exec_assess_resources({"dataset_path": "urgent"})
        assert "Fine-tune base model" in result
        assert "gemini-2.5-flash" in result
        assert "coralflow models list" in result
