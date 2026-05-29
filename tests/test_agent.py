"""Tests for coralflow agent module."""

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner


class TestAgentState:
    def test_default_state(self):
        from edge_train.agent import AgentState

        state = AgentState()
        assert state.dataset_path == ""
        assert state.model_path is None
        assert state.task_type == ""
        assert state.deployment_target is None
        assert state.created_at != ""

    def test_save_and_load(self, tmp_path):
        from edge_train.agent import AgentState

        state_file = tmp_path / "state.json"
        state = AgentState(
            dataset_path="/data/test.csv",
            model_path="/models/test",
            task_type="text-classification",
            deployment_target="10.0.0.1",
            last_step="train",
            conversation_summary="User trained a model.",
        )
        state.save(str(state_file))
        assert state_file.exists()

        loaded = AgentState.load(str(state_file))
        assert loaded.dataset_path == "/data/test.csv"
        assert loaded.model_path == "/models/test"
        assert loaded.task_type == "text-classification"
        assert loaded.deployment_target == "10.0.0.1"
        assert loaded.last_step == "train"
        assert loaded.conversation_summary == "User trained a model."

    def test_load_nonexistent_returns_default(self, tmp_path):
        from edge_train.agent import AgentState

        state = AgentState.load(str(tmp_path / "nonexistent.json"))
        assert state.dataset_path == ""

    def test_roundtrip_preserves_fields(self, tmp_path):
        from edge_train.agent import AgentState

        state_file = tmp_path / "roundtrip.json"
        state = AgentState(
            dataset_path="builtin:urgent",
            model_path="./model_output/urgent",
            task_type="text",
            last_step="deploy",
        )
        state.save(str(state_file))

        loaded = AgentState.load(str(state_file))
        assert loaded.dataset_path == state.dataset_path
        assert loaded.model_path == state.model_path
        assert loaded.task_type == state.task_type
        assert loaded.last_step == state.last_step


class TestDatasetScanner:
    def test_finds_csv_files(self, tmp_path):
        from edge_train.agent import DatasetScanner

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_file = data_dir / "test.csv"
        csv_file.write_text("text,label\nhello,greeting\nworld,question\n")

        datasets = DatasetScanner.scan([str(data_dir)])
        assert any(d["name"] == "test" for d in datasets)

    def test_includes_builtin(self):
        from edge_train.agent import DatasetScanner

        datasets = DatasetScanner.scan(["/nonexistent/path"])
        builtin_names = {d["name"] for d in datasets if d["source"] == "built-in"}
        assert "urgent" in builtin_names
        assert "expense" in builtin_names

    def test_excludes_non_csv(self, tmp_path):
        from edge_train.agent import DatasetScanner

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "notes.txt").write_text("hello")

        datasets = DatasetScanner.scan([str(data_dir)])
        names = {d["name"] for d in datasets}
        assert "notes" not in names

    def test_returns_modality_and_rows(self, tmp_path):
        from edge_train.agent import DatasetScanner

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_file = data_dir / "sample.csv"
        csv_file.write_text("text,label\nhello,greeting\nworld,question\n")

        datasets = DatasetScanner.scan([str(data_dir)])
        match = [d for d in datasets if d["name"] == "sample"]
        assert len(match) == 1
        assert match[0]["modality"] in ("text", "unknown")
        assert match[0]["rows"] == 2


class TestRecommender:
    def test_recommends_local_for_small_text(self):
        from edge_train.agent import Recommender

        rec = Recommender.recommend(
            {"rows": 500, "classes": ["A", "B"], "modality": "text"}
        )
        assert rec["method"] == "local"
        assert rec["epochs"] is not None

    def test_recommends_cloud_for_large_table_dataset(self):
        from edge_train.agent import Recommender

        rec = Recommender.recommend(
            {"rows": 50000, "classes": ["A", "B"], "modality": "table"}
        )
        assert rec["method"] == "cloud"

    def test_large_text_recommends_cloud_finetune(self):
        from edge_train.agent import Recommender

        rec = Recommender.recommend(
            {"rows": 50000, "classes": ["A", "B"], "modality": "text"}
        )
        assert rec["method"] == "cloud"
        assert "fine-tuning" in rec["reason"].lower()

    def test_recommends_cloud_for_image(self):
        from edge_train.agent import Recommender

        rec = Recommender.recommend(
            {"rows": 100, "classes": ["A", "B"], "modality": "image"}
        )
        assert rec["method"] == "cloud"

    def test_recommends_cloud_for_table(self):
        from edge_train.agent import Recommender

        rec = Recommender.recommend(
            {"rows": 200, "classes": ["X"], "modality": "table"}
        )
        assert rec["method"] == "cloud"

    def test_small_dataset_gets_more_epochs(self):
        from edge_train.agent import Recommender

        rec_small = Recommender.recommend(
            {"rows": 100, "classes": ["A"], "modality": "text"}
        )
        rec_large = Recommender.recommend(
            {"rows": 3000, "classes": ["A"], "modality": "text"}
        )
        assert rec_small["epochs"] >= rec_large["epochs"]


class TestGeminiLLMCompat:
    def test_is_gemini_compatible(self):
        from edge_train.agent.llm import LLMConfig, is_gemini_compatible

        assert is_gemini_compatible(
            LLMConfig(
                endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
                model="gemini-3.1-pro-preview",
            )
        )
        assert not is_gemini_compatible(
            LLMConfig(endpoint="https://api.openai.com/v1", model="gpt-4o")
        )

    def test_ensure_gemini_adds_skip_signature(self):
        from edge_train.agent.llm import (
            GEMINI_SKIP_THOUGHT_SIGNATURE,
            LLMConfig,
            ensure_gemini_tool_signatures,
        )

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "scan_datasets", "arguments": "{}"},
                    }
                ],
            }
        ]
        config = LLMConfig(
            endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-3.1-pro-preview",
        )
        out = ensure_gemini_tool_signatures(messages, config)
        sig = out[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"]
        assert sig == GEMINI_SKIP_THOUGHT_SIGNATURE

    def test_ensure_gemini_preserves_existing_signature(self):
        from edge_train.agent.llm import LLMConfig, ensure_gemini_tool_signatures

        existing = {"google": {"thought_signature": "real-signature"}}
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "scan_datasets", "arguments": "{}"},
                        "extra_content": existing,
                    }
                ],
            }
        ]
        config = LLMConfig(
            endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-3.1-pro-preview",
        )
        out = ensure_gemini_tool_signatures(messages, config)
        assert out[0]["tool_calls"][0]["extra_content"] == existing

    def test_chat_preserves_tool_call_extra_content(self, monkeypatch):
        from edge_train.agent.llm import LLMClient, LLMConfig

        captured: dict = {}

        class FakeResp:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "recommend_datasets",
                                            "arguments": "{}",
                                        },
                                        "extra_content": {
                                            "google": {"thought_signature": "sig-123"}
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }

        def fake_post(url, headers, json, timeout):
            captured["messages"] = json["messages"]
            return FakeResp()

        monkeypatch.setattr("edge_train.agent.llm.requests.post", fake_post)
        client = LLMClient(
            LLMConfig(
                api_key="test",
                endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
                model="gemini-3.1-pro-preview",
            )
        )
        history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "recommend_datasets",
                            "arguments": "{}",
                        },
                        "extra_content": {"google": {"thought_signature": "sig-123"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "ok"},
        ]
        resp = client.chat(history, tools=[{"type": "function", "function": {}}])
        assert resp.tool_calls
        assert resp.tool_calls[0].extra_content == {
            "google": {"thought_signature": "sig-123"}
        }
        assert (
            captured["messages"][0]["tool_calls"][0]["extra_content"]["google"][
                "thought_signature"
            ]
            == "sig-123"
        )

    def test_build_assistant_tool_calls_message(self):
        from edge_train.agent.llm import (
            LLMResponse,
            ToolCall,
            build_assistant_tool_calls_message,
        )

        msg = build_assistant_tool_calls_message(
            LLMResponse(
                content="",
                reasoning_content="think",
                tool_calls=[
                    ToolCall(
                        id="call_x",
                        name="scan_datasets",
                        arguments={},
                        extra_content={"google": {"thought_signature": "sig-123"}},
                    )
                ],
            )
        )
        assert msg["reasoning_content"] == "think"
        assert msg["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == (
            "sig-123"
        )


class TestLLMConfig:
    def test_from_env_defaults(self, clear_env):
        from edge_train.agent.llm import LLMConfig

        config = LLMConfig.from_env()
        assert config.endpoint == "https://api.openai.com/v1"
        assert config.api_key == ""
        assert config.model == "gpt-4o"
        assert not config.is_valid()

    def test_from_env_with_key(self, monkeypatch):
        from edge_train.agent.llm import LLMConfig

        monkeypatch.setenv("CORALFLOW_LLM_API_KEY", "sk-test")
        config = LLMConfig.from_env()
        assert config.is_valid()

    def test_from_env_custom_values(self, monkeypatch):
        from edge_train.agent.llm import LLMConfig

        monkeypatch.setenv("CORALFLOW_LLM_API_KEY", "sk-custom")
        monkeypatch.setenv("CORALFLOW_LLM_ENDPOINT", "https://custom.api.com/v1")
        monkeypatch.setenv("CORALFLOW_LLM_MODEL", "gpt-4o-mini")

        config = LLMConfig.from_env()
        assert config.endpoint == "https://custom.api.com/v1"
        assert config.model == "gpt-4o-mini"
        assert config.is_valid()

    def test_format_llm_setup_hint_includes_cli_flags(self):
        from edge_train.agent.llm import LLMConfig, format_llm_setup_hint

        hint = format_llm_setup_hint(LLMConfig(model="gpt-4o-mini"))
        assert "CORALFLOW_LLM_API_KEY" in hint
        assert "coralflow agent --api-key" in hint
        assert "gpt-4o-mini" in hint

    def test_is_llm_error_response(self):
        from edge_train.agent.llm import LLMResponse, is_llm_error_response

        assert is_llm_error_response(LLMResponse(content="Error: timeout"))
        assert not is_llm_error_response(LLMResponse(content="Hello"))
        assert not is_llm_error_response(LLMResponse(content=None))

    def test_verify_connection_success(self, monkeypatch):
        from edge_train.agent.llm import LLMClient, LLMConfig, LLMResponse

        client = LLMClient(LLMConfig(api_key="sk-test"))
        monkeypatch.setattr(
            client,
            "chat",
            lambda messages, tools=None: LLMResponse(content="pong"),
        )
        ok, err = client.verify_connection()
        assert ok
        assert err == ""

    def test_verify_connection_failure(self, monkeypatch):
        from edge_train.agent.llm import LLMClient, LLMConfig, LLMResponse

        client = LLMClient(LLMConfig(api_key="sk-test"))
        monkeypatch.setattr(
            client,
            "chat",
            lambda messages, tools=None: LLMResponse(
                content="Error: LLM request failed: 401"
            ),
        )
        ok, err = client.verify_connection()
        assert not ok
        assert "401" in err

    def test_prompt_llm_config_interactive(self):
        from edge_train.agent.llm import LLMConfig, prompt_llm_config_interactive

        answers = iter(["sk-new", "https://custom.example/v1", "gpt-4o-mini"])
        config = prompt_llm_config_interactive(
            LLMConfig(),
            lambda label, *, default="": next(answers),
        )
        assert config.api_key == "sk-new"
        assert config.endpoint == "https://custom.example/v1"
        assert config.model == "gpt-4o-mini"

    def test_prompt_llm_config_skip_clears_key(self):
        from edge_train.agent.llm import LLMConfig, prompt_llm_config_interactive

        config = prompt_llm_config_interactive(
            LLMConfig(api_key="sk-old", model="gpt-4o"),
            lambda label, *, default="": "skip",
        )
        assert config.api_key == ""
        assert config.model == "gpt-4o"

    def test_ensure_llm_client_non_tty_exits_without_key(self, monkeypatch, clear_env):
        from edge_train.agent.llm import LLMConfig, ensure_llm_client

        monkeypatch.setattr("edge_train.agent.llm.sys.stdout.isatty", lambda: False)
        monkeypatch.setattr("edge_train.agent.llm.sys.stdin.isatty", lambda: False)

        with pytest.raises(SystemExit) as exc:
            ensure_llm_client(
                LLMConfig(),
                lambda label, *, default="": "",
                is_tty=False,
            )
        assert exc.value.code == 1

    def test_ensure_llm_client_persists_valid_env_config(
        self, monkeypatch, clear_env, tmp_path
    ):
        from edge_train import config as cfg
        from edge_train.agent.llm import LLMConfig, ensure_llm_client

        repo = tmp_path / "repo"
        repo.mkdir()
        monkeypatch.setattr(cfg, "_PKG_ROOT", repo)
        monkeypatch.setenv("CORALFLOW_LLM_API_KEY", "sk-from-env")
        monkeypatch.setenv("CORALFLOW_LLM_ENDPOINT", "https://api.example/v1")
        monkeypatch.setenv("CORALFLOW_LLM_MODEL", "gpt-test")

        monkeypatch.setattr(
            "edge_train.agent.llm.LLMClient.verify_connection",
            lambda self: (True, ""),
        )

        llm = ensure_llm_client(
            LLMConfig.from_env(),
            lambda label, *, default="": "",
            is_tty=False,
        )
        assert llm.config.api_key == "sk-from-env"
        text = (repo / ".env").read_text(encoding="utf-8")
        assert "CORALFLOW_LLM_API_KEY=sk-from-env" in text
        assert "CORALFLOW_LLM_ENDPOINT=https://api.example/v1" in text
        assert "CORALFLOW_LLM_MODEL=gpt-test" in text


class TestManualToolInput:
    def test_collect_tool_arguments_required_only(self):
        from edge_train.agent.tools import collect_tool_arguments_interactive

        answers = iter(["data/urgent.csv"])
        args = collect_tool_arguments_interactive(
            "analyze_dataset",
            lambda label: next(answers, ""),
        )
        assert args == {"path": "data/urgent.csv"}

    def test_collect_tool_arguments_skips_optional(self):
        from edge_train.agent.tools import collect_tool_arguments_interactive

        answers = iter(["data/urgent.csv", "", "", ""])
        args = collect_tool_arguments_interactive(
            "train_model",
            lambda label: next(answers, ""),
        )
        assert args == {"dataset_path": "data/urgent.csv"}


class TestCLIAgent:
    def test_help(self):
        from edge_train.cli.agent import agent
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(agent, ["--help"])
        assert result.exit_code == 0
        assert "LLM-powered" in result.output
        assert "--model" in result.output
        assert "--endpoint" in result.output
        assert "--api-key" in result.output
        assert "--resume" in result.output

    def test_exits_without_api_key(self, monkeypatch, clear_env):
        from edge_train.cli.agent import agent
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(agent, [])
        assert result.exit_code == 1
        assert "LLM API key is not set" in result.output

    def test_exits_on_connection_failure(self, monkeypatch, clear_env):
        from edge_train.cli.agent import agent
        from click.testing import CliRunner

        monkeypatch.setenv("CORALFLOW_LLM_API_KEY", "sk-test")
        monkeypatch.setattr(
            "edge_train.agent.llm.LLMClient.verify_connection",
            lambda self: (False, "Error: LLM request failed: unauthorized"),
        )

        runner = CliRunner()
        result = runner.invoke(agent, [])
        assert result.exit_code == 1
        assert "LLM connection failed" in result.output

    def test_api_key_cli_flag_passes_validation(self, monkeypatch, clear_env):
        from edge_train.cli.agent import agent
        from click.testing import CliRunner

        monkeypatch.setattr(
            "edge_train.agent.llm.LLMClient.verify_connection",
            lambda self: (True, ""),
        )
        monkeypatch.setattr(
            "edge_train.agent.google_env.ensure_google_env_at_startup",
            lambda *args, **kwargs: False,
        )
        monkeypatch.setattr(
            "edge_train.agent.loop.run_agent_loop",
            lambda *args, **kwargs: None,
        )

        runner = CliRunner()
        result = runner.invoke(agent, ["--api-key", "sk-from-cli"])
        assert result.exit_code == 0

    def test_resume_flag_accepted(self):
        from edge_train.cli.agent import agent
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(agent, ["--resume", "--help"])
        assert result.exit_code == 0
        assert "--resume" in result.output


class TestToolSchemas:
    def test_all_tools_have_name_and_description(self):
        from edge_train.agent.tools import TOOLS

        for tool in TOOLS:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_required_tools_exist(self):
        from edge_train.agent.tools import TOOLS

        names = {t["function"]["name"] for t in TOOLS}
        required = {
            "scan_datasets",
            "analyze_dataset",
            "assess_resources",
            "validate_dataset",
            "recommend_datasets",
            "train_model",
            "validate_model",
            "deploy_model",
            "predict",
            "check_monitoring",
            "check_retrain",
            "label_predictions",
            "run_shell",
            "get_status",
        }
        assert required <= names

    def test_tool_schemas_are_valid_json(self):
        from edge_train.agent.tools import TOOLS

        # Must be serializable
        serialized = json.dumps(TOOLS)
        assert len(serialized) > 0
        parsed = json.loads(serialized)
        assert len(parsed) == len(TOOLS)


class TestValidateDataset:
    def test_missing_file(self):
        from edge_train.agent.validate_dataset import validate_dataset

        result = validate_dataset("/nonexistent/dataset.csv")
        assert "not found" in result

    def test_builtin_skipped(self):
        from edge_train.agent.validate_dataset import validate_dataset

        # Should handle builtin gracefully (it passes through to the file reader check)
        result = validate_dataset("builtin:urgent")
        assert "not found" in result  # builtin: is not a real path


class TestRecommendDatasets:
    def test_empty_task_returns_prompt(self):
        from edge_train.agent.recommend import recommend_datasets

        result = recommend_datasets("")
        assert "task description" in result.lower()

    def test_fallback_without_llm(self):
        from edge_train.agent.recommend import recommend_datasets

        result = recommend_datasets("spam detection")
        assert "HuggingFace" in result or "Kaggle" in result or "UCI" in result


class TestResourceAssessment:
    def test_returns_report_with_cpu_and_ram(self):
        from edge_train.agent.resources import assess_resources

        result = assess_resources()
        assert "CPU:" in result
        assert "RAM:" in result
        assert "Disk:" in result
        assert "TensorFlow:" in result
        assert "Verdict:" in result

    def test_nonexistent_dataset_is_skipped(self):
        from edge_train.agent.resources import assess_resources

        result = assess_resources("/nonexistent/dataset.csv")
        assert "Verdict:" in result

    def test_verdict_is_either_viable_or_insufficient(self):
        from edge_train.agent.resources import assess_resources

        result = assess_resources()
        assert ("viable" in result) or ("insufficient" in result)


class TestScanModels:
    def test_returns_list(self):
        from edge_train.agent import scan_models

        models = scan_models(["/nonexistent/path"])
        assert isinstance(models, list)

    def test_finds_saved_models(self, tmp_path):
        from edge_train.agent import scan_models

        model_dir = tmp_path / "models" / "test_model"
        model_dir.mkdir(parents=True)
        (model_dir / "saved_model.pb").write_text("fake pb")
        (model_dir / "model_meta.json").write_text(json.dumps({"classes": ["A", "B"]}))

        models = scan_models([str(tmp_path / "models")])
        assert len(models) == 1
        assert models[0]["name"] == "test_model"
        assert models[0]["classes"] == ["A", "B"]


class TestCoralFlowUI:
    def _make_ui(self, capsys):
        """Create a CoralFlowUI that captures output to stdout (pytest capsys)."""
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        # Override console to use the captured stdout
        ui.console.file = capsys  # type: ignore[assignment]
        return ui

    def test_markdown_renders_bold(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.markdown("Hello **world**!")
        captured = capsys.readouterr()
        assert "world" in captured.out

    def test_separator_produces_output(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.separator()
        captured = capsys.readouterr()
        # Rule produces a line of characters
        assert len(captured.out.strip()) > 0

    def test_info_renders_text(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.info("status message")
        captured = capsys.readouterr()
        assert "status message" in captured.out

    def test_error_renders_text(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.error("something went wrong")
        captured = capsys.readouterr()
        assert "something went wrong" in captured.out

    def test_tool_start_shows_name(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.tool_start("scan_datasets")
        captured = capsys.readouterr()
        assert "scan_datasets" in captured.out

    def test_tool_start_shows_run_shell_command(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.tool_start("run_shell", "train -d data.csv -o ./out")
        captured = capsys.readouterr()
        assert "run_shell" in captured.out
        assert "train -d data.csv -o ./out" in captured.out

    def test_success_shows_checkmark(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.success("done")
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_panel_renders_with_title(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.panel("panel **content**", title="My Title")
        captured = capsys.readouterr()
        assert "content" in captured.out
        assert "My Title" in captured.out

    def test_step_renders_title(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.step("Training Options")
        captured = capsys.readouterr()
        assert "Training Options" in captured.out

    def test_raw_output(self, capsys):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui.raw("streaming text")
        captured = capsys.readouterr()
        assert "streaming text" in captured.out

    def test_prompt_fallback_uses_input(self, monkeypatch):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        # Force no TTY so it uses input() fallback
        ui._has_tty = False
        monkeypatch.setattr("builtins.input", lambda prompt="": "hello")
        result = ui.prompt("coralflow")
        assert result == "hello"

    def test_choose_returns_option_by_number(self, monkeypatch):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui._has_tty = False
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")
        result = ui.choose(["option A", "option B", "option C"])
        assert result == "option B"

    def test_choose_returns_custom_text(self, monkeypatch):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui._has_tty = False

        inputs = iter(["0", "my custom input"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        result = ui.choose(["option A", "option B"])
        assert result == "my custom input"

    def test_choose_returns_direct_text(self, monkeypatch):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui._has_tty = False
        monkeypatch.setattr("builtins.input", lambda prompt="": "free text response")
        result = ui.choose(["option A"])
        assert result == "free text response"

    def test_confirm_yes(self, monkeypatch):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui._has_tty = False
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        assert ui.confirm("Proceed") is True

    def test_confirm_yes_default_empty(self, monkeypatch):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui._has_tty = False
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        assert ui.confirm("Proceed") is True

    def test_confirm_no(self, monkeypatch):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui._has_tty = False
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        assert ui.confirm("Proceed") is False

    def test_choose_out_of_range_returns_none(self, monkeypatch):
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        ui._has_tty = False
        monkeypatch.setattr("builtins.input", lambda prompt="": "99")
        result = ui.choose(["only one"])
        assert result is None


class TestMessageHistory:
    def test_prepare_drops_orphan_tool_message(self):
        from edge_train.agent.loop import _prepare_messages_for_llm

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "tool_call_id": "call_abc", "content": "orphan"},
            {"role": "user", "content": "3"},
        ]
        prepared = _prepare_messages_for_llm(messages)
        roles = [m["role"] for m in prepared]
        assert roles == ["system", "user"]

    def test_prepare_keeps_complete_tool_turn(self):
        from edge_train.agent.loop import _prepare_messages_for_llm

        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "predict", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "done"},
            {"role": "user", "content": "3"},
        ]
        prepared = _prepare_messages_for_llm(messages)
        assert len(prepared) == 3
        assert prepared[0]["tool_calls"]
        assert prepared[1]["role"] == "tool"

    def test_trim_does_not_split_tool_block(self):
        from edge_train.agent.loop import _trim_history

        assistant_tc = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "scan_datasets", "arguments": "{}"},
                }
            ],
        }
        tool_msg = {"role": "tool", "tool_call_id": "call_x", "content": "ok"}
        messages = [{"role": "system", "content": "s"}]
        for i in range(25):
            messages.append({"role": "user", "content": f"u{i}"})
            messages.append({"role": "assistant", "content": f"a{i}"})
        messages.extend([assistant_tc, tool_msg, {"role": "user", "content": "3"}])

        _trim_history(messages, max_messages=10)
        prepared_roles = []
        from edge_train.agent.loop import _prepare_messages_for_llm

        for m in _prepare_messages_for_llm(messages):
            prepared_roles.append(m["role"])

        assert "tool" not in prepared_roles or (
            "assistant" in prepared_roles
            and prepared_roles.index("assistant") < prepared_roles.index("tool")
        )


class TestSlashCommands:
    def test_datasets_slash_runs_scan(self):
        from edge_train.agent.loop import _handle_slash_command
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        msg, result = _handle_slash_command("/datasets", ui)
        assert msg is not None
        assert msg["tool_calls"][0]["function"]["name"] == "scan_datasets"
        assert "dataset" in result.lower() or "No datasets" in result

    def test_unknown_slash_returns_none(self, capsys):
        from edge_train.agent.loop import _handle_slash_command
        from edge_train.agent.ui import CoralFlowUI

        ui = CoralFlowUI()
        msg, result = _handle_slash_command("/notacommand", ui)
        assert msg is None
        assert result == ""


class TestRunShell:
    def test_coralflow_subcommand_streams_cli(self, monkeypatch):
        from edge_train.agent.tools import execute_tool

        calls = []

        def fake_stream(argv, timeout=300, *, skip_phoenix=False):
            calls.append((argv, timeout))
            return "epoch 1/10", 0

        monkeypatch.setattr("edge_train.agent.tools._stream_coralflow_cli", fake_stream)
        result = execute_tool("run_shell", {"command": "train --help"})
        assert "epoch" in result
        assert "-m" in calls[0][0]
        assert "edge_train.cli" in calls[0][0]

    def test_other_commands_use_shell(self, monkeypatch):
        from edge_train.agent.tools import execute_tool

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs.get("shell")))

            class R:
                stdout = "done"
                stderr = ""
                returncode = 0

            return R()

        monkeypatch.setattr("edge_train.agent.tools.subprocess.run", fake_run)
        result = execute_tool("run_shell", {"command": "echo hello"})
        assert "done" in result
        assert calls[0] == ("echo hello", True)
