# CLAUDE.md

Guidance for Claude Code when working on CoralFlow.

## Project Overview

CoralFlow — TinyML continuous training agent. Train, validate, deploy, monitor, and auto-retrain ML models on edge devices from the CLI, zero GPU needed. Includes an LLM-powered interactive agent (`coralflow agent`) that orchestrates the full pipeline.

- **Version**: 0.4.0 (Week 4: LLM-powered agent + dataset intelligence)
- **License**: MIT

## Architecture

### Module structure
```
edge_train/
├── __init__.py           # version
├── config.py             # GCPConfig, ArizeConfig, TrainConfig, EdgeConfig, LLMConfig
├── agent/                # LLM-powered interactive agent (v0.4.0)
│   ├── __init__.py       # AgentState, DatasetScanner, Recommender, scan_models
│   ├── loop.py           # REPL loop: prompt → LLM → tools → response
│   ├── llm.py            # OpenAI-compatible LLM API client
│   ├── tools.py          # Tool definitions (train, predict, deploy, scan...)
│   ├── resources.py      # Local hardware resource assessment (CPU, RAM, disk)
│   ├── validate_dataset.py # LLM-driven dataset quality validation
│   ├── recommend.py      # Web-based dataset recommendations
│   └── progress.py       # TrainingProgressCallback for live epoch output
├── cli/                  # Click CLI — 8 commands
│   ├── __init__.py, agent.py, init.py, train.py, validate.py, deploy.py, monitor.py, cost.py, predict.py
├── cloud/                # Vertex AI AutoML integration (text/image/table)
├── datasets/             # Built-in datasets (urgent, expense) + modality detection
├── inference/            # Local inference — TextClassifier loads SavedModel + model_meta.json
├── trainer/              # Local training — TF Keras text classifier
├── validation/           # TFLite conversion, size/latency constraints
└── edge/                 # Device SDK (Week 2)
    ├── config.py, registry.py, model.py, deploy.py, transport/
    └── transport/http.py # HTTP transport (aiohttp)
```

### Key decisions
- CLI entry point: `coralflow` (package name `edge_train` unchanged)
- 8 CLI commands: agent, init, train, validate, deploy, monitor, cost, predict
- `coralflow agent` is an LLM-powered REPL — slash commands go through the LLM as tool calls
- OpenAI-compatible API with function calling for tool use
- Agent state persists to `~/.coralflow/agent_state.json`
- Vertex AI AutoML for cloud training (`--cloud` flag), local TF Keras by default
- Local inference uses SavedModel (not TFLite) to keep TextVectorization in-graph
- Phoenix OTEL spans created per prediction with OpenInference conventions
- Prediction log (JSON lines) enables retrain loop — append `ground_truth` to trigger retraining
- Device SDK supports HTTP transport with aiohttp, extensible for MQTT/SSH/BLE

### Agent flow
```
coralflow agent
  → load LLM config, scan datasets/models, print banner
  → REPL: prompt("coralflow> ") → LLM → tools → response → repeat
  → Slash commands parsed into tool calls, routed through LLM
  → /exit saves state and quits
```

### Predict/Retrain flow
```
coralflow predict --model <path> --text "..."      # single prediction
coralflow predict --model <path> --csv data.csv    # batch prediction
coralflow monitor --retrain --dataset original.csv # check accuracy, retrain if needed
```
Predictions are logged to `prediction_log.jsonl` (configurable via `EDGE_PREDICTION_LOG_PATH`).
Phoenix OTEL spans are created per prediction when Phoenix is configured.
Add `"ground_truth"` fields to log entries to enable retraining.

## Development

### Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Commits
- Never include `Co-Authored-By` trailers in commit messages
- Use conventional commit prefixes: `feat:`, `fix:`, `style:`, `test:`, `chore:`

### Before pushing
**Always run locally first — never skip this.** CI failures waste time.
```bash
source .venv/bin/activate && black --check --target-version py310 edge_train/ tests/
source .venv/bin/activate && python -m pytest tests/ -v
```
Both must pass before `git push`. If black check fails, run `black --target-version py310 edge_train/ tests/` to auto-fix.

### Run all tests
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

### Run specific tests
```bash
python -m pytest tests/test_edge_deploy.py -v
python -m pytest tests/ -k "deploy or edge" -v
```

### Run CLI
```bash
source .venv/bin/activate && python -c "from edge_train.cli import main; main(['deploy', '--help'])"
```

### Run agent
```bash
CORALFLOW_LLM_API_KEY=sk-xxx coralflow agent
coralflow agent --model gpt-4o-mini
```

### Config
All config reads from environment variables:
- `GCP_PROJECT`, `GCP_LOCATION`, `GCP_STAGING_BUCKET` — GCP config
- `PHOENIX_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT` — Arize Phoenix OTEL tracing
  - `PHOENIX_PROJECT_NAME` (optional, default: `edge-train`) — project name in Phoenix dashboard
  - `PHOENIX_COLLECTOR_ENDPOINT` defaults to `https://app.phoenix.arize.com/v1/traces`
- `EDGE_REGISTRY_PATH` — device registry file path
- `EDGE_PREDICTION_LOG_PATH` — prediction log file path (default: `./prediction_log.jsonl`)
- `CORALFLOW_LLM_ENDPOINT` — LLM API endpoint (default: `https://api.openai.com/v1`)
- `CORALFLOW_LLM_API_KEY` — LLM API key (required for `coralflow agent`)
- `CORALFLOW_LLM_MODEL` — LLM model name (default: `gpt-4o`)

### Versioning
Bump version in `edge_train/__init__.py` and `pyproject.toml`.
