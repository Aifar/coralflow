# CLAUDE.md

Guidance for Claude Code when working on CoralFlow / edge-train.

## Project Overview

CoralFlow — TinyML continuous training agent. Train, validate, deploy, monitor, and auto-retrain ML models on edge devices from the CLI, zero GPU needed.

- **Version**: 0.3.0 (Week 3: Local inference + auto-retrain loop)
- **License**: MIT

## Architecture

### Module structure
```
edge_train/
├── __init__.py           # version
├── config.py             # GCPConfig, ArizeConfig, TrainConfig, EdgeConfig
├── cli/                  # Click CLI — 7 commands
│   ├── __init__.py, init.py, train.py, validate.py, deploy.py, monitor.py, cost.py, predict.py
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
- Flat module layout under `edge_train/`
- Click CLI framework, 7 commands: init, train, validate, deploy, monitor, cost, predict
- Vertex AI AutoML for cloud training (`--cloud` flag), local TF Keras by default
- Local inference uses SavedModel (not TFLite) to keep TextVectorization in-graph
- Phoenix OTEL spans created per prediction with OpenInference conventions
- Prediction log (JSON lines) enables retrain loop — append `ground_truth` to trigger retraining
- Device SDK supports HTTP transport with aiohttp, extensible for MQTT/SSH/BLE
- Local device registry backed by JSON file
- Model packaging: .tflite + manifest.json (SHA-256, version, modality)

### Predict/Retrain flow
```
edge-train predict --model <path> --text "..."      # single prediction
edge-train predict --model <path> --csv data.csv    # batch prediction
edge-train monitor --retrain --dataset original.csv # check accuracy, retrain if needed
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

### Config
All config reads from environment variables:
- `GCP_PROJECT`, `GCP_LOCATION`, `GCP_STAGING_BUCKET` — GCP config
- `PHOENIX_API_KEY`, `PHOENIX_COLLECTOR_ENDPOINT` — Arize Phoenix OTEL tracing
  - `PHOENIX_PROJECT_NAME` (optional, default: `edge-train`) — project name in Phoenix dashboard
  - `PHOENIX_COLLECTOR_ENDPOINT` defaults to `https://app.phoenix.arize.com/v1/traces`
- `EDGE_REGISTRY_PATH` — device registry file path
- `EDGE_PREDICTION_LOG_PATH` — prediction log file path (default: `./prediction_log.jsonl`)

### Versioning
Bump version in `edge_train/__init__.py` and `pyproject.toml`.
