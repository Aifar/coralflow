# CLAUDE.md

Guidance for Claude Code when working on CoralFlow / edge-train.

## Project Overview

CoralFlow — TinyML continuous training agent. Train, validate, deploy, monitor, and auto-retrain ML models on edge devices from the CLI, zero GPU needed.

- **Version**: 0.2.0 (Week 2: Device SDK & OTA deploy complete)
- **License**: MIT

## Architecture

### Module structure
```
edge_train/
├── __init__.py           # version
├── config.py             # GCPConfig, ArizeConfig, TrainConfig, EdgeConfig
├── cli/                  # Click CLI — 6 commands
│   ├── __init__.py, init.py, train.py, validate.py, deploy.py, monitor.py, cost.py
├── cloud/                # Vertex AI AutoML integration (text/image/table)
├── datasets/             # Built-in datasets (urgent, expense) + modality detection
├── validation/           # TFLite conversion, size/latency constraints
└── edge/                 # Device SDK (Week 2)
    ├── config.py, registry.py, model.py, deploy.py, transport/
    └── transport/http.py # HTTP transport (aiohttp)
```

### Key decisions
- Flat module layout under `edge_train/`
- Click CLI framework, 6 commands: init, train, validate, deploy, monitor, cost
- Vertex AI AutoML for cloud training
- Device SDK supports HTTP transport with aiohttp, extensible for MQTT/SSH/BLE
- Local device registry backed by JSON file
- Model packaging: .tflite + manifest.json (SHA-256, version, modality)

## Development

### Setup
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

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
- `ARIZE_API_KEY`, `ARIZE_SPACE_KEY`, `ARIZE_ENDPOINT` — Arize AI monitoring
- `EDGE_REGISTRY_PATH` — device registry file path

### Versioning
Bump version in `edge_train/__init__.py` and `pyproject.toml`.
