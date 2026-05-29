# CoralFlow

**TinyML continuous training** — train, validate, deploy, monitor, and auto-retrain text classifiers from the CLI. No GPU required for local training. Includes an LLM-powered interactive agent (`coralflow agent`) that guides the full pipeline.

**Version:** 0.4.0 · **License:** MIT

```
pip install -e ".[dev]"
coralflow init list
coralflow train -d ./data/urgent.csv -o ./model_output
coralflow predict --model ./model_output --text "urgent meeting now"
coralflow agent
```

## What it does

| Step | Command | Description |
|------|---------|-------------|
| Setup | `init list` / `init download` | Built-in datasets or your own CSV |
| Train | `train -d <csv> [-o dir]` | Local TF Keras training (default); `--cloud` for Vertex AI AutoML |
| Validate | `validate --model <path>` | TFLite conversion, size & latency checks |
| Predict | `predict --model <path> --text "..."` | Local inference; logs to `prediction_log.jsonl` |
| Deploy | `deploy --model <path> [--device <id>]` | Push TFLite to gateways in `EDGE_DEVICES` (.env) |
| Edge pipeline | `coralflow-edge-pipeline ./data/urgent.csv` | Train → validate → deploy in one command |
| Monitor | `monitor` | Phoenix OTEL tracing & retrain triggers |
| Cost | `cost <dataset>` | Estimate Vertex AI training cost |
| Agent | `agent` | LLM REPL with tools for the full workflow |
| Demo | `demo retrain-loop` | Scripted drift → retrain → accuracy improvement |

## Quick start

### Install (development)

```bash
git clone <repo>
cd coralflow
./scripts/dev pip install -e ".[dev]"
# or: make install
```

`scripts/dev` creates and activates `.venv` automatically (`uv venv` if available, else `python3 -m venv`).

### Train locally (no cloud keys)

```bash
coralflow init download urgent -o ./data
coralflow train -d ./data/urgent.csv -o ./model_output --epochs 10
coralflow validate --model ./model_output
coralflow predict --model ./model_output --text "need this done today"
```

### Cloud training (GCP)

```bash
export GCP_PROJECT=your-project
export GCP_STAGING_BUCKET=gs://your-bucket
coralflow train -d ./data/urgent.csv --cloud
```

### Interactive agent

Requires an OpenAI-compatible API key:

```bash
export CORALFLOW_LLM_API_KEY=sk-...
coralflow agent
# optional: coralflow agent --model gpt-4o-mini
```

Slash commands such as `/datasets`, `/train`, and `/help` run locally first; the agent then summarizes results. For long `train` / `validate` runs, CLI output streams live in the terminal.

## Monitoring with Arize Phoenix

Predictions emit OpenInference OTEL spans when Phoenix is configured.

**Local Phoenix:**

```bash
pip install arize-phoenix
phoenix serve
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
export PHOENIX_PROJECT_NAME=edge-train
coralflow monitor          # check connectivity
coralflow predict --model ./model_output --text "hello"
```

**Phoenix Cloud:** set `PHOENIX_API_KEY` and `PHOENIX_COLLECTOR_ENDPOINT` (defaults to `https://app.phoenix.arize.com/v1/traces`).

`coralflow monitor` and the agent's `check_monitoring` tool verify Phoenix is reachable before predict/monitor workflows.

## Retrain loop

Predictions append to `prediction_log.jsonl` (override with `EDGE_PREDICTION_LOG_PATH`). Add `"ground_truth"` to log entries, then:

```bash
coralflow monitor --retrain --dataset ./data/urgent.csv
```

The agent can walk through labeling via `label_predictions` and `check_retrain`.

### One-command retrain demo

Runs a full loop: train a weak baseline, mispredict challenge phrases, merge labels, retrain, and print before/after accuracy:

```bash
coralflow demo retrain-loop
# faster CI / laptop: coralflow demo retrain-loop --epochs 2 --retrain-epochs 5
```

Uses `builtin:urgent` by default. With `--hard` (default), the baseline never sees **紧急** / **重要** labels; the challenge set does, so accuracy drops below the retrain threshold, then improves after merge + retrain.

## Built-in datasets

| Name | Modality | Samples | Classes | Description |
|------|----------|---------|---------|-------------|
| urgent | text | 400 | 4 | Message urgency classification |
| expense | text | 500 | 5 | Personal spending intent classification |

## Configuration

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT` | Google Cloud project (cloud training) |
| `GCP_LOCATION` | GCP region (default: `us-central1`) |
| `GCP_STAGING_BUCKET` | GCS bucket for cloud datasets |
| `PHOENIX_API_KEY` | Phoenix Cloud API key |
| `PHOENIX_COLLECTOR_ENDPOINT` | OTEL trace endpoint |
| `PHOENIX_PROJECT_NAME` | Project name in Phoenix (default: `edge-train`) |
| `EDGE_DEVICES` | Edge gateways in `.env` — JSON array or `id@host:port,...` |
| `EDGE_DEFAULT_DEVICE` | Default gateway id for `coralflow deploy` |
| `EDGE_REGISTRY_PATH` | Legacy JSON registry fallback (optional) |
| `EDGE_MODEL_PATH` | Edge gateway model file path (default: `/var/lib/coralflow/model.tflite`) |
| `EDGE_PREDICTION_LOG_PATH` | Prediction log path (default: `./prediction_log.jsonl`) |
| `CORALFLOW_LLM_API_KEY` | LLM API key (required for `coralflow agent`) |
| `CORALFLOW_LLM_ENDPOINT` | OpenAI-compatible API base URL |
| `CORALFLOW_LLM_MODEL` | Model name (default: `gpt-4o`) |

## Development

```bash
make install          # create .venv + editable install
make test             # pytest
make format-check     # black --check
make format           # auto-format
```

Before pushing, run `make lint` (format-check + test).

## License

MIT
