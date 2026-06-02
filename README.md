# CoralFlow

**The AI-Driven TinyML Pipeline.**

Seamlessly train, validate, and deploy models to edge devices via CLI. With built-in drift detection, auto-retraining, and HTTP-based updates, our LLM agent takes care of the heavy lifting—managing your entire workflow automatically.

<p align="center">
  <a href="https://www.tensorflow.org/">TensorFlow</a>
  &nbsp;·&nbsp;
  <a href="https://www.tensorflow.org/lite">TensorFlow Lite</a>
  &nbsp;·&nbsp;
  <a href="https://cloud.google.com/vertex-ai">Vertex AI</a>
  &nbsp;·&nbsp;
  <a href="https://phoenix.arize.com/">Arize Phoenix</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/Arize-ai/openinference">OpenInference</a>
  &nbsp;·&nbsp;
  <a href="https://platform.deepseek.com/">DeepSeek</a>
  &nbsp;·&nbsp;
  <a href="https://ai.google.dev/gemini-api">Gemini</a>
</p>

**Version:** 0.4.0 · **License:** MIT

```
git clone <repo> && cd coralflow
cp .env.example .env          # optional
./scripts/dev pip install -e .
coralflow demo retrain-loop   # no API keys required
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

### Install

**Requirements:** Python 3.10+, Linux or macOS (WSL2 works). First install pulls TensorFlow and may take a few minutes.

```bash
git clone <repo>
cd coralflow
cp .env.example .env    # optional — local train/predict need no keys
./scripts/dev pip install -e .
coralflow --help
```

`scripts/dev` creates `.venv` automatically (`uv venv` if available, else `python3 -m venv`).

**Contributors** (pytest, black, Flask gateway dev deps):

```bash
make install    # same as: ./scripts/dev pip install -e ".[dev]"
```

**Install from GitHub without cloning** (same package, no editable mode):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "git+https://github.com/<your-org>/coralflow.git"
coralflow --help
```

### Try it (no API keys)

Fastest path — built-in dataset, full retrain demo:

```bash
coralflow demo retrain-loop
```

Step-by-step local workflow:

```bash
coralflow init download urgent -o ./data
coralflow train -d ./data/urgent.csv -o ./model_output --epochs 10
coralflow validate --model ./model_output
coralflow predict --model ./model_output --text "need this done today"
```

### Configure optional features

Copy `.env.example` to `.env` and uncomment what you need:

| Goal | Variables to set |
|------|------------------|
| Cloud training | `GCP_PROJECT`, `GCP_STAGING_BUCKET`, credentials (see below) |
| Deploy to edge | `EDGE_DEVICES`, `EDGE_DEFAULT_DEVICE` |
| Phoenix tracing | `PHOENIX_COLLECTOR_ENDPOINT` (+ `PHOENIX_API_KEY` for cloud) |
| LLM agent | `CORALFLOW_LLM_API_KEY` |

**Cloud training (GCP)** — add to `.env` or export:

```bash
# In .env:
# GCP_PROJECT=your-project
# GCP_STAGING_BUCKET=gs://your-bucket
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
coralflow train -d ./data/urgent.csv --cloud
```

Or use Application Default Credentials: `gcloud auth application-default login`.

**Edge deploy** — point at a gateway running `examples/edge_gateway`:

```bash
# In .env:
# EDGE_DEVICES=[{"device_id":"gw1","host":"127.0.0.1","port":8080}]
# EDGE_DEFAULT_DEVICE=gw1
coralflow deploy --model ./model_output
```

**Interactive agent** — requires an OpenAI-compatible API key in `.env`:

```bash
# CORALFLOW_LLM_API_KEY=sk-...
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

Start from the template:

```bash
cp .env.example .env
```

See [`.env.example`](.env.example) for commented examples (GCP, Phoenix, edge gateways, LLM agent).

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT` | Google Cloud project (cloud training) |
| `GCP_LOCATION` | GCP region (default: `us-central1`) |
| `GCP_STAGING_BUCKET` | GCS bucket for cloud datasets |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON (optional with ADC) |
| `GCP_VERTEX_MACHINE_TYPE` | Vertex deploy machine type (default: `n1-standard-2`) |
| `PHOENIX_API_KEY` | Phoenix Cloud API key |
| `PHOENIX_COLLECTOR_ENDPOINT` | OTEL trace endpoint (local or cloud) |
| `PHOENIX_PROJECT_NAME` | Project name in Phoenix (default: `edge-train`) |
| `EDGE_DEVICES` | Edge gateways — JSON array or `id@host:port,...` |
| `EDGE_DEFAULT_DEVICE` | Default gateway id for `coralflow deploy` |
| `EDGE_REGISTRY_PATH` | Legacy JSON registry fallback (optional) |
| `EDGE_MODEL_PATH` | Edge gateway model file path (gateway server only) |
| `EDGE_PREDICTION_LOG_PATH` | Prediction log path (default: `./prediction_log.jsonl`) |
| `CORALFLOW_LLM_API_KEY` | LLM API key (required for `coralflow agent`) |
| `CORALFLOW_LLM_ENDPOINT` | OpenAI-compatible API base URL |
| `CORALFLOW_LLM_MODEL` | Model name (default: `gpt-4o`) |
| `CORALFLOW_TRAINING_HISTORY_PATH` | Override training history JSON path |
| `CORALFLOW_DEPLOYMENTS_PATH` | Override cloud deployment registry path |

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
