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

## Usage guide

The recommended path is the **interactive agent**. It walks you through dataset discovery, training, optional cloud/observability setup, and retraining — in plain language.

**Requirements:** Python 3.10+, Linux or macOS (WSL2 works). First install pulls TensorFlow and may take a few minutes.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install coralflow
coralflow                   # launch the interactive agent
```

Optional: copy settings template before first run (`cp .env.example .env` in a project directory). The agent can also save LLM and cloud settings to `.env` interactively.

**From source (contributors):**

```bash
git clone https://github.com/Aifar/coralflow.git
cd coralflow
cp .env.example .env
./scripts/dev pip install -e ".[dev]"
coralflow
```

`scripts/dev` creates `.venv` automatically (`uv venv` if available, else `python3 -m venv`).

---

### 1. Launch the agent

Run `coralflow` with no subcommand (or `coralflow agent`). You enter a REPL:

```text
coralflow>
```

On first launch the agent scans local datasets and trained models, then prints a short summary. Type `/help` for slash commands or just describe what you want in natural language.

---

### 2. Configure LLM (required)

The agent **must** have a working LLM before it can orchestrate training. On first run you will be prompted to:

1. **Choose a provider** — DeepSeek, Gemini, or other OpenAI-compatible API  
2. **Set endpoint** — defaults are shown; press Enter or edit  
3. **Set API key** — saved to `.env` as `CORALFLOW_LLM_API_KEY`  
4. **Set model** — e.g. `deepseek-chat`, `gemini-2.0-flash`, `gpt-4o`

You can also preconfigure in `.env`:

```bash
CORALFLOW_LLM_API_KEY=sk-...
# CORALFLOW_LLM_ENDPOINT=https://api.deepseek.com/v1
# CORALFLOW_LLM_MODEL=deepseek-chat
```

Or pass flags once: `coralflow --api-key sk-... --endpoint ... --model ...`

---

### 3. Describe the model you want to train

Tell the agent your task in natural language. For example:

```text
coralflow> I want a text classifier that tells whether support messages are urgent or not
```

The agent will:

- **Scan for data** — built-in datasets (`urgent`, `expense`), CSV files on disk, and paths you mention  
- **Recommend public datasets** from the web if nothing local fits your task  
- **Ask you to pick** a built-in dataset, download one (`coralflow init download urgent -o ./data`), or point to **your own CSV** (text + label columns)

Example follow-ups you might see:

```text
coralflow> Use the built-in urgent dataset
coralflow> Train on ./data/my_labels.csv — text column is "message", label is "category"
```

Built-in datasets:

| Name | Modality | Samples | Classes | Description |
|------|----------|---------|---------|-------------|
| urgent | text | 400 | 4 | Message urgency classification |
| expense | text | 500 | 5 | Personal spending intent classification |

Download a built-in dataset without the agent:

```bash
coralflow init download urgent -o ./data
```

---

### 4. Validate data and choose local or cloud training

Before training, the agent **validates dataset quality** (columns, class balance, samples) and **assesses your machine** (CPU, RAM, disk).

It then asks you to choose:

| Option | When to use | Extra setup |
|--------|-------------|-------------|
| **1 — Local training** | Laptop/edge-friendly text models; no GCP bill | None |
| **2 — Cloud training** | Large data, AutoML, or Gemini fine-tuning on Vertex AI | Google Cloud (below) |

Reply with `1` or `2`. Local training uses TensorFlow on your machine; cloud training uses Vertex AI (`train --cloud`).

**Google Cloud (only for cloud training)** — when you pick option 2, or at startup when prompted, configure:

```bash
GCP_PROJECT=your-gcp-project-id
GCP_LOCATION=us-central1
GCP_STAGING_BUCKET=gs://your-staging-bucket
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

Or run `gcloud auth application-default login` instead of a service-account file.

Used by: `train --cloud`, `coralflow models list`, `coralflow cost`, Vertex predict/deploy.

After training completes, the agent can **validate** the model (TFLite size/latency) and **deploy** to edge gateways if `EDGE_DEVICES` is set in `.env`.

---

### 5. Optional: Arize Phoenix logging

If you want **prediction traces and monitoring** in [Arize Phoenix](https://phoenix.arize.com/), configure Phoenix when the agent asks (after LLM setup), or add to `.env`:

**Local Phoenix:**

```bash
pip install arize-phoenix
phoenix serve
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_PROJECT_NAME=edge-train
```

**Phoenix Cloud:**

```bash
PHOENIX_API_KEY=your-phoenix-api-key
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/v1/traces
PHOENIX_PROJECT_NAME=edge-train
```

Once configured, each `predict` emits OpenInference OTEL spans. The agent's `check_monitoring` tool verifies Phoenix is reachable before monitor/predict workflows.

Skip this step if you only need local train/predict without a trace dashboard.

---

### 6. Monitor the model and retrain

After you have a trained model:

1. **Run predictions** — the agent calls `predict`, or use CLI:

   ```bash
   coralflow predict --model ./model_output --text "need this done today"
   coralflow predict --model ./model_output --csv holdout.csv
   ```

   Results append to `prediction_log.jsonl` (override with `EDGE_PREDICTION_LOG_PATH`).

2. **Check monitoring** — ask the agent *"Is Phoenix receiving traces?"* or run:

   ```bash
   coralflow monitor --status
   ```

3. **Label mistakes** — the agent uses `label_predictions` to list unlabeled rows; you correct wrong predictions (add `ground_truth` to the log).

4. **Retrain when accuracy drops** — ask *"Should we retrain?"* or run:

   ```bash
   coralflow monitor --retrain --dataset ./data/urgent.csv
   ```

   The agent flow: list unlabeled predictions → you label errors → `check_retrain` merges labels and retrains if accuracy is below threshold.

---

## CLI reference

You can run every step without the agent:

| Step | Command | Description |
|------|---------|-------------|
| Setup | `init list` / `init download` | Built-in datasets or your own CSV |
| Train | `train -d <csv> [-o dir]` | Local TF Keras (default); `--cloud` for Vertex AI |
| Validate | `validate --model <path>` | TFLite conversion, size & latency checks |
| Predict | `predict --model <path> --text "..."` | Local inference; logs to `prediction_log.jsonl` |
| Deploy | `deploy --model <path> [--device <id>]` | Push TFLite to gateways in `EDGE_DEVICES` |
| Edge pipeline | `coralflow-edge-pipeline ./data/urgent.csv` | Train → validate → deploy in one command |
| Monitor | `monitor` | Phoenix OTEL tracing & retrain triggers |
| Cost | `cost <dataset>` | Estimate Vertex AI training cost |
| Agent | `coralflow` / `agent` | LLM REPL with tools for the full workflow |
| Demo | `demo retrain-loop` | Scripted drift → retrain → accuracy improvement |

Slash commands in the agent (e.g. `/datasets`, `/help`) run locally first; long `train` / `validate` jobs stream live output in the terminal.

---

## Configuration

Copy the template and uncomment what you need:

```bash
cp .env.example .env
```

See [`.env.example`](.env.example) for full commented examples.

| Variable | Description |
|----------|-------------|
| `CORALFLOW_LLM_API_KEY` | LLM API key (**required for agent**) |
| `CORALFLOW_LLM_ENDPOINT` | OpenAI-compatible API base URL |
| `CORALFLOW_LLM_MODEL` | Model name (default: `gpt-4o`) |
| `GCP_PROJECT` | Google Cloud project (cloud training) |
| `GCP_LOCATION` | GCP region (default: `us-central1`) |
| `GCP_STAGING_BUCKET` | GCS bucket for cloud datasets |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON (optional with ADC) |
| `PHOENIX_API_KEY` | Phoenix Cloud API key |
| `PHOENIX_COLLECTOR_ENDPOINT` | OTEL trace endpoint (local or cloud) |
| `PHOENIX_PROJECT_NAME` | Project name in Phoenix (default: `edge-train`) |
| `EDGE_DEVICES` | Edge gateways — JSON array or `id@host:port,...` |
| `EDGE_DEFAULT_DEVICE` | Default gateway id for `coralflow deploy` |
| `EDGE_PREDICTION_LOG_PATH` | Prediction log path (default: `./prediction_log.jsonl`) |

---

## Development

**Contributors** (pytest, black, Flask gateway dev deps):

```bash
make install    # ./scripts/dev pip install -e ".[dev]"
make test
make format-check
```

Before pushing, run `make lint` (format-check + test).

Install the latest published release:

```bash
pip install -U coralflow
```

---

## License

MIT
