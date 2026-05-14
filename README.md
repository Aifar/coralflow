# CoralFlow — edge-train

**TinyML continuous training CLI** — from dataset to edge deployment in one pipeline.

```
pip install edge-train
edge-train init list
edge-train train --dataset urgent
edge-train validate --model output/model
```

## What it does

| Step | Command | Description |
|------|---------|-------------|
| Setup | `init list / download / custom` | Browse built-in datasets or prepare your own |
| Train | `train --dataset <name>` | Submit to Vertex AI AutoML, poll until done |
| Validate | `validate --model <path>` | Convert to TFLite, check size & latency constraints |
| Deploy | `deploy` | OTA push to edge devices (coming in v0.2.0) |
| Monitor | `monitor` | Arize AI model health dashboard (coming in v0.2.0) |
| Cost | `cost <dataset>` | Estimate training cost before running |

## Quick start

```bash
# Install
pip install edge-train

# List built-in datasets
edge-train init list

# Download a dataset
edge-train init download urgent -o ./data

# Train (requires GCP credentials)
edge-train train --dataset urgent

# Validate the exported model
edge-train validate --model ./saved_model --output ./model.tflite

# Estimate cost
edge-train cost urgent
```

## Built-in datasets

| Name | Modality | Samples | Classes | Description |
|------|----------|---------|---------|-------------|
| urgent | text | 400 | 4 | 消息紧急程度分类 |
| expense | text | 500 | 5 | 个人消费意图分类 |

## Configuration

Set these environment variables:

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `GCP_LOCATION` | GCP region (default: us-central1) |
| `GCP_STAGING_BUCKET` | GCS bucket for datasets |
| `ARIZE_API_KEY` | Arize AI API key |
| `ARIZE_SPACE_KEY` | Arize AI space key |

## License

MIT
