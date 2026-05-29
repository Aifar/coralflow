# CoralFlow Edge Gateway

Minimal Flask service for edge devices (Raspberry Pi, Jetson, industrial PC).
Implements the HTTP contract expected by `coralflow deploy`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness (200) |
| POST | `/api/v1/model` | Receive multipart `model` (`.tflite`) |
| GET | `/api/v1/checksum` | Return `{"sha256": "..."}` |
| POST | `/api/v1/reload` or `/reload` | Hot-reload TFLite interpreter (optional) |

## Quick start (dev)

```bash
cd examples/edge_gateway
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
EDGE_MODEL_PATH=/tmp/coralflow-model.tflite python app.py
```

On the development machine, point `.env` at the gateway:

```bash
EDGE_DEVICES='[{"device_id":"dev-gw","host":"127.0.0.1","port":8080}]'
EDGE_DEFAULT_DEVICE=dev-gw
```

## systemd (production)

```bash
sudo useradd -r -s /usr/sbin/nologin coralflow || true
sudo mkdir -p /opt/coralflow/edge_gateway /var/lib/coralflow
sudo cp -r examples/edge_gateway/* /opt/coralflow/edge_gateway/
cd /opt/coralflow/edge_gateway && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sudo cp coralflow-edge-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now coralflow-edge-gateway
curl -fsS http://localhost:8080/health
```
