"""Local device registry — maps device IDs to connection info."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class DeviceInfo:
    """Connection details for a single edge device."""

    device_id: str
    transport: str = "http"
    host: str = "localhost"
    port: int = 8080
    api_key: str | None = None
    label: str = ""


class DeviceRegistry:
    """Persistent registry of edge devices backed by a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._devices: dict[str, DeviceInfo] = {}
        self._load()

    def resolve(self, device_id: str) -> DeviceInfo | None:
        return self._devices.get(device_id)

    def register(self, device: DeviceInfo) -> None:
        self._devices[device.device_id] = device
        self._save()

    def unregister(self, device_id: str) -> None:
        self._devices.pop(device_id, None)
        self._save()

    def list_devices(self) -> list[DeviceInfo]:
        return list(self._devices.values())

    def _load(self) -> None:
        if not self._path.exists():
            self._devices = {}
            return
        raw = self._path.read_text()
        if not raw.strip():
            self._devices = {}
            return
        data = json.loads(raw)
        self._devices = {
            did: DeviceInfo(**info) for did, info in data.get("devices", {}).items()
        }

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"devices": {did: asdict(info) for did, info in self._devices.items()}}
        self._path.write_text(json.dumps(data, indent=2))
