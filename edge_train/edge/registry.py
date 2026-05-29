"""Edge device registry — load gateways from .env or legacy JSON file."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
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


_COMPACT_ENTRY = re.compile(r"^(?P<id>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)$")


def parse_edge_devices(raw: str) -> list[DeviceInfo]:
    """Parse EDGE_DEVICES from .env — JSON array or compact id@host:port list."""
    text = raw.strip()
    if not text:
        return []

    if text.startswith("["):
        return _parse_edge_devices_json(text)
    return _parse_edge_devices_compact(text)


def _parse_edge_devices_json(text: str) -> list[DeviceInfo]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"EDGE_DEVICES JSON is invalid: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("EDGE_DEVICES JSON must be an array of device objects")

    devices: list[DeviceInfo] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"EDGE_DEVICES[{index}] must be an object")
        device_id = str(item.get("device_id") or item.get("id") or "").strip()
        host = str(item.get("host") or "").strip()
        if not device_id or not host:
            raise ValueError(
                f"EDGE_DEVICES[{index}] requires device_id (or id) and host"
            )
        port_raw = item.get("port", 8080)
        try:
            port = int(port_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"EDGE_DEVICES[{index}].port must be an integer") from exc

        api_key = item.get("api_key")
        devices.append(
            DeviceInfo(
                device_id=device_id,
                transport=str(item.get("transport") or "http"),
                host=host,
                port=port,
                api_key=str(api_key) if api_key else None,
                label=str(item.get("label") or ""),
            )
        )
    return devices


def _parse_edge_devices_compact(text: str) -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    for part in text.split(","):
        entry = part.strip()
        if not entry:
            continue
        match = _COMPACT_ENTRY.match(entry)
        if not match:
            raise ValueError(
                "EDGE_DEVICES compact entry must look like id@host:port "
                f"(got {entry!r})"
            )
        devices.append(
            DeviceInfo(
                device_id=match.group("id").strip(),
                host=match.group("host").strip(),
                port=int(match.group("port")),
            )
        )
    return devices


class DeviceRegistry:
    """Registry of edge devices (in-memory from .env or backed by a JSON file)."""

    def __init__(self, path: str | Path) -> None:
        self._path: Path | None = Path(path)
        self._devices: dict[str, DeviceInfo] = {}
        self._load()

    @classmethod
    def from_list(cls, devices: list[DeviceInfo]) -> DeviceRegistry:
        reg = cls.__new__(cls)
        reg._path = None
        reg._devices = {d.device_id: d for d in devices}
        return reg

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
        if self._path is None or not self._path.exists():
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
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"devices": {did: asdict(info) for did, info in self._devices.items()}}
        self._path.write_text(json.dumps(data, indent=2))


def load_device_registry() -> DeviceRegistry:
    """Load devices from EDGE_DEVICES in .env, else legacy JSON if present."""
    raw = os.environ.get("EDGE_DEVICES", "").strip()
    if raw:
        return DeviceRegistry.from_list(parse_edge_devices(raw))

    legacy_path = os.environ.get(
        "EDGE_REGISTRY_PATH", os.path.expanduser("~/.edge-train/devices.json")
    )
    path = Path(legacy_path)
    if path.exists() and path.read_text().strip():
        return DeviceRegistry(path)
    return DeviceRegistry.from_list([])


def resolve_deploy_targets(
    registry: DeviceRegistry,
    device: str | None,
    default_device: str = "",
) -> list[DeviceInfo]:
    """Resolve one or many deploy targets from CLI/agent device selector."""
    if device == "all":
        devices = registry.list_devices()
        if not devices:
            raise ValueError(
                "EDGE_DEVICES is empty. Add gateways to .env, e.g.\n"
                '  EDGE_DEVICES=[{"device_id":"gw1","host":"192.168.1.50","port":8080}]'
            )
        return devices

    device_id = device or default_device
    if not device_id:
        raise ValueError(
            "Provide --device, set EDGE_DEFAULT_DEVICE in .env, or use --host.\n"
            "Configure gateways in .env:\n"
            '  EDGE_DEVICES=[{"device_id":"gw1","host":"192.168.1.50","port":8080}]\n'
            "  EDGE_DEFAULT_DEVICE=gw1"
        )

    resolved = registry.resolve(device_id)
    if resolved is None:
        known = ", ".join(d.device_id for d in registry.list_devices()) or "(none)"
        raise ValueError(
            f"Unknown device {device_id!r}. Configured in EDGE_DEVICES: {known}"
        )
    return [resolved]
