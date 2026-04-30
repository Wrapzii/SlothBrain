"""Agent preset manager — CRUD for agent presets stored as JSON files."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

PRESETS_DIR = Path("data/agent_presets")

_REQUIRED_FIELDS = {"name", "system_prompt", "context_size", "temperature", "max_tokens"}


class PresetManager:
    def __init__(self, presets_dir: Path | str = PRESETS_DIR) -> None:
        self._dir = Path(presets_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, preset_id: str) -> Path:
        # Sanitise to prevent directory traversal
        safe_id = Path(preset_id).name
        return self._dir / f"{safe_id}.json"

    def _load(self, preset_id: str) -> dict:
        path = self._path(preset_id)
        if not path.exists():
            raise KeyError(f"Preset not found: {preset_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        path = self._path(data["id"])
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_presets(self) -> list[dict]:
        presets = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                presets.append(json.loads(p.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return presets

    def get_preset(self, preset_id: str) -> dict:
        return self._load(preset_id)

    def create_preset(self, data: dict[str, Any]) -> dict:
        missing = _REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        preset: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "name": str(data["name"]),
            "description": str(data.get("description", "")),
            "system_prompt": str(data["system_prompt"]),
            "context_size": int(data["context_size"]),
            "temperature": float(data["temperature"]),
            "max_tokens": int(data["max_tokens"]),
            "tool_profile": str(data.get("tool_profile", "minimal")),
        }
        self._save(preset)
        return preset

    def update_preset(self, preset_id: str, data: dict[str, Any]) -> dict:
        existing = self._load(preset_id)
        updatable = {"name", "description", "system_prompt", "context_size", "temperature", "max_tokens", "tool_profile"}
        for field in updatable:
            if field in data:
                existing[field] = data[field]
        self._save(existing)
        return existing

    def delete_preset(self, preset_id: str) -> None:
        path = self._path(preset_id)
        if not path.exists():
            raise KeyError(f"Preset not found: {preset_id!r}")
        path.unlink()
