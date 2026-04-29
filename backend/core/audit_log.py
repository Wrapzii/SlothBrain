"""Append-only audit log for self-modification actions."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AUDIT_LOG_PATH = Path("data/audit.log")


class AuditLog:
    def __init__(self, log_path: Path | str = AUDIT_LOG_PATH) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        action: str,
        actor: str = "system",
        before: Any = None,
        after: Any = None,
        details: str = "",
    ) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "before": before,
            "after": after,
            "details": details,
        }
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error("Failed to write audit log: %s", exc)
        return entry

    def tail(self, n: int = 100) -> list[dict]:
        if not self._path.exists():
            return []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
