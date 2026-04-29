from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings


def _restore_settings(snapshot: dict) -> None:
    for key, value in snapshot.items():
        setattr(settings, key, value)


def test_chat_returns_503_when_backend_unavailable() -> None:
    snapshot = settings.model_dump()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/chat", json={"message": "hello", "agent": "auto"})
            assert resp.status_code == 503
    finally:
        _restore_settings(snapshot)


def test_restart_approval_does_not_500_with_missing_server_path() -> None:
    snapshot = settings.model_dump()
    try:
        settings.require_approval_server_restart = True
        settings.llama_server_path = ""

        with TestClient(app, raise_server_exceptions=False) as client:
            queued = client.post("/api/server/restart")
            assert queued.status_code == 200
            approval_id = queued.json()["pending_approval"]["id"]

            approved = client.post(f"/api/approvals/{approval_id}/approve")
            assert approved.status_code == 200
            body = approved.json()
            assert body["approved"] is True
            assert body["action"] == "server_restart"
            assert "error" in body
    finally:
        _restore_settings(snapshot)


def test_emergency_stop_queues_when_approval_required() -> None:
    snapshot = settings.model_dump()
    try:
        settings.require_approval_emergency_stop = True
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/emergency-stop")
            assert resp.status_code == 200
            body = resp.json()
            assert "pending_approval" in body
            assert body["pending_approval"]["action"] == "emergency_stop"
    finally:
        _restore_settings(snapshot)
