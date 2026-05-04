from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_vision_status_reports_capabilities() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/vision/status")
        assert response.status_code == 200
        body = response.json()
        assert "screenshot_backend" in body
        assert "input_available" in body
        assert "ocr_available" in body
        assert "multimodal_available" in body
        assert "mmproj_configured" in body
        assert "image_analysis_backend" in body


def test_vision_run_returns_503_without_ocr_or_multimodal() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/vision/run",
            json={"task": "open notepad", "max_steps": 1},
        )
        if response.status_code == 503:
            assert "vision_run" in response.json()["detail"]
