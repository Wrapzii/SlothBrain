from fastapi.testclient import TestClient

from backend.main import app


CASES = [
    ("desktop", "/api/chat/direct", {"message": "what's on my desktop?", "mode": "direct"}),
    ("bytebrew", "/api/chat/direct", {"message": "tell me about bytebrew", "mode": "direct"}),
    ("files", "/api/chat/direct", {"message": "check files in backend/agents", "mode": "direct"}),
    ("status", "/api/chat", {"message": "/status", "mode": "auto"}),
]


def main() -> None:
    with TestClient(app) as client:
        for name, path, payload in CASES:
            response = client.post(path, json=payload)
            text = response.text.replace("\n", " ")
            if len(text) > 320:
                text = text[:320]
            print(f"CASE={name} STATUS={response.status_code}")
            print(f"RESP={text}")
            print("---")


if __name__ == "__main__":
    main()
