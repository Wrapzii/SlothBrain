import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("health", get("/health"))
    tests = [
        ("hello", {"message": "Hello", "mode": "auto", "max_steps": 10}),
        (
            "find_github_dir",
            {"message": "/task Find my github directory and list projects in it", "mode": "auto", "max_steps": 10},
        ),
        (
            "create_file",
            {"message": "/task Create a file named smoke_test.txt in the github directory with the text hello from slothbrain", "mode": "auto", "max_steps": 10},
        ),
        (
            "read_file",
            {"message": "/task Read the file smoke_test.txt from the github directory", "mode": "auto", "max_steps": 10},
        ),
        (
            "append_file",
            {"message": "/task Append a new line saying second line to smoke_test.txt in the github directory", "mode": "auto", "max_steps": 10},
        ),
        (
            "research",
            {"message": "/task Research sun hydraulics llc and provide sources", "mode": "auto", "max_steps": 10},
        ),
        (
            "sub_agent",
            {"message": "/task Use a sub agent to summarize what this repository is for", "mode": "auto", "max_steps": 10},
        ),
    ]
    for name, payload in tests:
        print("\n===", name, "===")
        started = time.time()
        try:
            result = post("/api/chat", payload)
            print(json.dumps({
                "agent": result.get("agent"),
                "response": result.get("response"),
                "duration": round(time.time() - started, 2),
            }, indent=2))
        except Exception as exc:
            print(json.dumps({"error": str(exc), "duration": round(time.time() - started, 2)}, indent=2))


if __name__ == "__main__":
    main()
