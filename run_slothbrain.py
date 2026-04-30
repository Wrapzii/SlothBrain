"""Start SlothBrain backend + Python TUI in one command.

Usage:
    python run_slothbrain.py

Optional flags:
    python run_slothbrain.py --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _wait_for_backend(host: str, port: int, timeout_s: float = 30.0) -> bool:
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.3)
    return False


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start SlothBrain backend and TUI")
    parser.add_argument("--host", default="127.0.0.1", help="Backend host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn --reload")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    # WebSocket buffer sizes for large screenshot payloads (up to 2MB base64)
    # Default 1MB was causing "message too big" errors during agent vision tasks
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ws-max-size",
        "16777216",  # 16 MB
        "--ws-max-queue",
        "32",
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"[launcher] Starting backend: {' '.join(cmd)}")
    backend_proc = subprocess.Popen(cmd, cwd=str(repo_root))

    try:
        if not _wait_for_backend(args.host, args.port):
            print("[launcher] Backend failed to become healthy in time.")
            return 1

        print("[launcher] Backend is healthy. Starting TUI...")
        from tui.app import main as tui_main

        tui_main()
        return 0
    except KeyboardInterrupt:
        print("\n[launcher] Interrupted by user.")
        return 0
    finally:
        print("[launcher] Stopping backend...")
        _terminate_process(backend_proc)


if __name__ == "__main__":
    raise SystemExit(main())
