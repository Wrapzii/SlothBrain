"""Start SlothBrain backend + Python TUI in one command.

Usage:
    python run_slothbrain.py

Optional flags:
    python run_slothbrain.py --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil


def _verify_required_runtime_dependencies() -> None:
    """Fail fast when required LanceDB dependencies are broken or missing."""
    try:
        import lancedb  # noqa: F401
        from backend.memory.lancedb_memory import LanceDBMemory  # noqa: F401
        from backend.memory.workspace_indexer import WorkspaceIndexer  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Required LanceDB runtime dependencies are unavailable. "
            "Install pinned requirements with: python -m pip install -r requirements.txt. "
            f"Original error: {exc}"
        ) from exc


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


def _find_backend_pid_on_port(host: str, port: int) -> int | None:
    """Return a PID if a SlothBrain uvicorn backend is listening on host:port."""
    for conn in psutil.net_connections(kind="tcp"):
        if conn.status != psutil.CONN_LISTEN:
            continue
        if not conn.laddr:
            continue
        lhost, lport = conn.laddr.ip, conn.laddr.port
        if lport != port:
            continue
        if host in ("127.0.0.1", "localhost") and lhost not in ("127.0.0.1", "0.0.0.0", "::", "::1"):
            continue
        if host not in ("127.0.0.1", "localhost") and lhost != host:
            continue
        pid = conn.pid
        if not pid:
            continue
        try:
            p = psutil.Process(pid)
            cmd = " ".join(p.cmdline()).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "uvicorn" in cmd and "backend.main:app" in cmd:
            return pid
    return None


def _stop_pid_gracefully(pid: int, timeout_s: float = 10.0) -> bool:
    """Try graceful terminate, then force kill. Returns True when process exits."""
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True

    p.terminate()
    try:
        p.wait(timeout=timeout_s)
        return True
    except psutil.TimeoutExpired:
        try:
            p.kill()
            p.wait(timeout=5)
            return True
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            return False


def _reclaim_backend_port_if_needed(host: str, port: int) -> None:
    """Stop stale SlothBrain backend on host:port so launcher can own lifecycle."""
    pid = _find_backend_pid_on_port(host, port)
    if pid is None:
        return
    print(f"[launcher] Found existing SlothBrain backend on {host}:{port} (pid={pid}); stopping it...")
    if not _stop_pid_gracefully(pid):
        raise RuntimeError(
            f"Could not stop existing backend process on {host}:{port} (pid={pid})."
        )


def _cleanup_backend_port(host: str, port: int, launched_pid: int | None) -> None:
    """Ensure no stale SlothBrain backend remains on host:port after exit."""
    pid = _find_backend_pid_on_port(host, port)
    if pid is None:
        return
    # Prefer cleaning up only the process launched by this run. If something else
    # matching SlothBrain backend remains on the same port, clean it too.
    if launched_pid is not None and pid != launched_pid:
        print(
            f"[launcher] Warning: backend pid on port differs (expected {launched_pid}, found {pid}); attempting cleanup."
        )
    _stop_pid_gracefully(pid)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start SlothBrain backend and TUI")
    parser.add_argument("--host", default="127.0.0.1", help="Backend host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn --reload")
    args = parser.parse_args()

    try:
        _verify_required_runtime_dependencies()
    except RuntimeError as exc:
        print(f"[launcher] {exc}")
        return 1

    repo_root = Path(__file__).resolve().parent

    # Make launcher the owner of backend lifecycle by reclaiming stale backend
    # listeners on the requested port before starting a new instance.
    _reclaim_backend_port_if_needed(args.host, args.port)

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
    backend_proc = subprocess.Popen(cmd, cwd=str(repo_root), env=os.environ.copy())

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
        _cleanup_backend_port(args.host, args.port, backend_proc.pid)


if __name__ == "__main__":
    raise SystemExit(main())
