from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    # llama.cpp connection
    llama_host: str = "127.0.0.1"
    llama_port: int = 8080

    # Slot assignments
    watcher_slot: int = 0
    main_slot: int = 1

    # Context sizes
    watcher_context_size: int = 4096
    main_context_size: int = 32768

    # KV cache quantisation
    idle_kv_quant: str = "q4"
    active_kv_quant: str = "q8"

    # Memory / embedding
    lancedb_path: str = "./data/lancedb"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Resource limits
    # Deprecated name kept for backwards compatibility.
    vram_threshold_mb: int = 2048
    # Preferred name (system RAM threshold used for idle-mode fallback).
    ram_threshold_mb: int = 2048
    max_context_size: int = 131072
    max_slots: int = 8
    max_pending_approvals: int = 500

    # Operating mode
    mode: str = "idle"

    # llama-server process management
    llama_server_path: str = ""
    llama_server_args: list[str] = []
    max_restarts_per_hour: int = 3
    enable_server_watchdog: bool = True

    # API surface hardening:
    # - If api_key is set, all API/WS calls must present it.
    # - If api_key is empty, only loopback clients may access API/WS routes.
    api_key: str = ""

    # Approval gates – set to True to require human approval for that action
    require_approval_server_restart: bool = True
    require_approval_kv_cache_change: bool = True
    require_approval_large_context_increase: bool = True
    require_approval_emergency_stop: bool = True

    # Safety supervisor settings
    supervisor_poll_interval: float = 15.0   # seconds between supervision polls
    supervisor_step_timeout: float = 120.0   # seconds before a step is declared stalled

    # Tool system settings
    # Root directory that file/patch/diff tools are confined to.
    tool_workspace_root: str = "./workspace"
    # Allowlisted command prefixes for the shell/process tools.
    # An empty list disables the shell tool unless allow_unrestricted_shell is True.
    shell_allowlist: list[str] = []
    # When True, the shell and process tools accept any command (no allowlist check).
    # Enable only in trusted local environments.
    allow_unrestricted_shell: bool = False
    # Default tool profile for the main agent ("full" gives access to every tool).
    main_tool_profile: str = "full"

    # Discord integration (optional – leave empty to disable DiscordTool)
    discord_webhook_url: str = ""
    discord_bot_token: str = ""
    discord_channel_id: str = ""

    # Web search: set to a SearXNG base URL to use it instead of DuckDuckGo
    searxng_url: str = ""

    cors_allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("max_context_size")
    @classmethod
    def _validate_max_context(cls, v: int) -> int:
        if v < 512:
            raise ValueError("max_context_size must be at least 512")
        return v

    @field_validator("max_slots")
    @classmethod
    def _validate_max_slots(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_slots must be at least 1")
        return v

    @field_validator("max_restarts_per_hour")
    @classmethod
    def _validate_max_restarts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_restarts_per_hour must be at least 1")
        return v

    @field_validator("max_pending_approvals")
    @classmethod
    def _validate_max_pending_approvals(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_pending_approvals must be at least 1")
        return v

    model_config = {"env_prefix": "SLOTHBRAIN_", "env_file": ".env", "extra": "ignore"}


settings = AppConfig()
