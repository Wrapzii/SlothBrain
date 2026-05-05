from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings


_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class AppConfig(BaseSettings):
    # llama.cpp connection
    llama_host: str = "127.0.0.1"
    llama_port: int = 8080

    # Slot assignments
    main_slot: int = 1

    # Context sizes
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
    # Maximum number of concurrent inference tasks allowed by the API layer.
    # Set to 2+ when running llama.cpp with multiple parallel slots/workers.
    inference_concurrency: int = 1

    # Operating mode
    mode: str = "idle"

    # llama-server process management
    llama_server_path: str = ""
    llama_server_args: list[str] = []
    # Saved launch profiles shown as UI cards in the main app settings.
    # Each item: {"id": str, "name": str, "command": str}
    llama_launch_profiles: list[dict] = []
    # Profile id from llama_launch_profiles used as startup default.
    default_launch_profile_id: str = ""
    # Optional hard cap for effective per-slot context budget.
    # Set to 0 to disable and rely on /slots metadata.
    llama_slot_context_cap: int = 0
    # Cache /slots responses for this many seconds to reduce llama.cpp log spam.
    slot_info_cache_ttl_seconds: float = 5.0
    max_restarts_per_hour: int = 3
    enable_server_watchdog: bool = True

    # API surface hardening:
    # - If api_key is set, all API/WS calls must present it.
    # - If api_key is empty, only loopback clients may access API/WS routes.
    api_key: str = ""

    # Approval gates – set to True to require human approval for that action
    require_approval_server_restart: bool = True
    # Allow automatic health-recovery restarts to bypass manual approval.
    # This keeps chat/tool flows resilient when llama.cpp drops unexpectedly.
    allow_auto_recovery_restart_without_approval: bool = True
    require_approval_kv_cache_change: bool = True
    require_approval_large_context_increase: bool = True
    require_approval_emergency_stop: bool = True

    # Safety supervisor settings
    supervisor_poll_interval: float = 15.0   # seconds between supervision polls
    supervisor_step_timeout: float = 120.0   # seconds before a step is declared stalled
    supervisor_slowdown_monitor_enabled: bool = True
    supervisor_slowdown_threshold_tps: float = 20.0
    supervisor_slowdown_consecutive_polls: int = 3
    supervisor_slowdown_restart_enabled: bool = False
    supervisor_slowdown_cooldown_seconds: float = 300.0
    # Timeout budget (seconds) for llama.cpp /completion requests.
    # Keep this comfortably above worst-case prompt-eval + generation times to
    # avoid client-side cancellations that invalidate llama.cpp slot checkpoints.
    llama_completion_timeout_seconds: float = 600.0
    # Discord DM per-message processing timeout (seconds).
    # 0 disables timeout and lets the request run to completion.
    discord_dm_timeout_seconds: float = -1.0
    supervisor_max_repeated_tool_calls: int = 3
    supervisor_max_failed_tool_calls: int = 3
    supervisor_max_no_progress_steps: int = 3
    supervisor_max_empty_responses: int = 2
    supervisor_max_give_up_signals: int = 1

    # Tool system settings
    # Root directory that file/patch/diff tools are confined to.
    tool_workspace_root: str = "./workspace"
    # When True, FileTool also accepts absolute paths outside tool_workspace_root.
    # Intended for trusted local usage (for example: Desktop/Documents inspection).
    file_tool_allow_absolute_paths: bool = True
    # When False, desktop UI / screenshot style tools are not registered.
    desktop_tools_enabled: bool = True
    # Allowlisted command prefixes for the shell/process tools.
    # An empty list disables the shell tool unless allow_unrestricted_shell is True.
    shell_allowlist: list[str] = []
    # When True, the shell and process tools accept any command (no allowlist check).
    # Enable only in trusted local environments.
    allow_unrestricted_shell: bool = False
    # When True, the code_exec tool is available. Disabled by default because
    # exec() is not a full sandbox and should only be used in trusted environments.
    code_exec_enabled: bool = False
    # Semantic tool routing
    semantic_tool_routing_enabled: bool = True
    semantic_tool_routing_embedding_model: str = ""
    semantic_tool_routing_top_k: int = 8
    semantic_tool_routing_min_similarity: float = 0.2
    semantic_tool_routing_critical_tools: list[str] = ["screenshot", "ui"]

    # Discord integration (optional – leave empty to disable DiscordTool)
    discord_webhook_url: str = ""
    discord_bot_token: str = ""
    discord_channel_id: str = ""
    # If set, the bot will open a DM channel with this user ID at startup
    # and use that channel for send/receive instead of discord_channel_id.
    discord_owner_user_id: str = ""

    # Scheduler defaults
    scheduler_timezone: str = "America/New_York"

    # Web search: set to a SearXNG base URL to use it instead of DuckDuckGo
    searxng_url: str = ""

    # Image analysis backend
    # - cpu_ocr: run OCR + metadata extraction on CPU (no llama.cpp vision call)
    # - llama: force llama.cpp multimodal analysis
    # - auto: use llama.cpp vision when available, then CPU OCR fallback
    image_analysis_backend: str = "auto"
    # Use a dedicated slot for llama.cpp vision calls to avoid polluting the
    # main assistant slot cache state. Set to -1 for "any available slot".
    image_analysis_llama_slot_id: int = 0
    image_analysis_cpu_max_text_chars: int = 4000

    # Workspace / file-system indexing
    # When True, WorkspaceIndexTool is registered and FileTool auto-triggers
    # background indexing whenever a directory is listed.
    workspace_index_enabled: bool = True
    # Set to a custom path to store the workspace index in a separate location
    # from the main LanceDB memory.  Defaults to a sub-directory of lancedb_path.
    workspace_index_db_path: str = ""

    # Offline agentic failure diagnostics
    # When enabled, every agentic run writes a structured JSON bundle to
    # diagnostics_output_dir/{run_id}/bundle.json for offline analysis.
    diagnostics_enabled: bool = False
    diagnostics_output_dir: str = "diagnostics/runs"

    # Agentic debug loop defaults
    # Enable to run the loop with runtime-tunable feature toggles and loop telemetry.
    debug_loop_enabled: bool = False
    debug_loop_llm_only: bool = False
    debug_loop_planning_enabled: bool = True
    debug_loop_rolling_context_enabled: bool = True
    debug_loop_tool_calls_enabled: bool = True
    debug_loop_semantic_routing_enabled: bool = True
    debug_loop_checkpointing_enabled: bool = True
    debug_loop_supervisor_enabled: bool = True
    debug_loop_per_event_logging: bool = True
    debug_loop_allowed_tools: list[str] = []

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

    @field_validator("llama_slot_context_cap")
    @classmethod
    def _validate_slot_context_cap(cls, v: int) -> int:
        if v < 0:
            raise ValueError("llama_slot_context_cap must be >= 0")
        return v

    @field_validator("slot_info_cache_ttl_seconds")
    @classmethod
    def _validate_slot_info_cache_ttl_seconds(cls, v: float) -> float:
        if v < 0:
            raise ValueError("slot_info_cache_ttl_seconds must be >= 0")
        return v

    @field_validator("max_pending_approvals")
    @classmethod
    def _validate_max_pending_approvals(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_pending_approvals must be at least 1")
        return v

    @field_validator("inference_concurrency")
    @classmethod
    def _validate_inference_concurrency(cls, v: int) -> int:
        if v < 1:
            raise ValueError("inference_concurrency must be at least 1")
        return v

    @field_validator("supervisor_slowdown_threshold_tps")
    @classmethod
    def _validate_slowdown_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("supervisor_slowdown_threshold_tps must be > 0")
        return v

    @field_validator("supervisor_slowdown_consecutive_polls")
    @classmethod
    def _validate_slowdown_polls(cls, v: int) -> int:
        if v < 1:
            raise ValueError("supervisor_slowdown_consecutive_polls must be at least 1")
        return v

    @field_validator("supervisor_slowdown_cooldown_seconds")
    @classmethod
    def _validate_slowdown_cooldown(cls, v: float) -> float:
        if v < 0:
            raise ValueError("supervisor_slowdown_cooldown_seconds must be >= 0")
        return v

    @field_validator("llama_completion_timeout_seconds")
    @classmethod
    def _validate_llama_completion_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("llama_completion_timeout_seconds must be > 0")
        return v

    @field_validator("discord_dm_timeout_seconds")
    @classmethod
    def _validate_discord_dm_timeout(cls, v: float) -> float:
        if v < -1:
            raise ValueError("discord_dm_timeout_seconds must be >= -1")
        return v

    @field_validator(
        "supervisor_max_repeated_tool_calls",
        "supervisor_max_failed_tool_calls",
        "supervisor_max_no_progress_steps",
        "supervisor_max_empty_responses",
        "supervisor_max_give_up_signals",
    )
    @classmethod
    def _validate_supervisor_positive_thresholds(cls, v: int) -> int:
        if v < 1:
            raise ValueError("supervisor detection thresholds must be at least 1")
        return v

    @field_validator("semantic_tool_routing_top_k")
    @classmethod
    def _validate_semantic_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("semantic_tool_routing_top_k must be at least 1")
        return v

    @field_validator("semantic_tool_routing_min_similarity")
    @classmethod
    def _validate_semantic_similarity(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError("semantic_tool_routing_min_similarity must be between -1 and 1")
        return v

    @field_validator("image_analysis_backend")
    @classmethod
    def _validate_image_analysis_backend(cls, v: str) -> str:
        allowed = {"cpu_ocr", "llama", "auto"}
        norm = (v or "").strip().lower()
        if norm not in allowed:
            raise ValueError("image_analysis_backend must be one of: cpu_ocr, llama, auto")
        return norm

    @field_validator("image_analysis_cpu_max_text_chars")
    @classmethod
    def _validate_image_analysis_cpu_max_text_chars(cls, v: int) -> int:
        if v < 256:
            raise ValueError("image_analysis_cpu_max_text_chars must be at least 256")
        return v

    model_config = {"env_prefix": "SLOTHBRAIN_", "env_file": str(_ENV_FILE), "extra": "ignore"}


settings = AppConfig()
