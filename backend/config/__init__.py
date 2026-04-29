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
    vram_threshold_mb: int = 2048
    max_context_size: int = 131072
    max_slots: int = 8

    # Operating mode
    mode: str = "idle"

    # llama-server process management
    llama_server_path: str = ""
    llama_server_args: list[str] = []
    max_restarts_per_hour: int = 3

    # Approval gates – set to True to require human approval for that action
    require_approval_server_restart: bool = True
    require_approval_kv_cache_change: bool = True
    require_approval_large_context_increase: bool = True

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

    model_config = {"env_prefix": "SLOTHBRAIN_", "env_file": ".env", "extra": "ignore"}


settings = AppConfig()
