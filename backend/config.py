from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    llama_host: str = "127.0.0.1"
    llama_port: int = 8080
    watcher_slot: int = 0
    main_slot: int = 1
    watcher_context_size: int = 4096
    main_context_size: int = 32768
    idle_kv_quant: str = "q4"
    active_kv_quant: str = "q8"
    lancedb_path: str = "./data/lancedb"
    embedding_model: str = "all-MiniLM-L6-v2"
    vram_threshold_mb: int = 2048
    mode: str = "idle"

    model_config = {"env_prefix": "SLOTHBRAIN_", "env_file": ".env", "extra": "ignore"}


settings = AppConfig()
