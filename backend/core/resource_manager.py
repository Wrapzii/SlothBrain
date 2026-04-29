from __future__ import annotations

import psutil

from backend.config import AppConfig
from backend.core.llama_client import LlamaClient


class ResourceManager:
    """Manages operating mode (idle/active) and system resource monitoring.

    The resource manager controls the KV-cache quantisation level used by
    llama.cpp and can automatically switch to idle mode when system RAM
    exceeds a configurable threshold.

    Modes
    -----
    idle
        Uses ``idle_kv_quant`` (default ``q4``).  Lower memory footprint,
        suitable for background / light workloads.
    active
        Uses ``active_kv_quant`` (default ``q8``).  Higher quality outputs,
        suitable for demanding tasks.

    TODO: Expose GPU VRAM monitoring (nvidia-smi / nvml) in addition to
          system RAM so the threshold can be driven by actual VRAM pressure.
    """

    def __init__(self, config: AppConfig, llama_client: LlamaClient) -> None:
        self._config = config
        self._client = llama_client
        self._mode: str = config.mode

    @property
    def mode(self) -> str:
        return self._mode

    async def set_mode(self, mode: str) -> None:
        if mode not in ("idle", "active"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'idle' or 'active'.")
        self._mode = mode
        self._config.mode = mode

    async def get_system_stats(self) -> dict:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        ram_used_mb = ram.used / (1024 * 1024)
        ram_total_mb = ram.total / (1024 * 1024)
        return {
            "cpu_percent": cpu_percent,
            "ram_used_mb": round(ram_used_mb, 1),
            "ram_total_mb": round(ram_total_mb, 1),
            "mode": self._mode,
        }

    async def auto_adjust(self) -> None:
        stats = await self.get_system_stats()
        threshold_mb = getattr(self._config, "ram_threshold_mb", self._config.vram_threshold_mb)
        if stats["ram_used_mb"] > threshold_mb:
            await self.set_mode("idle")
        elif self._mode == "active":
            # Already active — keep it
            pass

    async def get_kv_quant(self) -> str:
        if self._mode == "idle":
            return self._config.idle_kv_quant
        return self._config.active_kv_quant
