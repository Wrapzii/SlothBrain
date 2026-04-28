from __future__ import annotations

import asyncio
import logging
import time

from backend.config import AppConfig
from backend.core.llama_client import LlamaClient

logger = logging.getLogger(__name__)


class BenchmarkSuite:
    def __init__(self, llama_client: LlamaClient, config: AppConfig) -> None:
        self._client = llama_client
        self._config = config

    async def run_inference_speed(
        self, context_lengths: list[int] | None = None
    ) -> list[dict]:
        if context_lengths is None:
            context_lengths = [512, 1024, 2048, 4096]

        results: list[dict] = []
        for length in context_lengths:
            prompt = "Write a detailed technical explanation. " * (length // 8)
            prompt = prompt[:length]
            try:
                start = time.perf_counter()
                response = await self._client.complete(
                    prompt=prompt,
                    slot_id=self._config.main_slot,
                    max_tokens=128,
                    temperature=0.0,
                )
                elapsed = time.perf_counter() - start
                token_count = max(len(response.split()), 1)
                tokens_per_sec = token_count / elapsed if elapsed > 0 else 0
                results.append({
                    "context_length": length,
                    "elapsed_seconds": round(elapsed, 3),
                    "output_tokens": token_count,
                    "tokens_per_sec": round(tokens_per_sec, 2),
                    "status": "ok",
                })
            except Exception as exc:
                logger.error("Inference speed benchmark failed for length %d: %s", length, exc)
                results.append({
                    "context_length": length,
                    "elapsed_seconds": None,
                    "output_tokens": None,
                    "tokens_per_sec": None,
                    "status": "error",
                })

        return results

    async def run_vram_benchmark(self) -> dict:
        try:
            metrics_text = await self._client.get_metrics()
            vram_lines = [
                line for line in metrics_text.splitlines()
                if "vram" in line.lower() or "gpu" in line.lower()
            ]
            return {
                "current_kv_quant": self._config.active_kv_quant
                if self._config.mode == "active"
                else self._config.idle_kv_quant,
                "mode": self._config.mode,
                "metrics_excerpt": vram_lines[:10],
                "status": "ok",
            }
        except Exception as exc:
            logger.error("VRAM benchmark failed: %s", exc)
            return {"status": "error"}

    async def run_slot_interference(self) -> dict:
        prompt = "Explain the concept of neural networks in detail."
        try:
            start = time.perf_counter()
            watcher_task = self._client.complete(
                prompt=prompt,
                slot_id=self._config.watcher_slot,
                max_tokens=128,
            )
            main_task = self._client.complete(
                prompt=prompt,
                slot_id=self._config.main_slot,
                max_tokens=128,
            )
            results = await asyncio.gather(
                watcher_task, main_task, return_exceptions=True
            )
            elapsed = time.perf_counter() - start
            watcher_ok = not isinstance(results[0], Exception)
            main_ok = not isinstance(results[1], Exception)
            if not watcher_ok:
                logger.error("Slot interference watcher request failed: %s", results[0])
            if not main_ok:
                logger.error("Slot interference main request failed: %s", results[1])
            return {
                "total_elapsed_seconds": round(elapsed, 3),
                "watcher_status": "ok" if watcher_ok else "error",
                "main_status": "ok" if main_ok else "error",
                "interference_detected": elapsed > 30,
                "status": "ok",
            }
        except Exception as exc:
            logger.error("Slot interference benchmark failed: %s", exc)
            return {"status": "error"}

    async def run_all(self) -> dict:
        speed_task = self.run_inference_speed()
        vram_task = self.run_vram_benchmark()
        slot_task = self.run_slot_interference()
        speed, vram, slots = await asyncio.gather(
            speed_task, vram_task, slot_task, return_exceptions=True
        )
        if isinstance(speed, Exception):
            logger.error("run_all speed benchmark failed: %s", speed)
        if isinstance(vram, Exception):
            logger.error("run_all vram benchmark failed: %s", vram)
        if isinstance(slots, Exception):
            logger.error("run_all slot benchmark failed: %s", slots)
        return {
            "inference_speed": speed if not isinstance(speed, Exception) else {"status": "error"},
            "vram": vram if not isinstance(vram, Exception) else {"status": "error"},
            "slot_interference": slots if not isinstance(slots, Exception) else {"status": "error"},
        }
