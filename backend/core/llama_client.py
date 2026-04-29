from __future__ import annotations

import httpx


class LlamaClient:
    """Async HTTP client for the llama.cpp inference server REST API.

    All methods open a new ``httpx.AsyncClient`` per call.  This is safe
    but slightly less efficient than a persistent connection pool.

    TODO: Replace per-call AsyncClient creation with a persistent client
          (created in ``__init__``, closed on shutdown) to benefit from
          HTTP connection pooling. See BUGS.md BUG-010.
    TODO: Add configurable retry with exponential back-off for transient
          errors (connection refused, 503 Service Unavailable).
    """

    def __init__(self, host: str, port: int) -> None:
        self.base_url = f"http://{host}:{port}"
        self._timeout = httpx.Timeout(120.0)

    async def complete(
        self,
        prompt: str,
        slot_id: int,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict = {
            "prompt": prompt,
            "id_slot": slot_id,
            "n_predict": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(f"{self.base_url}/completion", json=payload)
            response.raise_for_status()
            result = response.json()

        # Handle both response shapes
        if "choices" in result and result["choices"]:
            return result["choices"][0].get("text", "")
        return result.get("content", "")

    async def get_slots(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self.base_url}/slots")
            response.raise_for_status()
            return response.json()

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def get_metrics(self) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.text
