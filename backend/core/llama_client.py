from __future__ import annotations

import json

import httpx

from backend.config import settings


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
        completion_timeout = float(settings.llama_completion_timeout_seconds)
        # /completion can legitimately take minutes when prompt eval is large.
        self._completion_timeout = httpx.Timeout(
            timeout=None,
            connect=10.0,
            read=completion_timeout,
            write=60.0,
            pool=60.0,
        )
        # Keep lightweight probe calls snappy so diagnostics fail fast.
        self._probe_timeout = httpx.Timeout(10.0)

    # Pre-filled closed think block injected when the prompt ends with the
    # assistant response prefix.  Qwen3 / DeepSeek-R1 models in raw completion
    # mode interpret a complete <think>…</think> at the start of the assistant
    # turn as "thinking already done" and skip the chain-of-thought phase.
    # This prevents thinking tokens (which are NOT capped by n_predict) from
    # silently consuming minutes of inference time.
    _THINK_SKIP_SUFFIX = "<think>\n\n</think>\n\n"
    _ASSISTANT_PREFIXES = ("assistant:", "assistant:\n")

    async def complete(
        self,
        prompt: str,
        slot_id: int | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        if slot_id is None:
            slot_id = int(settings.main_slot)

        # Inject a pre-filled empty think block so thinking models skip CoT.
        _stripped = prompt.rstrip()
        if any(_stripped.endswith(p) for p in self._ASSISTANT_PREFIXES):
            prompt = _stripped + "\n" + self._THINK_SKIP_SUFFIX

        payload: dict = {
            "prompt": prompt,
            "id_slot": slot_id,
            "n_predict": max_tokens,
            "temperature": temperature,
            # Also pass thinking:false for llama.cpp builds that support it.
            "thinking": False,
        }
        if stop:
            payload["stop"] = stop

        async with httpx.AsyncClient(timeout=self._completion_timeout) as client:
            response = await client.post(f"{self.base_url}/completion", json=payload)
            if response.status_code >= 400:
                self._raise_completion_error(response)
            result = response.json()
        text = self._extract_text(result)
        if text.strip():
            return text

        # Compatibility fallback: some servers/models return empty output when
        # the optional thinking flag is present. Retry once without it.
        retry_payload = dict(payload)
        retry_payload.pop("thinking", None)
        async with httpx.AsyncClient(timeout=self._completion_timeout) as client:
            retry_response = await client.post(f"{self.base_url}/completion", json=retry_payload)
            if retry_response.status_code >= 400:
                self._raise_completion_error(retry_response)
            retry_result = retry_response.json()
        return self._extract_text(retry_result)

    @staticmethod
    def _extract_text(result: dict) -> str:
        # Handle both response shapes.
        if "choices" in result and result["choices"]:
            return result["choices"][0].get("text", "")
        return result.get("content", "")

    async def get_slots(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self._probe_timeout) as client:
            response = await client.get(f"{self.base_url}/slots")
            response.raise_for_status()
            return response.json()

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=self._probe_timeout) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def get_metrics(self) -> str:
        async with httpx.AsyncClient(timeout=self._probe_timeout) as client:
            response = await client.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.text

    def _raise_completion_error(self, response: httpx.Response) -> None:
        """Map llama.cpp completion errors to clearer exceptions.

        In particular, context-window overflow should be surfaced as a
        ValueError so callers can recover quickly (e.g. abort/reset) instead
        of repeatedly retrying oversized prompts.
        """
        message = response.text
        try:
            payload = response.json()
            if isinstance(payload, dict):
                message = str(payload.get("error") or payload.get("message") or message)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        lower = message.lower()
        if "exceeds the available context size" in lower or "context size" in lower:
            raise ValueError("llama.cpp context window exceeded")

        response.raise_for_status()
