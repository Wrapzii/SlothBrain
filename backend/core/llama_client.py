from __future__ import annotations

import asyncio
import json
from typing import Any

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
    _MAX_COMPLETION_RETRIES = 2
    _RETRY_BASE_DELAY_SECONDS = 0.35
    _RECOVERY_HEALTH_WAIT_SECONDS = 25.0
    _RECOVERY_HEALTH_POLL_SECONDS = 1.0

    @staticmethod
    def _is_transient_completion_error(exc: Exception) -> bool:
        return isinstance(exc, (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError))

    async def _wait_for_server_recovery(self, timeout_seconds: float) -> bool:
        """Poll /health until llama.cpp becomes reachable again.

        This allows in-flight completion calls to survive short server restarts
        (manual restart, watchdog restart, model reload) instead of failing
        immediately on a transient ConnectError.
        """
        deadline = asyncio.get_running_loop().time() + max(0.0, float(timeout_seconds))
        while asyncio.get_running_loop().time() < deadline:
            try:
                await self.health()
                return True
            except Exception:
                await asyncio.sleep(self._RECOVERY_HEALTH_POLL_SECONDS)
        return False

    async def _post_completion(self, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._MAX_COMPLETION_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self._completion_timeout) as client:
                    response = await client.post(f"{self.base_url}/completion", json=payload)
                if response.status_code >= 400:
                    self._raise_completion_error(response)
                return response.json()
            except Exception as exc:
                if not self._is_transient_completion_error(exc) or attempt >= self._MAX_COMPLETION_RETRIES:
                    raise
                last_exc = exc
                # If llama.cpp temporarily crashed/restarted, wait for /health
                # before retrying so callers recover automatically.
                await self._wait_for_server_recovery(self._RECOVERY_HEALTH_WAIT_SECONDS)
                await asyncio.sleep(self._RETRY_BASE_DELAY_SECONDS * (2 ** attempt))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Completion call failed without an exception")

    async def _post_chat_completion(self, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._MAX_COMPLETION_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self._completion_timeout) as client:
                    response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                if response.status_code >= 400:
                    self._raise_completion_error(response)
                return response.json()
            except Exception as exc:
                if not self._is_transient_completion_error(exc) or attempt >= self._MAX_COMPLETION_RETRIES:
                    raise
                last_exc = exc
                await self._wait_for_server_recovery(self._RECOVERY_HEALTH_WAIT_SECONDS)
                await asyncio.sleep(self._RETRY_BASE_DELAY_SECONDS * (2 ** attempt))

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Chat completion call failed without an exception")

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

        result = await self._post_completion(payload)
        text = self._extract_text(result)
        if text.strip():
            return text

        # Compatibility fallback: some servers/models return empty output when
        # the optional thinking flag is present. Retry once without it.
        retry_payload = dict(payload)
        retry_payload.pop("thinking", None)
        retry_result = await self._post_completion(retry_payload)
        return self._extract_text(retry_result)

    async def chat_complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        *,
        model: str = "local",
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Run an OpenAI-compatible llama.cpp chat completion.

        This path is required for multimodal input on current llama.cpp
        servers: images are passed as chat content parts rather than embedded
        into the legacy raw ``/completion`` prompt.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Supported by some llama.cpp builds/templates; harmless to retry
            # without it if a model/template responds with empty content.
            "thinking": False,
        }
        if extra:
            payload.update(extra)

        result = await self._post_chat_completion(payload)
        text = self._extract_chat_text(result)
        if text.strip():
            return text

        retry_payload = dict(payload)
        retry_payload.pop("thinking", None)
        retry_result = await self._post_chat_completion(retry_payload)
        return self._extract_chat_text(retry_result)

    async def complete_with_image(
        self,
        prompt: str,
        image_b64: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        *,
        mime_type: str = "image/png",
    ) -> str:
        image_payload = (image_b64 or "").strip()
        if image_payload.startswith("data:image"):
            image_url = image_payload
        else:
            image_url = f"data:{mime_type};base64,{image_payload}"

        return await self.chat_complete(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @staticmethod
    def _extract_text(result: dict) -> str:
        # Handle both response shapes.
        if "choices" in result and result["choices"]:
            return result["choices"][0].get("text", "")
        return result.get("content", "")

    @staticmethod
    def _extract_chat_text(result: dict) -> str:
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return ""

        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
            if parts:
                return "\n".join(parts)

        # Several thinking-capable local models emit useful multimodal
        # observations in reasoning_content even when visible content is empty.
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str):
            return reasoning.strip()
        return ""

    async def get_slots(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self._probe_timeout) as client:
            response = await client.get(f"{self.base_url}/slots")
            response.raise_for_status()
            return response.json()

    async def get_props(self) -> dict:
        async with httpx.AsyncClient(timeout=self._probe_timeout) as client:
            response = await client.get(f"{self.base_url}/props")
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

        if response.status_code >= 500 or response.status_code == 429:
            msg = message.strip() or response.text.strip() or response.reason_phrase
            raise RuntimeError(
                f"llama.cpp completion HTTP {response.status_code}: {msg[:280]}"
            )

        response.raise_for_status()
