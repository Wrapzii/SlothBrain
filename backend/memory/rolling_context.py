from __future__ import annotations

import asyncio

from backend.core.llama_client import LlamaClient


class RollingContext:
    """Rolling conversation buffer with automatic LLM-based summarisation.

    Accumulates ``{role, content}`` message dicts and estimates the total
    token count using a ``len(content) // 4`` heuristic.  When the estimate
    exceeds ``summarize_at`` the entire conversation is summarised by calling
    the LLM and the buffer is replaced with the summary.

    This keeps the watcher's context window from overflowing on long sessions.

    Parameters
    ----------
    llama_client:
        Used to call the LLM for summarisation.
    slot_id:
        The inference slot to use for summarisation calls.
    max_tokens:
        Context window size in tokens (used as an upper bound reference).
    summarize_at:
        Token estimate threshold that triggers summarisation (default 3000).

    TODO: Run summarisation as a background task so it does not block
          ``add_message`` callers. See BUGS.md BUG-002.
    TODO: Use a proper tokeniser (tiktoken or the llama.cpp /tokenize
          endpoint) instead of the ``len // 4`` heuristic. See BUGS.md BUG-012.
    """

    def __init__(
        self,
        llama_client: LlamaClient,
        slot_id: int,
        max_tokens: int = 4096,
        summarize_at: int = 3000,
        summary_timeout_seconds: float = 0.25,
        min_messages_before_resummarize: int = 4,
    ) -> None:
        self._client = llama_client
        self._slot_id = slot_id
        self.max_tokens = max_tokens
        self.summarize_at = summarize_at
        self.summary_timeout_seconds = max(0.1, float(summary_timeout_seconds))
        self.min_messages_before_resummarize = max(1, int(min_messages_before_resummarize))
        self.messages: list[dict] = []
        self._messages_since_summary = 0
        self._has_summary = False

    @property
    def token_estimate(self) -> int:
        return sum(len(m["content"]) // 4 for m in self.messages)

    async def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._messages_since_summary += 1
        should_summarize = self._summarization_estimate > self.summarize_at
        if should_summarize and self._has_summary:
            should_summarize = (
                self._messages_since_summary >= self.min_messages_before_resummarize
            )

        if should_summarize:
            await self._summarize()

    @property
    def _summarization_estimate(self) -> int:
        # Trigger summarization earlier than token_estimate because role
        # markers/newlines and tool-heavy prompts consume extra budget.
        return sum((len(m["content"]) // 2) + 12 for m in self.messages)

    async def _summarize(self) -> None:
        conversation = self.get_context_prompt()
        summary_prompt = (
            f"Summarize the following conversation concisely, preserving key facts:\n\n"
            f"{conversation}\n\nSummary:"
        )
        try:
            summary = await asyncio.wait_for(
                self._client.complete(
                    prompt=summary_prompt,
                    slot_id=self._slot_id,
                    max_tokens=256,
                    temperature=0.3,
                ),
                timeout=self.summary_timeout_seconds,
            )
            summary_text = summary.strip()
        except Exception:
            # Fast deterministic fallback keeps add_message non-blocking when
            # the LLM is slow/unavailable.
            lines = [
                f"{m['role']}: {m['content'].strip()}"
                for m in self.messages[-4:]
                if m.get("content")
            ]
            summary_text = " | ".join(lines)[:600].strip()

        self.messages = [{"role": "system", "content": f"Summary: {summary_text}"}]
        self._messages_since_summary = 0
        self._has_summary = True

    def get_context_prompt(self) -> str:
        return "".join(f"{m['role']}: {m['content']}\n" for m in self.messages)
