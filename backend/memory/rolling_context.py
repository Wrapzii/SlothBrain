from __future__ import annotations

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
    ) -> None:
        self._client = llama_client
        self._slot_id = slot_id
        self.max_tokens = max_tokens
        self.summarize_at = summarize_at
        self.messages: list[dict] = []

    @property
    def token_estimate(self) -> int:
        return sum(len(m["content"]) // 4 for m in self.messages)

    async def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if self.token_estimate > self.summarize_at:
            await self._summarize()

    async def _summarize(self) -> None:
        conversation = self.get_context_prompt()
        summary_prompt = (
            f"Summarize the following conversation concisely, preserving key facts:\n\n"
            f"{conversation}\n\nSummary:"
        )
        summary = await self._client.complete(
            prompt=summary_prompt,
            slot_id=self._slot_id,
            max_tokens=512,
            temperature=0.3,
        )
        self.messages = [{"role": "system", "content": f"Summary: {summary.strip()}"}]

    def get_context_prompt(self) -> str:
        return "".join(f"{m['role']}: {m['content']}\n" for m in self.messages)
