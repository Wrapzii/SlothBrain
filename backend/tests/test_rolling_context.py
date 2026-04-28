from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.memory.rolling_context import RollingContext


@pytest_asyncio.fixture
def mock_llama_client():
    client = MagicMock()
    client.complete = AsyncMock(return_value="This is a concise summary of the conversation.")
    return client


@pytest_asyncio.fixture
def rolling_context(mock_llama_client):
    return RollingContext(
        llama_client=mock_llama_client,
        slot_id=0,
        max_tokens=4096,
        summarize_at=50,  # Low threshold for testing
    )


@pytest.mark.asyncio
async def test_add_message_basic(rolling_context):
    await rolling_context.add_message("user", "Hello!")
    assert len(rolling_context.messages) == 1
    assert rolling_context.messages[0]["role"] == "user"
    assert rolling_context.messages[0]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_token_estimate(rolling_context):
    await rolling_context.add_message("user", "abcd")  # 4 chars → ~1 token
    assert rolling_context.token_estimate == 1


@pytest.mark.asyncio
async def test_get_context_prompt(rolling_context):
    await rolling_context.add_message("user", "Hi")
    await rolling_context.add_message("assistant", "Hello")
    prompt = rolling_context.get_context_prompt()
    assert "user: Hi" in prompt
    assert "assistant: Hello" in prompt


@pytest.mark.asyncio
async def test_summarization_triggered(mock_llama_client):
    """Summarization should collapse messages when token_estimate > summarize_at."""
    ctx = RollingContext(
        llama_client=mock_llama_client,
        slot_id=0,
        max_tokens=4096,
        summarize_at=10,  # Very low threshold
    )
    # Add a message long enough to exceed the threshold
    long_content = "A" * 100  # 100 chars → ~25 tokens > 10
    await ctx.add_message("user", long_content)

    # Summarization should have been triggered
    mock_llama_client.complete.assert_awaited_once()
    # After summarization, messages should be collapsed to a single system message
    assert len(ctx.messages) == 1
    assert ctx.messages[0]["role"] == "system"
    assert "Summary:" in ctx.messages[0]["content"]


@pytest.mark.asyncio
async def test_no_summarization_below_threshold(mock_llama_client):
    ctx = RollingContext(
        llama_client=mock_llama_client,
        slot_id=0,
        max_tokens=4096,
        summarize_at=3000,
    )
    await ctx.add_message("user", "short message")
    mock_llama_client.complete.assert_not_awaited()
    assert len(ctx.messages) == 1


@pytest.mark.asyncio
async def test_multiple_messages_before_summarization(mock_llama_client):
    ctx = RollingContext(
        llama_client=mock_llama_client,
        slot_id=0,
        max_tokens=4096,
        summarize_at=3000,
    )
    await ctx.add_message("user", "message one")
    await ctx.add_message("assistant", "response one")
    await ctx.add_message("user", "message two")
    assert len(ctx.messages) == 3
    mock_llama_client.complete.assert_not_awaited()
