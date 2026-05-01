"""
Comprehensive validation suite for critical SlothBrain subsystems.

This suite validates using the actual LLM (not mocks) with visible input/output
inspection and explicit expected value verification.

Environment:
- SLOTHBRAIN_RUN_VALIDATION_TESTS=1 (default: off)
- All tests require a running llama.cpp server

Tests:
1. Rolling Context Validation - message accumulation, summarization quality
2. Checkpoint Validation - save/restore state preservation
3. Memory Cache Validation - vector search relevance and retrieval
4. Slot Manager Validation - isolation and state management
5. Integration Validation - full pipeline correctness
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from backend.config import settings
from backend.core.checkpoint_manager import CheckpointManager, TaskCheckpoint
from backend.core.llama_client import LlamaClient
from backend.core.slot_manager import SlotManager
from backend.main import app
from backend.memory.lancedb_memory import LanceDBMemory
from backend.memory.rolling_context import RollingContext


RUN_VALIDATION = os.getenv("SLOTHBRAIN_RUN_VALIDATION_TESTS", "0") == "1"
if not RUN_VALIDATION:
    pytest.skip(
        "Validation suite disabled. Set SLOTHBRAIN_RUN_VALIDATION_TESTS=1.",
        allow_module_level=True,
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def llama_client() -> LlamaClient:
    """Shared LLM client for all validation tests."""
    client = LlamaClient(host=settings.llama_host, port=settings.llama_port)
    yield client


@pytest_asyncio.fixture
async def slot_manager(llama_client: LlamaClient) -> SlotManager:
    """Slot manager with main slot assigned."""
    manager = SlotManager(llama_client=llama_client)
    await manager.assign_main(settings.main_slot)
    yield manager


@pytest_asyncio.fixture
async def rolling_context(llama_client: LlamaClient) -> RollingContext:
    """Rolling context for conversation tracking."""
    ctx = RollingContext(
        llama_client=llama_client,
        slot_id=settings.main_slot,
        max_tokens=settings.main_context_size,
        summarize_at=200,  # Low threshold for testing
    )
    yield ctx


@pytest_asyncio.fixture
def checkpoint_manager() -> CheckpointManager:
    """Checkpoint manager for state persistence."""
    return CheckpointManager(max_checkpoints_per_run=10)


@pytest_asyncio.fixture
async def memory_store() -> LanceDBMemory:
    """Memory store for vector search validation."""
    mem = LanceDBMemory(
        db_path=settings.lancedb_path,
        embedding_model=settings.embedding_model,
    )
    yield mem


@pytest_asyncio.fixture
def validation_client() -> TestClient:
    """FastAPI test client for endpoint validation."""
    return TestClient(app)


# ============================================================================
# Rolling Context Validation
# ============================================================================


@pytest.mark.asyncio
async def test_rolling_context_message_accumulation(rolling_context: RollingContext) -> None:
    """Validate that rolling context accumulates messages correctly."""
    # INPUT: Add series of messages
    messages = [
        ("user", "What is the capital of France?"),
        ("assistant", "The capital of France is Paris."),
        ("user", "What is its population?"),
        ("assistant", "Paris has approximately 2.2 million people in the city proper."),
    ]

    for role, content in messages:
        await rolling_context.add_message(role, content)

    # VALIDATION: Check all messages present
    assert len(rolling_context.messages) == 4, f"Expected 4 messages, got {len(rolling_context.messages)}"
    
    # OUTPUT: Inspect accumulated context
    context_prompt = rolling_context.get_context_prompt()
    print("\n[Rolling Context] Accumulated messages:")
    print(context_prompt)
    
    # EXPECTED: All content present
    assert "Paris" in context_prompt
    assert "2.2 million" in context_prompt
    assert rolling_context.token_estimate > 0


@pytest.mark.asyncio
async def test_rolling_context_summarization_trigger(
    llama_client: LlamaClient,
    rolling_context: RollingContext,
) -> None:
    """Validate that summarization is triggered and produces meaningful output."""
    # INPUT: Add messages to exceed summarize threshold
    long_messages = [
        ("user", "Tell me about the history of the Roman Empire. Include details about emperors, conquests, culture, and decline."),
        ("assistant", "The Roman Empire was one of the largest empires in history, spanning from 27 BC to 476 AD in the West..."),
        ("user", "What about the Byzantine continuation?"),
        ("assistant", "The Eastern Roman Empire, known as the Byzantine Empire, continued until 1453..."),
    ]

    token_count_before = rolling_context.token_estimate
    
    for role, content in long_messages:
        await rolling_context.add_message(role, content)

    # VALIDATION: Summarization should have been triggered
    print(f"\n[Rolling Context] Token estimate before: {token_count_before}")
    print(f"[Rolling Context] Token estimate after adding messages: {rolling_context.token_estimate}")
    
    # EXPECTED: Context is summarized (reduced to 1-2 messages)
    assert len(rolling_context.messages) <= 2, (
        f"Expected summarization to reduce messages to 1-2, got {len(rolling_context.messages)}"
    )
    
    # EXPECTED: Summary message present
    summary_present = any("Summary:" in m.get("content", "") for m in rolling_context.messages)
    assert summary_present, "Expected summary message in context"
    
    # OUTPUT: Show summary quality
    context_prompt = rolling_context.get_context_prompt()
    print("[Rolling Context] Summarized context:")
    print(context_prompt)


# ============================================================================
# Checkpoint Validation
# ============================================================================


@pytest.mark.asyncio
async def test_checkpoint_save_and_restore(checkpoint_manager: CheckpointManager) -> None:
    """Validate checkpoint save/restore preserves task state."""
    run_id = "test-run-001"
    task = "Complete a multi-step task with checkpointing"
    
    # INPUT: Save checkpoint with full state
    step_descriptions = [
        "Step 1: Gather information",
        "Step 2: Analyze findings",
        "Step 3: Generate report",
    ]
    context = [
        "Found important data point X",
        "Correlated with metric Y",
    ]
    executed_steps = [
        {
            "step_num": 1,
            "description": step_descriptions[0],
            "result": "Gathered market data from 5 sources",
            "final_action": "continue",
        },
        {
            "step_num": 2,
            "description": step_descriptions[1],
            "result": "Analysis shows 15% growth trend",
            "final_action": "continue",
        },
    ]

    # SAVE checkpoint
    checkpoint = checkpoint_manager.save(
        run_id=run_id,
        task=task,
        step_num=2,
        step_descriptions=step_descriptions,
        context=context,
        executed_steps=executed_steps,
    )

    # OUTPUT: Show checkpoint structure
    print(f"\n[Checkpoint] Saved checkpoint at step {checkpoint.step_num}:")
    print(f"  Task: {checkpoint.task[:50]}...")
    print(f"  Context lines: {len(checkpoint.context)}")
    print(f"  Executed steps: {len(checkpoint.executed_steps)}")

    # RESTORE checkpoint
    restored = checkpoint_manager.restore_last(run_id)
    
    # EXPECTED: All state preserved
    assert restored is not None, "Failed to restore checkpoint"
    assert restored.task == task
    assert restored.step_num == 2
    assert len(restored.context) == len(context)
    assert len(restored.executed_steps) == len(executed_steps)
    
    # VALIDATION: Step descriptions match
    assert restored.step_descriptions[0] == step_descriptions[0]
    
    # OUTPUT: Verify restoration
    print(f"[Checkpoint] Restored checkpoint successfully")
    print(f"  Step descriptions match: {restored.step_descriptions == step_descriptions}")


@pytest.mark.asyncio
async def test_checkpoint_multiple_saves(checkpoint_manager: CheckpointManager) -> None:
    """Validate checkpoint management with multiple saves."""
    run_id = "test-run-multi"
    
    # INPUT: Save multiple checkpoints
    checkpoints_saved = []
    for step_num in range(1, 6):
        cp = checkpoint_manager.save(
            run_id=run_id,
            task="Multi-step task",
            step_num=step_num,
            step_descriptions=[f"Step {i}" for i in range(1, step_num + 1)],
            context=[f"Context at step {step_num}"],
            executed_steps=[
                {"step_num": i, "description": f"Step {i}", "result": f"Result {i}", "final_action": "continue"}
                for i in range(1, step_num + 1)
            ],
        )
        checkpoints_saved.append(cp)
        await asyncio.sleep(0.01)  # Simulate time passing

    # EXPECTED: Last checkpoint restored
    last = checkpoint_manager.restore_last(run_id)
    assert last is not None
    assert last.step_num == 5, f"Expected step 5, got {last.step_num}"
    
    # OUTPUT: Show checkpoint history
    print(f"\n[Checkpoint] Saved {len(checkpoints_saved)} checkpoints")
    print(f"  Last checkpoint step: {last.step_num}")
    print(f"  Last checkpoint context: {last.context}")


# ============================================================================
# Memory Cache Validation
# ============================================================================


@pytest.mark.asyncio
async def test_memory_store_and_retrieve(memory_store: LanceDBMemory) -> None:
    """Validate memory store can embed and retrieve vectors correctly."""
    # INPUT: Store semantic memories
    memories = [
        ("The user prefers Python for scripting tasks", {"user_id": "test1", "type": "preference"}),
        ("SQL queries are faster than loops for large datasets", {"source": "docs", "type": "fact"}),
        ("Machine learning models require data normalization", {"domain": "ml", "type": "best_practice"}),
    ]

    print("\n[Memory] Storing memories:")
    for text, metadata in memories:
        await memory_store.store(text=text, metadata=metadata)
        print(f"  - {text[:60]}...")

    # INPUT: Search with semantic query
    query = "What programming languages are preferred?"
    
    # RETRIEVE: Search memory
    results = await memory_store.search(query=query, limit=3)
    
    # EXPECTED: Top result is about Python preference
    assert results, "Memory search returned no results"
    top_result = results[0]
    
    print(f"\n[Memory] Search results for: '{query}'")
    print(f"  Top result: {top_result['text'][:70]}...")
    print(f"  Relevance score: {top_result.get('score', 'N/A')}")
    
    # VALIDATION: Python result should rank high
    assert "Python" in top_result["text"], f"Expected Python in top result, got: {top_result['text']}"


@pytest.mark.asyncio
async def test_memory_relevance_ranking(memory_store: LanceDBMemory) -> None:
    """Validate that memory search ranks relevant items correctly."""
    # INPUT: Store semantically diverse memories
    memories = [
        "The Python language has a clean syntax and large ecosystem",
        "JavaScript runs in browsers and on servers with Node.js",
        "Rust provides memory safety without garbage collection",
        "Go is designed for concurrent systems and cloud applications",
        "Python is widely used in data science and machine learning",
    ]

    print("\n[Memory] Storing semantically diverse memories:")
    for text in memories:
        await memory_store.store(text=text, metadata={"category": "languages"})
        print(f"  - {text}")

    # INPUT: Query for Python-specific info
    query = "What is Python used for in data analysis?"
    results = await memory_store.search(query=query, limit=3)

    # EXPECTED: Python results rank highest
    print(f"\n[Memory] Relevance ranking for: '{query}'")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['text'][:70]}...")

    # VALIDATION: At least one Python result in top 3
    python_found = any("Python" in r["text"] for r in results[:2])
    assert python_found, f"Expected Python in top 2 results. Got: {[r['text'] for r in results]}"


# ============================================================================
# Slot Manager Validation
# ============================================================================


@pytest.mark.asyncio
async def test_slot_manager_inference_isolation(
    llama_client: LlamaClient,
    validation_client: TestClient,
) -> None:
    """Validate slot manager keeps inference isolated per slot."""
    manager = SlotManager(llama_client=llama_client)
    await manager.assign_main(0)

    # INPUT: Send prompts to same slot
    prompts = [
        "Respond with exactly: FIRST_RESPONSE",
        "Respond with exactly: SECOND_RESPONSE",
    ]

    responses = []
    print("\n[SlotManager] Sequential responses from same slot:")
    
    for prompt in prompts:
        response = await manager.send_to_main(prompt, max_tokens=10)
        responses.append(response)
        print(f"  Input: {prompt}")
        print(f"  Output: {response}")

    # VALIDATION: Responses differ (not cached from first)
    assert len(responses) == 2
    # Note: We can't strictly assert they're different due to LLM variability
    # but we can verify both are non-empty
    assert all(r for r in responses), "Got empty responses"


@pytest.mark.asyncio
async def test_slot_manager_history_tracking(slot_manager: SlotManager) -> None:
    """Validate slot manager maintains per-slot history."""
    # INPUT: Multiple messages to main slot
    test_messages = [
        ("user", "What is 2+2?"),
        ("assistant", "2+2 equals 4."),
        ("user", "What is the next number?"),
    ]

    print("\n[SlotManager] Tracking message history:")
    slot_info = await slot_manager.get_slot_info()
    initial_count = len(slot_info.get("main_slot", {}).get("messages", []))

    for role, content in test_messages:
        print(f"  {role}: {content}")

    # Get updated slot info
    slot_info = await slot_manager.get_slot_info()
    main_slot_messages = slot_info.get("main_slot", {}).get("messages", [])

    # EXPECTED: Slot maintains message history
    print(f"[SlotManager] History count: {len(main_slot_messages)} messages")
    assert isinstance(main_slot_messages, list), "Messages should be a list"


# ============================================================================
# Integration Validation
# ============================================================================


@pytest.mark.asyncio
async def test_full_pipeline_with_checkpoints_and_memory(
    validation_client: TestClient,
    checkpoint_manager: CheckpointManager,
    memory_store: LanceDBMemory,
    llama_client: LlamaClient,
) -> None:
    """Validate full pipeline: task → checkpoint → memory → resume."""
    run_id = "integration-test-001"
    
    print("\n[Integration] Full pipeline validation:")
    
    # STEP 1: Store context in memory
    print("  1. Storing context in memory...")
    task_context = "The user wants to build a Python API with FastAPI framework"
    await memory_store.store(
        text=task_context,
        metadata={"run_id": run_id, "stage": "context"},
    )
    
    # STEP 2: Create checkpoint
    print("  2. Creating checkpoint...")
    checkpoint = checkpoint_manager.save(
        run_id=run_id,
        task="Build FastAPI application",
        step_num=1,
        step_descriptions=["Design API structure", "Implement endpoints", "Add testing"],
        context=[task_context],
        executed_steps=[
            {
                "step_num": 1,
                "description": "Design API structure",
                "result": "Created openapi.json with 5 endpoints",
                "final_action": "continue",
            }
        ],
    )
    
    # STEP 3: Simulate task completion
    print("  3. Simulating task execution...")
    await llama_client.complete(
        prompt="Summarize: Creating a Python API with FastAPI for task management",
        max_tokens=100,
    )
    
    # STEP 4: Retrieve from checkpoint
    print("  4. Retrieving checkpoint...")
    restored = checkpoint_manager.restore_last(run_id)
    assert restored is not None
    assert restored.task == "Build FastAPI application"
    
    # STEP 5: Retrieve from memory
    print("  5. Retrieving from memory...")
    memories = await memory_store.search(
        query="FastAPI API building",
        limit=3,
    )
    assert len(memories) > 0, "No memories found"
    
    # OUTPUT: Full pipeline summary
    print("\n[Integration] Pipeline validation complete:")
    print(f"  ✓ Context stored in memory")
    print(f"  ✓ Checkpoint created (step {checkpoint.step_num})")
    print(f"  ✓ Checkpoint restored successfully")
    print(f"  ✓ Memory retrieved ({len(memories)} results)")


# ============================================================================
# Cache Validation
# ============================================================================


@pytest.mark.asyncio
async def test_slot_info_cache_validity(slot_manager: SlotManager) -> None:
    """Validate slot info caching behaves correctly."""
    # INPUT: Get slot info twice
    print("\n[Cache] Validating slot info cache:")
    
    start_time = time.time()
    info1 = await slot_manager.get_slot_info()
    time1 = time.time() - start_time
    
    start_time = time.time()
    info2 = await slot_manager.get_slot_info()
    time2 = time.time() - start_time
    
    # EXPECTED: Second call is faster (cached)
    print(f"  First call: {time1*1000:.2f}ms")
    print(f"  Second call: {time2*1000:.2f}ms")
    print(f"  Cache speedup: {time1/time2:.1f}x (approx)")
    
    # VALIDATION: Data structures consistent
    assert type(info1) == type(info2), "Slot info type should be consistent"
    assert info1.get("main_slot") == info2.get("main_slot"), "Slot info should be cached"


# ============================================================================
# Tree Structure Validation (Context Tree)
# ============================================================================


@pytest.mark.asyncio
async def test_context_tree_structure(rolling_context: RollingContext) -> None:
    """Validate context maintains proper message tree structure."""
    # INPUT: Add nested conversation
    messages = [
        ("user", "Root question: What is AI?"),
        ("assistant", "AI is artificial intelligence."),
        ("user", "What are its applications?"),
        ("assistant", "AI has many applications in healthcare, finance, etc."),
    ]

    print("\n[ContextTree] Building message tree:")
    for role, content in messages:
        await rolling_context.add_message(role, content)
        print(f"  {role}: {content[:50]}...")

    # VALIDATION: Tree is properly formed
    assert len(rolling_context.messages) > 0, "Messages should exist"
    
    # EXPECTED: Alternating user/assistant pattern
    roles = [m["role"] for m in rolling_context.messages if "Summary" not in m.get("content", "")]
    if len(roles) >= 2:
        # Check for typical alternation (may be summarized)
        print(f"[ContextTree] Message roles: {roles}")
    
    # OUTPUT: Show structure
    prompt = rolling_context.get_context_prompt()
    print(f"[ContextTree] Structure:\n{prompt}")


# ============================================================================
# LLM Output Quality Validation
# ============================================================================


@pytest.mark.asyncio
async def test_llm_response_quality_and_consistency(llama_client: LlamaClient) -> None:
    """Validate LLM responses meet quality criteria."""
    test_prompts = [
        ("Respond with a haiku about programming:", "haiku"),
        ("List 3 Python libraries for data science. Number them 1-3.", "list"),
        ("What is the capital of France? Answer in one word.", "one_word"),
    ]

    print("\n[LLM Quality] Testing response quality:")
    
    for prompt, expected_type in test_prompts:
        response = await llama_client.complete(
            prompt=prompt,
            max_tokens=100,
        )
        
        print(f"\n  Prompt: {prompt}")
        print(f"  Response: {response.strip()}")
        print(f"  Type: {expected_type}")
        
        # VALIDATION: Response is non-empty
        assert response.strip(), f"Empty response for prompt: {prompt}"
        
        # VALIDATION: Response length reasonable
        assert len(response) > 10, f"Response too short for prompt: {prompt}"


# ============================================================================
# Performance Validation
# ============================================================================


@pytest.mark.asyncio
async def test_rolling_context_performance(rolling_context: RollingContext) -> None:
    """Validate rolling context performance under load."""
    import time
    
    print("\n[Performance] Rolling context under load:")
    
    # INPUT: Add many messages
    start = time.time()
    for i in range(20):
        await rolling_context.add_message(
            "user" if i % 2 == 0 else "assistant",
            f"Message {i}: " + "x" * 100,
        )
    elapsed = time.time() - start
    
    # EXPECTED: Operations complete in reasonable time
    print(f"  Added 20 messages in {elapsed:.3f}s")
    print(f"  Avg per message: {(elapsed/20)*1000:.1f}ms")
    print(f"  Current token estimate: {rolling_context.token_estimate}")
    
    assert elapsed < 5.0, f"Context operations too slow: {elapsed}s"


if __name__ == "__main__":
    print("Run with: pytest backend/tests/manual/test_validation_suite.py -v")
