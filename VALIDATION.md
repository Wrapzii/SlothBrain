# SlothBrain Validation Framework

## Overview

The validation suite provides comprehensive testing of critical SlothBrain subsystems using the **actual LLM** (not mocks). Every test includes:

- **Visible INPUT** - what data is fed to the system
- **Visible OUTPUT** - what the system produces  
- **Expected VALIDATION** - what we verify about the results

This approach ensures we catch real issues that mocked tests would miss.

---

## Quick Start

### Enable Validation Tests

```bash
export SLOTHBRAIN_RUN_VALIDATION_TESTS=1
```

### Run All Validation Tests

```bash
pytest backend/tests/manual/test_validation_suite.py -v --tb=short
```

### Run Specific Subsystem

```bash
# Rolling context only
pytest backend/tests/manual/test_validation_suite.py::test_rolling_context_message_accumulation -v

# Memory validation
pytest backend/tests/manual/test_validation_suite.py -k "test_memory" -v

# Checkpoint tests
pytest backend/tests/manual/test_validation_suite.py -k "test_checkpoint" -v
```

### See Detailed Output

```bash
pytest backend/tests/manual/test_validation_suite.py -v -s
```

The `-s` flag shows all `print()` statements, making input/output visible.

---

## Validation Suite Structure

### 1. **Rolling Context Validation**

Tests the conversation history management system.

#### What's Tested:
- Message accumulation (building multi-turn conversations)
- Token counting accuracy  
- Summarization trigger and quality
- Context tree correctness

#### Key Tests:
- `test_rolling_context_message_accumulation` - Ensure messages accumulate correctly
- `test_rolling_context_summarization_trigger` - Verify LLM summarizes long conversations

#### Expected Behavior:
```
INPUT: 4 messages about geography
  → user: "What is France's capital?"
  → assistant: "Paris."
  → user: "Population?"
  → assistant: "~2.2 million."

OUTPUT: Context prompt showing all messages

VALIDATION: ✓ All content present in context
            ✓ Token count > 0
```

---

### 2. **Checkpoint Validation**

Tests the save/restore mechanism for task state.

#### What's Tested:
- Checkpoint structure preservation
- Full state restoration (task, steps, context)
- Multiple checkpoint management
- Checkpoint retrieval accuracy

#### Key Tests:
- `test_checkpoint_save_and_restore` - Save state, restore it, verify match
- `test_checkpoint_multiple_saves` - Handle multiple checkpoints per run

#### Expected Behavior:
```
INPUT: Task at step 2 with context and executed steps
  → Save checkpoint

VALIDATION: ✓ Checkpoint contains all state
            ✓ Restore returns identical state
            ✓ Step numbers preserved
            ✓ Context lines present
```

---

### 3. **Memory Cache Validation**

Tests vector embedding and semantic search.

#### What's Tested:
- Vector embedding correctness
- Semantic search ranking
- Memory retrieval relevance
- Cache performance

#### Key Tests:
- `test_memory_store_and_retrieve` - Store memories, search, verify top result
- `test_memory_relevance_ranking` - Query ranking semantic similarity correctly

#### Expected Behavior:
```
INPUT: Memories about programming languages
  - "Python is used in data science"
  - "Go for concurrent systems"
  - "Rust for memory safety"

SEARCH: "What language for data science?"

OUTPUT: Top result mentions Python

VALIDATION: ✓ Relevant items rank high
            ✓ Search returns results  
            ✓ Ranking follows semantic similarity
```

---

### 4. **Slot Manager Validation**

Tests per-slot inference isolation and history tracking.

#### What's Tested:
- Slot inference isolation
- Message history per slot
- Response consistency
- Cache behavior

#### Key Tests:
- `test_slot_manager_inference_isolation` - Sequential prompts get different responses
- `test_slot_manager_history_tracking` - Each slot maintains independent history

#### Expected Behavior:
```
INPUT: Slot 0 receives prompts:
  1. "Respond with: FIRST"
  2. "Respond with: SECOND"

VALIDATION: ✓ Different responses (not cached)
            ✓ History shows both exchanges
            ✓ No cross-slot contamination
```

---

### 5. **Integration Validation**

Tests full pipelines combining multiple subsystems.

#### What's Tested:
- Task → Checkpoint → Memory → Resume flow
- Data consistency across subsystems
- State transitions
- Full loop correctness

#### Key Tests:
- `test_full_pipeline_with_checkpoints_and_memory` - Complete workflow validation

#### Expected Behavior:
```
INPUT: Full task lifecycle
  1. Store context in memory
  2. Create checkpoint
  3. Simulate execution  
  4. Restore checkpoint
  5. Retrieve from memory

VALIDATION: ✓ Context stored and retrievable
            ✓ Checkpoint restores correctly
            ✓ Memory queries return results
            ✓ No state loss across operations
```

---

### 6. **Cache Validation**

Tests caching mechanisms.

#### What's Tested:
- Cache hit performance
- Cache expiration
- Data consistency
- Speedup verification

#### Key Tests:
- `test_slot_info_cache_validity` - Second call faster than first

#### Expected Behavior:
```
INPUT: Same query twice
  1. First call: 50ms (cache miss, compute)
  2. Second call: 5ms (cache hit)

VALIDATION: ✓ Second call faster
            ✓ Data identical
            ✓ Cache working correctly
```

---

### 7. **Context Tree Validation**

Tests the hierarchical structure of conversations.

#### What's Tested:
- Message ordering
- Parent-child relationships
- Tree traversal correctness  
- Structure preservation

#### Key Tests:
- `test_context_tree_structure` - Verify tree is properly formed

#### Expected Behavior:
```
INPUT: Nested conversation
  user: "What is AI?"
  assistant: "It's intelligence..."
  user: "Applications?"
  assistant: "Healthcare, finance..."

VALIDATION: ✓ Proper alternation
            ✓ Chain formation correct
            ✓ No orphaned messages
```

---

### 8. **LLM Quality Validation**

Tests LLM response quality and consistency.

#### What's Tested:
- Response non-emptiness
- Response length appropriateness
- Format correctness (haiku, lists, etc.)
- Consistency across calls

#### Key Tests:
- `test_llm_response_quality_and_consistency` - Various prompt types validated

#### Expected Behavior:
```
INPUT: Structured requests
  1. "Haiku about programming"
  2. "List 3 Python libraries"
  3. "Capital of France in one word"

VALIDATION: ✓ All responses non-empty
            ✓ Appropriate lengths
            ✓ Correct format (3 lines, numbered list, one word)
            ✓ Sensible content
```

---

### 9. **Performance Validation**

Tests system performance under load.

#### What's Tested:
- Operation speed under load
- Memory efficiency
- Responsiveness
- Degradation curves

#### Key Tests:
- `test_rolling_context_performance` - Add 20 messages, measure throughput

#### Expected Behavior:
```
INPUT: Add 20 messages to rolling context

EXPECTED: < 5 seconds total
         < 250ms per message average

VALIDATION: ✓ Completes in < 5s
            ✓ Per-message latency acceptable
            ✓ No exponential slowdown
```

---

## Test Output Format

Every test prints:

```
[Subsystem] Test description:
  Input/Operation details
  Actual output values
  Validation checks (✓ = pass)
```

Example:
```
[Memory] Search results for: 'What programming languages are preferred?'
  Top result: The user prefers Python for scripting tasks...
  Relevance score: 0.89

[Memory] Relevance ranking for: 'What is Python used for in data analysis?'
  1. Python is widely used in data science and machine learning...
  2. The Python language has a clean syntax and large ecosystem...
  3. JavaScript runs in browsers and on servers with Node.js...
```

---

## Adding New Validations

Template for new validation test:

```python
@pytest.mark.asyncio
async def test_subsystem_aspect(relevant_fixtures):
    """Validate specific behavior with actual LLM."""
    
    # INPUT: Describe what we're sending
    print("\n[Subsystem] Test description:")
    input_data = {...}
    
    # OPERATION: Perform the operation
    result = await some_operation(input_data)
    
    # OUTPUT: Show what we got
    print(f"  Result: {result}")
    
    # VALIDATION: Assert expected behavior
    assert result.property == expected_value, "Error message"
    assert len(result) > 0, "Result should not be empty"
```

---

## Environment Variables

```bash
# Enable validation tests (required)
export SLOTHBRAIN_RUN_VALIDATION_TESTS=1

# LLM server (if non-default)
export SLOTHBRAIN_LLAMA_HOST=localhost
export SLOTHBRAIN_LLAMA_PORT=8080

# Database path
export SLOTHBRAIN_LANCEDB_PATH=./data/lancedb
```

---

## Requirements

1. **llama.cpp server running** on configured host/port
2. **LanceDB + sentence-transformers** installed
3. **Python 3.9+** with async support
4. **Sufficient VRAM** for inference (check llama.cpp logs)

---

## Troubleshooting

### "Tests skipped - set SLOTHBRAIN_RUN_VALIDATION_TESTS=1"
```bash
export SLOTHBRAIN_RUN_VALIDATION_TESTS=1
```

### "Failed to connect to llama.cpp"
```bash
# Verify server is running:
curl http://localhost:8080/health

# Check LLAMA_HOST and LLAMA_PORT settings
echo $SLOTHBRAIN_LLAMA_HOST $SLOTHBRAIN_LLAMA_PORT
```

### "LanceDB connection failed"
```bash
# Clear corrupted database:
rm -rf ./data/lancedb

# Re-run test (will reinitialize)
```

### Test timeout on LLM inference
- LLM may be overloaded
- Reduce batch size or concurrent tests
- Increase timeout in test (change `timeout_seconds` param)

---

## Continuous Validation

Use GitHub Actions to run validation suite automatically:

```yaml
# .github/workflows/validation.yml
- name: Run validation suite
  run: |
    export SLOTHBRAIN_RUN_VALIDATION_TESTS=1
    pytest backend/tests/manual/test_validation_suite.py -v
```

---

## Success Criteria

✅ All tests PASS without mocking  
✅ Input/Output visible in logs  
✅ Every value explicitly verified  
✅ Performance within thresholds  
✅ Integration paths tested end-to-end  

---

## Related Files

- **Tests**: `backend/tests/manual/test_validation_suite.py`
- **Manual Full-Stack**: `backend/tests/manual/test_manual_llm_fullstack.py`
- **Unit Tests**: `backend/tests/`
