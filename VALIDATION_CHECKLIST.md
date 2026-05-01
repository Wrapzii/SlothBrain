# Validation Checklist

This checklist tracks validation of all critical SlothBrain subsystems. Every feature must pass validation before being marked as production-ready.

---

## ✅ Rolling Context Validation

**Purpose**: Ensure message accumulation, summarization, and token counting work correctly.

- [ ] **test_rolling_context_message_accumulation**
  - Verify 4+ messages accumulate correctly
  - Check token_estimate > 0
  - Validate all content present in context prompt

- [ ] **test_rolling_context_summarization_trigger**
  - Add messages exceeding summarize threshold
  - Verify summarization is triggered
  - Check summary message created and makes sense
  - Validate messages reduced to 1-2 after summarization

- [ ] **test_context_tree_structure**
  - Verify message tree is properly formed
  - Check message ordering preserved
  - Validate alternating user/assistant pattern
  - Verify no orphaned messages

---

## ✅ Checkpoint Validation

**Purpose**: Ensure state snapshots save and restore correctly for task recovery.

- [ ] **test_checkpoint_save_and_restore**
  - Save checkpoint with full task state
  - Restore from checkpoint
  - Verify all state fields preserved (task, step_num, context, steps)
  - Validate step descriptions match

- [ ] **test_checkpoint_multiple_saves**
  - Save 5+ checkpoints for same run
  - Verify last checkpoint correctly identified
  - Validate each checkpoint has correct step_num
  - Check context preserved per checkpoint

---

## ✅ Memory Cache Validation

**Purpose**: Ensure vector embedding and semantic search work correctly.

- [ ] **test_memory_store_and_retrieve**
  - Store 3+ semantic memories
  - Search with query
  - Verify top result is relevant
  - Check all memory fields present

- [ ] **test_memory_relevance_ranking**
  - Store 5+ semantically diverse memories
  - Search for specific topic (Python data science)
  - Verify Python items rank in top 2-3
  - Validate non-relevant items rank lower
  - Check semantic similarity scoring

---

## ✅ Slot Manager Validation

**Purpose**: Ensure per-slot inference isolation and history tracking.

- [ ] **test_slot_manager_inference_isolation**
  - Send 2 different prompts to same slot
  - Verify responses differ (not cached)
  - Check both responses non-empty
  - Validate no cross-contamination

- [ ] **test_slot_manager_history_tracking**
  - Add 3+ messages to slot
  - Retrieve slot info
  - Verify message history present
  - Check message count matches additions

---

## ✅ Cache Validation

**Purpose**: Ensure caching improves performance without stale data.

- [ ] **test_slot_info_cache_validity**
  - Get slot info twice in quick succession
  - Measure timing (second call faster)
  - Verify data identical in both calls
  - Check cache speedup is meaningful (>2x)

---

## ✅ LLM Quality Validation

**Purpose**: Ensure LLM responses meet quality and format requirements.

- [ ] **test_llm_response_quality_and_consistency**
  - Test 3 different prompt types (haiku, list, one-word)
  - Verify all responses non-empty
  - Check response lengths appropriate
  - Validate format correctness (3 lines for haiku, numbered for list, etc.)
  - Verify sensible content

---

## ✅ Integration Validation

**Purpose**: Ensure full pipelines work end-to-end with all subsystems.

- [ ] **test_full_pipeline_with_checkpoints_and_memory**
  - Store context in memory
  - Create checkpoint
  - Simulate task execution
  - Retrieve checkpoint
  - Search memory
  - Verify no state loss across operations
  - Validate data integrity throughout pipeline

---

## ✅ Performance Validation

**Purpose**: Ensure acceptable latency under load.

- [ ] **test_rolling_context_performance**
  - Add 20 messages to rolling context
  - Verify completes in < 5 seconds
  - Check per-message latency < 250ms
  - Validate no exponential slowdown

---

## Known Issues Being Validated

### BUG-002: RollingContext._summarize() blocks event loop
**Validation Status**: 🔄 In Progress
- Test: `test_rolling_context_summarization_trigger`
- Check: Does summarization happen without blocking?
- Expected: Summarization completes and event loop continues

### BUG-003: LanceDBMemory._get_table() creates/deletes seed row
**Validation Status**: 🔄 In Progress
- Test: `test_memory_store_and_retrieve`
- Check: Does memory store work without row creation issues?
- Expected: Store/search operations work cleanly

### BUG-012: token_estimate uses rough len//4 heuristic
**Validation Status**: ❌ Not Started
- Test: Need to create `test_rolling_context_token_accuracy`
- Check: Compare len//4 estimate vs actual token counts
- Expected: Estimate within 10% of actual

---

## Feature Validation Dependencies

```
Rolling Context ──────┐
                      ├──→ Integration Validation
Checkpoints ──────────┤    (full pipeline)
                      ├──→ Performance Tests
Memory Cache ─────────┤    (under load)
                      │
Slot Manager ─────────┘

LLM Quality Checks (foundational)
├── Response non-emptiness
├── Format correctness
└── Semantic accuracy
```

---

## Running All Validations

```bash
# Enable validation tests
export SLOTHBRAIN_RUN_VALIDATION_TESTS=1

# Run full suite
pytest backend/tests/manual/test_validation_suite.py -v -s

# Or use helper script
python run_validation.py --all
```

---

## Validation Pass Criteria

✅ **PASS** if:
- All assertions pass (no exceptions)
- All print() statements show reasonable values
- Performance within thresholds (< 5s, < 250ms/msg)
- No mocking (uses real LLM)
- Input/output visible in logs

❌ **FAIL** if:
- Any assertion fails
- Timeout (> 5s total)
- LLM connection error
- Memory/vector search returns empty
- State lost or corrupted across operations

---

## Adding New Validations

When adding new features:

1. **Create test function** in `test_validation_suite.py`
2. **Show INPUT** - print what data enters system
3. **Show OUTPUT** - print what system produces
4. **Add VALIDATION** - assert expected behavior
5. **Update this checklist** - add checkbox item
6. **Document in VALIDATION.md** - explain what's tested

---

## Validation Reports

Keep summary of validation runs:

| Date | Test Suite | Status | Issues Found |
|------|-----------|--------|--------------|
| 2024-01-15 | Full Suite | ✅ PASS | None |
| 2024-01-16 | Rolling Context | ✅ PASS | BUG-002 confirmation needed |
| 2024-01-17 | Memory Cache | ✅ PASS | Seed row creation noticed |

---

## Related Documentation

- [VALIDATION.md](VALIDATION.md) - Detailed validation framework docs
- [BUGS.md](BUGS.md) - Known issues (some being validated)
- [backend/tests/manual/](backend/tests/manual/) - All test files
- [README.md](README.md) - Running tests section
