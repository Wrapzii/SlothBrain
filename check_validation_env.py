#!/usr/bin/env python3
"""
Environment check for SlothBrain validation suite.

Verifies:
- Required Python packages installed
- LLM server accessible
- Database connectivity  
- Configuration correct

Run before validation tests to catch configuration issues early.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def check_imports() -> bool:
    """Check all required imports are available."""
    print("\n📦 Checking imports...")
    try:
        import pytest
        import pytest_asyncio
        from fastapi.testclient import TestClient
        from lancedb import connect
        from sentence_transformers import SentenceTransformer
        
        from backend.main import app
        from backend.memory.rolling_context import RollingContext
        from backend.core.checkpoint_manager import CheckpointManager
        from backend.memory.lancedb_memory import LanceDBMemory
        from backend.core.slot_manager import SlotManager
        from backend.core.llama_client import LlamaClient
        
        print("  ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


async def check_llm_server() -> bool:
    """Check LLM server is accessible."""
    print("\n🧠 Checking LLM server...")
    try:
        from backend.config import settings
        from backend.core.llama_client import LlamaClient
        
        client = LlamaClient(host=settings.llama_host, port=settings.llama_port)
        
        # Try a simple health check
        response = await client.complete(
            prompt="Hello",
            max_tokens=5,
            timeout=5,
        )
        
        if response and len(response) > 0:
            print(f"  ✅ LLM server responding ({settings.llama_host}:{settings.llama_port})")
            print(f"     Response: {response[:50].strip()}...")
            return True
        else:
            print(f"  ❌ LLM server returned empty response")
            return False
            
    except asyncio.TimeoutError:
        print(f"  ❌ LLM server timeout (check it's running)")
        return False
    except Exception as e:
        print(f"  ❌ LLM connection failed: {e}")
        return False


async def check_database() -> bool:
    """Check database connectivity."""
    print("\n📊 Checking database...")
    try:
        from backend.config import settings
        from backend.memory.lancedb_memory import LanceDBMemory
        
        mem = LanceDBMemory(
            db_path=settings.lancedb_path,
            embedding_model=settings.embedding_model,
        )
        
        # Try a simple store/search cycle
        await mem.store(
            text="Test memory entry",
            metadata={"type": "test", "timestamp": "now"},
        )
        
        results = await mem.search(query="memory", limit=1)
        
        if results:
            print(f"  ✅ Database working ({settings.lancedb_path})")
            print(f"     Stored and retrieved test entry")
            return True
        else:
            print(f"  ⚠️  Database connected but search returned empty")
            return True  # Database accessible, just empty
            
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False


async def check_environment_vars() -> bool:
    """Check required environment variables."""
    print("\n⚙️  Checking environment variables...")
    
    required = [
        ("SLOTHBRAIN_LLAMA_HOST", "localhost"),
        ("SLOTHBRAIN_LLAMA_PORT", "8080"),
        ("SLOTHBRAIN_LANCEDB_PATH", "./data/lancedb"),
        ("SLOTHBRAIN_MAIN_CONTEXT_SIZE", "4096"),
    ]
    
    all_set = True
    for var, default in required:
        value = os.getenv(var, default)
        if value == default:
            print(f"  ℹ️  {var} = {value} (default)")
        else:
            print(f"  ✅ {var} = {value}")
    
    # Check validation test flag
    validation_enabled = os.getenv("SLOTHBRAIN_RUN_VALIDATION_TESTS", "0") == "1"
    if validation_enabled:
        print(f"  ✅ SLOTHBRAIN_RUN_VALIDATION_TESTS = 1 (enabled)")
    else:
        print(f"  ⚠️  SLOTHBRAIN_RUN_VALIDATION_TESTS not set (tests will be skipped)")
        print(f"     Set it with: export SLOTHBRAIN_RUN_VALIDATION_TESTS=1")
        all_set = False
    
    return all_set


async def check_test_file() -> bool:
    """Check validation test file exists and is readable."""
    print("\n📝 Checking test file...")
    
    test_file = Path(__file__).parent / "backend" / "tests" / "manual" / "test_validation_suite.py"
    
    if test_file.exists():
        size = test_file.stat().st_size
        lines = test_file.read_text().count("\n")
        print(f"  ✅ Test file exists: {test_file}")
        print(f"     Size: {size:,} bytes, {lines:,} lines")
        return True
    else:
        print(f"  ❌ Test file not found: {test_file}")
        return False


async def main():
    """Run all checks."""
    print("\n" + "="*60)
    print(" SlothBrain Validation Suite Environment Check")
    print("="*60)
    
    checks = [
        ("Imports", check_imports),
        ("LLM Server", check_llm_server),
        ("Database", check_database),
        ("Environment Variables", check_environment_vars),
        ("Test File", check_test_file),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = await check_func()
        except Exception as e:
            print(f"\n⚠️  Check {name} failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print(" Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n✅ Environment ready! You can run validation tests.")
        print("\nTo run validation suite:")
        print("  export SLOTHBRAIN_RUN_VALIDATION_TESTS=1")
        print("  pytest backend/tests/manual/test_validation_suite.py -v -s")
        print("\nOr use the helper script:")
        print("  python run_validation.py")
        return 0
    else:
        print("\n❌ Environment check failed. Fix issues above and try again.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
