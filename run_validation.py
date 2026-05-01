#!/usr/bin/env python3
"""
SlothBrain Validation Suite Runner

Convenient wrapper for running validation tests with proper configuration.

Usage:
    python run_validation.py                    # All tests with output
    python run_validation.py --quick             # All tests, minimal output
    python run_validation.py --rolling-context   # Rolling context tests only
    python run_validation.py --memory            # Memory cache tests only
    python run_validation.py --checkpoints       # Checkpoint tests only
    python run_validation.py --slots             # Slot manager tests only
    python run_validation.py --integration       # Integration tests only
    python run_validation.py --performance       # Performance tests only
"""

import os
import sys
import subprocess
from pathlib import Path


def run_pytest(args: list[str]) -> int:
    """Run pytest with the given arguments."""
    return subprocess.run(["pytest"] + args, cwd=Path(__file__).parent).returncode


def main():
    """Main entry point."""
    # Enable validation tests environment variable
    os.environ["SLOTHBRAIN_RUN_VALIDATION_TESTS"] = "1"

    # Parse arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "--all"

    base_args = ["backend/tests/manual/test_validation_suite.py", "-v", "-s"]

    if mode == "--quick":
        # All tests, minimal output
        return run_pytest(["backend/tests/manual/test_validation_suite.py", "-v", "--tb=short"])

    elif mode == "--rolling-context":
        # Rolling context tests
        return run_pytest(base_args + ["-k", "test_rolling_context"])

    elif mode == "--memory":
        # Memory cache tests
        return run_pytest(base_args + ["-k", "test_memory"])

    elif mode == "--checkpoints":
        # Checkpoint tests
        return run_pytest(base_args + ["-k", "test_checkpoint"])

    elif mode == "--slots":
        # Slot manager tests
        return run_pytest(base_args + ["-k", "test_slot"])

    elif mode == "--integration":
        # Integration tests
        return run_pytest(base_args + ["-k", "test_full_pipeline or test_context_tree"])

    elif mode == "--performance":
        # Performance tests
        return run_pytest(base_args + ["-k", "test_performance"])

    elif mode == "--cache":
        # Cache tests
        return run_pytest(base_args + ["-k", "test_cache or test_slot_info"])

    elif mode == "--quality":
        # LLM quality tests
        return run_pytest(base_args + ["-k", "test_llm_response"])

    elif mode in ("--all", "--help", "-h"):
        if mode in ("--help", "-h"):
            print(__doc__)
            return 0
        # All tests with full output
        return run_pytest(base_args)

    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
