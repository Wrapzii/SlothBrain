"""Tool profile definitions.

A *profile* is a named set of tool names.  Sub-agents are assigned a profile
at spawn time so they only see (and can call) the tools relevant to their role.
This keeps prompts short and reduces the attack surface.

Special value ``"*"`` means *all registered tools* (the ``full`` profile).
"""
from __future__ import annotations

# Profile name → frozenset of tool names, or "*" for all tools.
PROFILES: dict[str, frozenset[str] | str] = {
    # Full access — main agent and trusted operators.
    "full": "*",

    # Coding tasks: file I/O, shell, code execution, diffs, patches,
    # web access, and memory retrieval.
    "coding": frozenset({
        "file",
        "shell",
        "code_exec",
        "diff",
        "patch",
        "web_fetch",
        "web_search",
        "memory_search",
    }),

    # Messaging-focused agents: Discord I/O + conversation history.
    "messaging": frozenset({
        "discord",
        "memory_search",
        "session",
    }),

    # Desktop / vision agents.
    "vision": frozenset({
        "screenshot",
        "ui",
        "image_analysis",
        "memory_search",
    }),

    # Minimal — status checks, memory search, agent listing only.
    # Default for newly spawned sub-agents.
    "minimal": frozenset({
        "memory_search",
        "session",
        "agent_list",
    }),

    # RAG / knowledge retrieval only.
    "rag": frozenset({
        "memory_search",
        "session_graph",
        "session",
    }),

    # Agent orchestration profile.
    "orchestration": frozenset({
        "sub_agent",
        "agent_list",
        "session",
        "memory_search",
    }),
}

# Default profile assigned when no profile is specified.
DEFAULT_PROFILE = "minimal"
