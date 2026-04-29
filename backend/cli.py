from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from backend.agents.handoff import HandoffManager
from backend.agents.main_agent import MainAgent
from backend.agents.watcher import WatcherAgent
from backend.config import settings
from backend.core.llama_client import LlamaClient
from backend.core.resource_manager import ResourceManager
from backend.core.slot_manager import SlotManager
from backend.memory.lancedb_memory import LanceDBMemory
from backend.memory.rolling_context import RollingContext


@dataclass
class RuntimeState:
    handoff: HandoffManager
    watcher: WatcherAgent
    main_agent: MainAgent
    resources: ResourceManager


async def build_runtime() -> RuntimeState:
    llama_client = LlamaClient(host=settings.llama_host, port=settings.llama_port)
    slot_manager = SlotManager(llama_client=llama_client)
    resource_manager = ResourceManager(config=settings, llama_client=llama_client)

    await slot_manager.assign_watcher(settings.watcher_slot)
    await slot_manager.assign_main(settings.main_slot)

    rolling_context = RollingContext(
        llama_client=llama_client,
        slot_id=settings.watcher_slot,
        max_tokens=settings.watcher_context_size,
    )

    memory: LanceDBMemory | None
    try:
        memory = LanceDBMemory(
            db_path=settings.lancedb_path,
            embedding_model=settings.embedding_model,
        )
    except ImportError:
        memory = None

    watcher = WatcherAgent(
        slot_manager=slot_manager,
        rolling_context=rolling_context,
        memory=memory,  # type: ignore[arg-type]
        config=settings,
    )
    main_agent = MainAgent(
        slot_manager=slot_manager,
        memory=memory,  # type: ignore[arg-type]
        config=settings,
    )
    handoff = HandoffManager(watcher=watcher, main_agent=main_agent)
    return RuntimeState(
        handoff=handoff,
        watcher=watcher,
        main_agent=main_agent,
        resources=resource_manager,
    )


def _print_help() -> None:
    print("Commands:")
    print("  /help                   Show commands")
    print("  /quit                   Exit")
    print("  /agent auto|watcher|main  Set routing")
    print("  /mode idle|active       Set resource mode")
    print("  /status                 Show resource status")


async def repl(default_agent: str) -> None:
    runtime = await build_runtime()
    agent = default_agent

    print("SlothBrain CLI (non-web)")
    print("Type /help for commands.")

    while True:
        try:
            user_input = input(f"[{agent}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input == "/help":
            _print_help()
            continue
        if user_input == "/quit":
            print("Bye.")
            break
        if user_input.startswith("/agent "):
            next_agent = user_input.split(" ", 1)[1].strip().lower()
            if next_agent not in {"auto", "watcher", "main"}:
                print("Invalid agent. Use auto, watcher, or main.")
                continue
            agent = next_agent
            continue
        if user_input.startswith("/mode "):
            next_mode = user_input.split(" ", 1)[1].strip().lower()
            try:
                await runtime.resources.set_mode(next_mode)
                print(f"mode={runtime.resources.mode}")
            except ValueError as exc:
                print(str(exc))
            continue
        if user_input == "/status":
            stats = await runtime.resources.get_system_stats()
            print(stats)
            continue

        try:
            if agent == "watcher":
                response = await runtime.watcher.process(user_input)
                print(f"watcher> {response}")
            elif agent == "main":
                response = await runtime.main_agent.process(user_input)
                print(f"main> {response}")
            else:
                routed = await runtime.handoff.route(user_input)
                print(f"{routed['agent']}> {routed['response']}")
        except Exception as exc:
            print(f"error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SlothBrain terminal interface")
    parser.add_argument(
        "--agent",
        default="auto",
        choices=["auto", "watcher", "main"],
        help="Default routing strategy for chat messages",
    )
    args = parser.parse_args()
    asyncio.run(repl(default_agent=args.agent))


if __name__ == "__main__":
    main()
