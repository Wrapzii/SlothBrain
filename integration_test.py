import asyncio
from pathlib import Path

from backend.agents.main_agent import MainAgent
from backend.config import settings
from backend.core.llama_client import LlamaClient
from backend.core.slot_manager import SlotManager
from backend.tools.registry import ToolRegistry
from backend.tools.impl.web_fetch_tool import WebFetchTool
from backend.tools.impl.web_search_tool import WebSearchTool
from backend.tools.impl.file_tool import FileTool
from backend.tools.impl.workspace_index_tool import WorkspaceIndexTool

class Cfg:
    workspace_root = str(Path.cwd())

calls = []

class SpyTool:
    def __init__(self, inner, calls):
        self._inner = inner
        self._calls = calls
    @property
    def name(self):
        return self._inner.name
    @property
    def description(self):
        return self._inner.description
    @property
    def parameters_schema(self):
        return self._inner.parameters_schema
    async def execute(self, **kwargs):
        self._calls.append((self._inner.name, kwargs))
        return await self._inner.execute(**kwargs)

def safe(s):
    return (s or "").encode("ascii", "ignore").decode("ascii")

async def run_prompt(agent, label, prompt, calls, timeout=240):
    calls.clear()
    print("\n===", label, "===")
    print("PROMPT:", prompt)
    try:
        resp = await asyncio.wait_for(agent.process_direct(prompt), timeout=timeout)
        print("STATUS: ok")
    except asyncio.TimeoutError:
        resp = "TIMEOUT"
        print("STATUS: timeout")
    except Exception as e:
        resp = f"ERROR: {type(e).__name__}: {e}"
        print("STATUS: error", type(e).__name__, str(e))

    print("TOOLS CALLED:")
    if calls:
        for n, a in calls:
            print(" -", n, a)
    else:
        print(" - (none)")

    print("RESPONSE PREVIEW:")
    print(safe(resp)[:1200])
    return resp, list(calls)

async def main():
    print("Target:", f"http://{settings.llama_host}:{settings.llama_port}")
    client = LlamaClient(host=settings.llama_host, port=settings.llama_port)
    slot = SlotManager(llama_client=client)
    await slot.assign_main(settings.main_slot)

    reg = ToolRegistry()
    reg.register(SpyTool(WebFetchTool(), calls))
    reg.register(SpyTool(WebSearchTool(searxng_url=getattr(settings, 'searxng_url', '')), calls))
    reg.register(SpyTool(FileTool(config=Cfg(), workspace_index=WorkspaceIndexTool(indexer=None)), calls))

    agent = MainAgent(slot_manager=slot, memory=None, config=settings)
    agent.set_tool_registry(reg)

    r1, c1 = await run_prompt(agent, "BYTEBREW_WEB", "Visit https://bytebrew.cc and summarize what the company does in 2 sentences.", calls)
    r2, c2 = await run_prompt(agent, "RESEARCH_SEARXNG", "Research what SearxNG is using web tools and give 3 bullets with source URLs.", calls)
    r3, c3 = await run_prompt(agent, "LOCAL_FILES", "Check local workspace: list three files in backend/agents and confirm whether backend/agents/main_agent.py exists.", calls)

    agents_dir = Path("backend/agents")
    files = sorted([p.name for p in agents_dir.iterdir() if p.is_file()])
    print("\n=== GROUND TRUTH ===")
    print("backend/agents sample:", files[:10])
    print("main_agent.py exists:", (agents_dir / "main_agent.py").exists())

    print("\n=== FLAGS ===")
    print("bytebrew_used_web_tool:", any(n in {"web_fetch", "web_search"} for n, _ in c1))
    print("research_used_web_tool:", any(n in {"web_fetch", "web_search"} for n, _ in c2))
    print("local_used_file_tool:", any(n == "file" for n, _ in c3))
    print("bytebrew_mentions_machining:", ("machin" in (r1 or "").lower()) or ("cnc" in (r1 or "").lower()))
    print("local_mentions_main_agent:", "main_agent.py" in (r3 or ""))

asyncio.run(main())
