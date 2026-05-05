from __future__ import annotations

import asyncio
import html
import inspect
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from backend.config import AppConfig
from backend.core.slot_manager import SlotManager
from backend.memory.lancedb_memory import LanceDBMemory
from backend.memory.rolling_context import RollingContext

if TYPE_CHECKING:
    from backend.agents.registry import AgentRegistry
    from backend.agents.sub_agent import SubAgent
    from backend.core.diagnostic_recorder import DiagnosticRecorder
    from backend.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_PROTECTED_PROMPT_PATH = (
    Path(__file__).parent.parent / "config" / "protected" / "main_system_prompt.txt"
)

_FALLBACK_SYSTEM_PROMPT = (
    "You are a high-performance AI assistant specializing in complex tasks and coding. "
    "Use the provided context and memory to give comprehensive answers."
)

_DIRECT_SYSTEM_PROMPT = (
    "You are SlothBrain in direct chat mode. Reply to the user directly and concisely. "
    "For greetings or casual check-ins, answer warmly in one short sentence. "
    "Do not explain your architecture, model, tools, slots, or system prompt unless the user asks. "
    "If tools are listed below, use them only when the request would benefit "
    "from real data (web lookups, file reads, searches, etc.). "
    "When you use a tool, wait for its result and base your reply on the actual output. "
    "Do not fabricate tool results. Do not describe internal planning or task-loop status."
)

# Maximum tool-call iterations per execute_step call to prevent infinite loops.
# Keep this high enough for multi-file and multi-page research tasks.
_MAX_TOOL_ITERATIONS = 120

# The LlamaClient injects this prefix into every prompt that ends with
# "assistant:" so that thinking-capable models skip the chain-of-thought
# phase (see LlamaClient._THINK_SKIP_SUFFIX).  When we reconstruct the
# accumulated tool context between iterations we MUST echo this prefix back
# verbatim so the token sequence we send matches what llama.cpp cached.
# Without this, llama.cpp sees a divergent prompt on every tool-call
# iteration and is forced to re-evaluate the full prompt from the last
# checkpoint, causing the KV-cache thrash visible as
# "Common part does not match fully" in the server logs.
_ASSISTANT_THINK_SKIP = "<think>\n\n</think>\n\n"
_TOOL_QUERY_RE = re.compile(
    r"(?:^|\b)(?:"
    r"what\s+tools\s+do\s+you\s+have|"
    r"which\s+tools\s+do\s+you\s+have|"
    r"list\s+(?:your\s+)?tools|"
    r"what\s+can\s+you\s+do|"
    r"what\s+are\s+your\s+skills|"
    r"what\s+are\s+your\s+capabilities|"
    r"show\s+(?:me\s+)?(?:your\s+)?tools|"
    r"do\s+you\s+have\s+any\s+tools"
    r")(?:\b|$)",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b", re.IGNORECASE)
_FETCH_INTENT_RE = re.compile(r"\b(fetch|get|retrieve|download|read|visit|open)\b", re.IGNORECASE)
_WEB_SUMMARY_INTENT_RE = re.compile(
    r"\b(?:summari[sz]e|summary|what\s+is|what'?s|tell\s+me\s+about|explain|describe|analy[sz]e)\b"
    r".{0,80}\b(?:site|website|web\s*page|url|domain|page)\b|"
    r"\b(?:site|website|web\s*page|url|domain|page)\b.{0,80}"
    r"\b(?:summari[sz]e|summary|about|overview|details?)\b",
    re.IGNORECASE,
)
_DEEPER_SUMMARY_INTENT_RE = re.compile(
    r"\b(?:more|deeper|indepth|in-depth|detailed|detail|expand|better|thorough)\b",
    re.IGNORECASE,
)
_PROVENANCE_INTENT_RE = re.compile(
    r"\b(?:how|where)\s+did\s+you\s+(?:get|find|know|see)\b|"
    r"\b(?:how|where)\s+(?:was|is)\s+that\s+(?:found|derived|sourced)\b|"
    r"\bwhat\s+(?:source|sources|tool|tools)\s+did\s+you\s+use\b",
    re.IGNORECASE,
)
_EXPLICIT_TOOL_INTENT_RE = re.compile(
    r"\b(?:web\s*fetch|use\s+(?:a\s+)?tool|run\s+(?:a\s+)?tool|look\s*up|lookup|search\s+(?:the\s+)?web|browse)\b",
    re.IGNORECASE,
)
_DELEGATION_INTENT_RE = re.compile(
    r"\b(?:delegate|sub[-\s]?agent|spawn\s+(?:an?\s+)?agent|parallel\s+agent|another\s+agent|speciali[sz]ed\s+agent)\b",
    re.IGNORECASE,
)
_SCHEDULE_INTENT_RE = re.compile(
    r"\b(?:schedule|remind\s+me|timer|every\s+(?:morning|day|hour|week)|daily|weekly|tomorrow|cron)\b",
    re.IGNORECASE,
)
_CASUAL_CHAT_RE = re.compile(
    r"^\s*(?:hi|hello|hey|yo|hiya|sup|what'?s up|how are you|hello how are you|hey how are you)[\s.!?]*$",
    re.IGNORECASE,
)
_ARCHITECTURE_DRIFT_RE = re.compile(
    r"\b(?:main agent|distributed system|sub-agents?|central orchestrator|llama\.?cpp|kv-cache|slots?|architecture|capabilities)\b",
    re.IGNORECASE,
)
_LOCAL_FILE_HINT_RE = re.compile(
    r"\b(file|folder|directory|filesystem|local|workspace|github\s+directory|github|documents|downloads|desktop|"
    r"repo(?:sitory)?|project(?:s)?)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_TOOL_NAME_RE = re.compile(r"^\s*(?:<[^>]+>|name|tool)\s*$", re.IGNORECASE)
_DESKTOP_VISION_TOOLS = {"screenshot", "ui", "image_analysis"}
_AGENT_HANDOFF_TOOLS = {"sub_agent", "agent_list", "session"}
_TOOL_BLOCK_RE = re.compile(
    r"</?(?:tool_call|tool_result|fetch|fetch_result|verify|sweep|think|sloth)>"
    r"|thinking\s+process:"
    r"|\bself-correction/verification\b"
    r"|\bsimulated content\b",
    re.IGNORECASE,
)
_RAW_HTML_RE = re.compile(
    r"<(?:!doctype|html|head|body|meta|link|script|style|title|div|section|header|footer|span|p)\b|"
    r"&(?:nbsp|amp|lt|gt|quot);",
    re.IGNORECASE,
)
_FILESYSTEM_TOOL_ALIASES = {
    "filesystem",
    "filesystem/search",
    "filesystem/list",
    "fs",
    "fs/search",
    "local_file",
    "local_filesystem",
    "path_search",
}
_SHELL_TOOL_ALIASES = {
    "run",
    "terminal",
    "terminal/run",
    "command",
    "cmd",
    "powershell",
    "bash",
}
_LOCAL_ROOTS = [
    ("home", str(Path.home())),
    ("desktop", str(Path.home() / "Desktop")),
    ("documents", str(Path.home() / "Documents")),
    ("downloads", str(Path.home() / "Downloads")),
    ("github", str(Path.home() / "Documents" / "GitHub")),
]

_MEMORY_ROLE_LINE_RE = re.compile(r"(?im)^\s*(?:user|assistant)\s*:\s*.*$")
_MEMORY_META_LINE_RE = re.compile(
    r"(?im)^\s*(?:overall\s+task\s*:|you\s+are\s+executing\s+a\s+multi-step\s+task|based\s+on\s+the\s+tool\s+result\(s\)\s+above).*$"
)
_SHORT_FOLLOWUP_EDIT_RE = re.compile(
    r"\b(?:compact|shorten|summari[sz]e|rephrase|rewrite|make\s+(?:it\s+)?short(?:er)?|brief(?:er)?|tl;dr)\b",
    re.IGNORECASE,
)

# Tool payload guardrails for websocket/event safety.
_MAX_EVENT_TEXT_CHARS = 600
_MAX_LIST_ITEMS = 20
_MAX_MODEL_RESPONSE_PREVIEW_CHARS = 240
# Tool output limits: event payloads vs. model-facing prompts.
# Event payloads can be compact (logging/discord), but prompts need
# enough room for larger fetches and multi-source summarization.
_MAX_TOOL_PROMPT_TEXT_CHARS = 20000
_MAX_TOOL_PROMPT_LIST_ITEMS = 250
_MAX_TOOL_CONTEXT_CHARS = 96000
# Direct mode should not be the limiting factor for large grounded summaries.
_DIRECT_MAX_TOKENS_CAP = 4096
_HEAVY_B64_KEYS = {
    "annotated_png_b64",
    "image_b64",
    "png_b64",
    "jpeg_b64",
    "jpg_b64",
}


def _truncate_text(value: str, limit: int = _MAX_EVENT_TEXT_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f" ...[truncated {len(value) - limit} chars]"


def _sanitize_tool_payload(value):
    """Return a compact, JSON-serializable tool payload for events/prompts."""
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            if k in _HEAVY_B64_KEYS and isinstance(v, str):
                out[k] = f"[omitted base64 payload: {len(v)} chars]"
                continue
            out[k] = _sanitize_tool_payload(v)
        return out
    if isinstance(value, list):
        trimmed = value[:_MAX_LIST_ITEMS]
        result = [_sanitize_tool_payload(v) for v in trimmed]
        if len(value) > _MAX_LIST_ITEMS:
            result.append(f"...[{len(value) - _MAX_LIST_ITEMS} more items]")
        return result
    if isinstance(value, str):
        return _truncate_text(value)
    return value


def _sanitize_tool_payload_for_prompt(value):
    """Compact tool payload for model-facing prompt context.

    Keep this significantly smaller than event payloads to avoid large prompt
    growth and decode slowdowns in multi-iteration tool loops.
    """
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            if k in _HEAVY_B64_KEYS and isinstance(v, str):
                out[k] = f"[omitted base64 payload: {len(v)} chars]"
                continue
            out[k] = _sanitize_tool_payload_for_prompt(v)
        return out
    if isinstance(value, list):
        trimmed = value[:_MAX_TOOL_PROMPT_LIST_ITEMS]
        result = [_sanitize_tool_payload_for_prompt(v) for v in trimmed]
        if len(value) > _MAX_TOOL_PROMPT_LIST_ITEMS:
            result.append(f"...[{len(value) - _MAX_TOOL_PROMPT_LIST_ITEMS} more items]")
        return result
    if isinstance(value, str):
        return _truncate_text(value, _MAX_TOOL_PROMPT_TEXT_CHARS)
    return value


def _sanitize_direct_response(text: str) -> str:
    """Clean up the final direct-mode response before returning to the caller.

    Tool calls executed inside the tool-call loop are fine — their results are
    already incorporated.  We only replace a dangling bare <tool_call> block
    that is the *entire* final answer (iteration limit reached with no prose).
    """
    stripped = (text or "").strip()
    if not stripped:
        return stripped
    _BARE_TOOL_CALL_RE = re.compile(
        r"^\s*<tool_call>.*?</tool_call>\s*$", re.DOTALL | re.IGNORECASE
    )
    if _BARE_TOOL_CALL_RE.match(stripped):
        return "I tried to use a tool but reached the iteration limit. Please rephrase or use /task."
    if _TOOL_BLOCK_RE.search(stripped):
        cleaned = re.sub(r"(?is)<[^>]+>", " ", stripped)
        cleaned = re.sub(r"(?im)^\s*(?:url\s*:|\.\.\.|thinking\s+process:|self-correction/verification).*?$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if re.search(r"\b(?:simulated content|self-correction/verification)\b", cleaned, re.IGNORECASE):
            cleaned = ""
        if cleaned and len(cleaned) > 20:
            return cleaned
        return "I couldn't produce a clean direct response. Please try again or use /task for tool-heavy work."
    return stripped


def _needs_direct_repair(text: str, *, casual_chat: bool = False) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if casual_chat and (len(stripped) > 180 or _ARCHITECTURE_DRIFT_RE.search(stripped)):
        return True
    if _TOOL_BLOCK_RE.search(stripped):
        return True
    if _RAW_HTML_RE.search(stripped):
        return True
    if stripped.count("<") >= 4 and stripped.count(">") >= 4:
        return True
    return False


def _dedupe_repeated_response(text: str) -> str:
    stripped = (text or "").strip()
    if len(stripped) < 80:
        return stripped
    midpoint = len(stripped) // 2
    left = stripped[:midpoint].strip()
    right = stripped[midpoint:].strip()
    if left and right and right.startswith(left[: min(80, len(left))]):
        return left
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", stripped) if p.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        key = re.sub(r"\s+", " ", paragraph).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(paragraph)
    return "\n\n".join(deduped) if deduped else stripped


def _htmlish_to_plain_summary(text: str) -> str:
    raw = (text or "").replace("\\r", "\n").replace("\\n", "\n")
    if not raw.strip():
        return ""
    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip()
    desc = ""
    desc_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if not desc_match:
        desc_match = re.search(
            r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']',
            raw,
            re.IGNORECASE | re.DOTALL,
        )
    if desc_match:
        desc = re.sub(r"\s+", " ", desc_match.group(1)).strip()

    cleaned = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", raw)
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&amp;", "&", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&lt;", "<", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&gt;", ">", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"&quot;", '"', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    parts: list[str] = []
    if title:
        parts.append(f"Page title: {title}.")
    if desc:
        parts.append(desc)
    elif cleaned:
        parts.append(cleaned[:500].strip())
    if not parts:
        return ""
    return " ".join(parts)


def _normalize_web_target(target: str) -> str:
    cleaned = (target or "").strip().strip("<>()[]{}\"'.,;:!?\n\r\t ")
    if not cleaned:
        return ""
    if cleaned.lower().startswith(("http://", "https://")):
        return cleaned
    return f"https://{cleaned}"


def _extract_url_or_domain(text: str, conversation_context: Optional[list[str]] = None) -> str:
    haystacks = [text or ""]
    if conversation_context:
        haystacks.extend(reversed([line or "" for line in conversation_context[-8:]]))
    for item in haystacks:
        url_match = _URL_RE.search(item)
        if url_match:
            return _normalize_web_target(url_match.group(0))
        domain_match = _DOMAIN_RE.search(item)
        if domain_match:
            return _normalize_web_target(domain_match.group(0))
    return ""


def _visible_html_items(raw: str, tag_names: str) -> list[str]:
    items: list[str] = []
    pattern = re.compile(fr"(?is)<(?:{tag_names})\b[^>]*>(.*?)</(?:{tag_names})>")
    for match in pattern.finditer(raw or ""):
        text = re.sub(r"(?is)<[^>]+>", " ", match.group(1))
        text = html.unescape(re.sub(r"\s+", " ", text)).strip()
        if text:
            items.append(text)
    return items


def _clean_visible_web_text(raw: str) -> str:
    cleaned = re.sub(r"(?is)<script.*?</script>|<style.*?</style>|<svg.*?</svg>|<noscript.*?</noscript>", " ", raw or "")
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"(?is)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?is)</(?:p|div|section|article|li|h[1-6])>", "\n", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _dedupe_text_items(items: list[str], *, max_items: int = 12) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", item).strip(" -|")
        if len(cleaned) < 4:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        if re.fullmatch(r"(?:home|menu|close|search|submit|learn more|read more|skip to content)", key):
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= max_items:
            break
    return result


def _render_web_fetch_answer(user_input: str, result_dict: dict) -> str:
    if not result_dict or not bool(result_dict.get("ok")):
        error = str((result_dict or {}).get("error") or "").strip()
        return f"web_fetch failed: {error}" if error else ""

    output = result_dict.get("output")
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False, indent=2, default=str)[:4000]
    if not isinstance(output, str) or not output.strip():
        return ""

    raw = output
    deep = bool(_DEEPER_SUMMARY_INTENT_RE.search(user_input or ""))
    title = ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    if title_match:
        title = html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()

    desc = ""
    desc_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if not desc_match:
        desc_match = re.search(
            r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']',
            raw,
            re.IGNORECASE | re.DOTALL,
        )
    if desc_match:
        desc = html.unescape(re.sub(r"\s+", " ", desc_match.group(1))).strip()

    headings = _dedupe_text_items(_visible_html_items(raw, "h1|h2|h3"), max_items=8 if deep else 5)
    body_items = _dedupe_text_items(
        _visible_html_items(raw, "p|li"),
        max_items=12 if deep else 6,
    )
    visible = _clean_visible_web_text(raw)

    lines: list[str] = []
    if title:
        lines.append(f"Page title: {title}.")
    if desc:
        lines.append(f"Overview: {desc}")
    elif visible:
        lines.append(f"Overview: {visible[:700 if deep else 420].strip()}")
    if headings:
        lines.append("Notable sections: " + "; ".join(headings) + ".")
    if body_items:
        detail_limit = 7 if deep else 4
        lines.append("Details:")
        lines.extend(f"- {item}" for item in body_items[:detail_limit])

    if not lines and visible:
        return visible[:1200 if deep else 700].strip()
    return "\n".join(lines).strip()


def _render_tool_result_answer(user_input: str, tool_name: str, result_dict: dict) -> str:
    if not result_dict or not bool(result_dict.get("ok")):
        if tool_name == "web_fetch" and result_dict:
            error = str(result_dict.get("error") or "").strip()
            return (
                "web_fetch could not retrieve the requested URL"
                + (f": {error}" if error else ".")
                + " I do not have page content to summarize from that fetch."
            )
        return ""
    output = result_dict.get("output")
    if tool_name == "file" and isinstance(output, dict):
        path = str(output.get("absolute_path") or output.get("path") or "").strip()
        entries = output.get("entries")
        if isinstance(entries, list):
            dirs = [str(e.get("name")) for e in entries if isinstance(e, dict) and e.get("type") == "dir"]
            files = [str(e.get("name")) for e in entries if isinstance(e, dict) and e.get("type") == "file"]
            bits = []
            if dirs:
                bits.append("Folders: " + ", ".join(dirs[:60]))
            if files:
                bits.append("Files: " + ", ".join(files[:30]))
            prefix = f"Found path: {path}" if path else "Found filesystem results"
            return prefix + ("\n" + "\n".join(bits) if bits else "")
        if "exists" in output:
            return f"Path exists: {bool(output.get('exists'))}\nPath: {output.get('path', '')}"
        if "written" in output:
            return f"Wrote file: {output.get('written')}"
        if "appended" in output:
            return f"Appended to file: {output.get('appended')}"
        if "deleted" in output:
            return f"Deleted file: {output.get('deleted')}"
    if tool_name == "file" and isinstance(output, str):
        return output[:4000]
    if tool_name == "shell" and isinstance(output, dict):
        stdout = str(output.get("stdout") or "").strip()
        stderr = str(output.get("stderr") or "").strip()
        if stdout:
            return stdout[:4000]
        if stderr and not result_dict.get("ok"):
            return f"Shell command failed:\n{stderr[:2000]}"
    if tool_name == "web_fetch":
        return _render_web_fetch_answer(user_input, result_dict)
    if tool_name == "sub_agent" and isinstance(output, dict):
        summary = str(output.get("handoff_summary") or "").strip()
        response = str(output.get("response") or "").strip()
        agent_id = str(output.get("agent_id") or "").strip()
        slot_id = output.get("slot_id")
        body = summary or response
        if not body:
            return ""
        prefix = "Sub-agent handoff summary"
        if agent_id or slot_id is not None:
            details: list[str] = []
            if agent_id:
                details.append(f"agent={agent_id}")
            if slot_id is not None:
                details.append(f"slot={slot_id}")
            prefix += f" ({', '.join(details)})"
        return f"{prefix}:\n{body[:4000]}"
    return ""


def _local_filesystem_context() -> str:
    lines = [
        "Local filesystem context:",
        "- You are running on the user's Windows machine with trusted local access.",
        "- For arbitrary folder discovery, use ONLY real tools named file or shell; do not invent tools.",
        "- The file tool accepts absolute paths in this trusted runtime.",
        "- Useful starting roots:",
    ]
    for label, path in _LOCAL_ROOTS:
        lines.append(f"  - {label}: {path}")
    lines.append("- To find an arbitrary folder, list or shell-search from these roots, then report the exact path.")
    return "\n".join(lines)


def _root_for_text(text: str) -> str:
    lower = (text or "").lower()
    for label, path in _LOCAL_ROOTS:
        if label in lower:
            return path
    return str(Path.home())


def _coerce_tool_args(tool_name: str, tool_args: dict, user_input: str) -> dict:
    args = dict(tool_args or {})
    if tool_name == "web_fetch":
        requested_url = _normalize_web_target(str(args.get("url") or ""))
        user_target = _extract_url_or_domain(user_input)
        if requested_url:
            requested_host = re.sub(r"^https?://", "", requested_url, flags=re.IGNORECASE).split("/", 1)[0].lower()
            user_host = re.sub(r"^https?://", "", user_target, flags=re.IGNORECASE).split("/", 1)[0].lower() if user_target else ""
            if user_host and requested_host == f"www.{user_host}":
                args["url"] = user_target
            else:
                args["url"] = requested_url
        elif user_target:
            args["url"] = user_target
        return args
    if tool_name == "file" and ("action" not in args or "path" not in args):
        args.setdefault("action", "list")
        args.setdefault("path", _root_for_text(user_input))
        return args
    if tool_name == "shell" and not args.get("command"):
        query = str(args.get("query") or args.get("q") or args.get("name") or "").strip()
        if not query:
            query = (user_input or "").strip()
        escaped_query = query.replace("'", "''")[:120]
        roots = "; ".join(
            f"'{path.replace(chr(39), chr(39) + chr(39))}'"
            for _, path in _LOCAL_ROOTS
        )
        args["command"] = (
            "powershell -NoProfile -Command "
            f"\"$roots=@({roots}); "
            f"$q='{escaped_query}'; "
            "$roots | Where-Object { Test-Path $_ } | ForEach-Object { "
            "Get-ChildItem -LiteralPath $_ -Directory -Recurse -ErrorAction SilentlyContinue "
            "| Where-Object { $_.Name -like ('*' + $q + '*') } "
            "| Select-Object -First 25 -ExpandProperty FullName }\""
        )
        args.setdefault("timeout", 45)
    return args


def _extract_folder_search_query(text: str) -> str:
    raw = (text or "").strip()
    match = re.search(
        r"\b(?:find|locate|search\s+for|look\s+for|open|get\s+(?:me\s+)?to)\s+(?:my\s+|the\s+)?(.+?)\s+"
        r"(?:folder|directory|project|repo(?:sitory)?)\b",
        raw,
        re.IGNORECASE,
    )
    if match:
        query = match.group(1)
    else:
        query = raw
    query = re.sub(
        r"\b(?:find|locate|search|look|for|my|the|folder|directory|project|projects|repo|repository|"
        r"files|list|show|check|open|create|make|build|write|edit|modify|delete)\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    query = " ".join(query.split()).strip(" .")
    return query


def _is_local_retrieval_request(text: str) -> bool:
    lower = (text or "").lower()
    if re.search(r"\b(?:create|make|build|write|edit|modify|delete|remove)\b", lower):
        return False
    return bool(re.search(r"\b(?:find|locate|list|show|check|open|where|search)\b", lower))


async def _try_local_filesystem_fast_path(tool_registry, text: str) -> str:
    if tool_registry is None or not _is_local_retrieval_request(text):
        return ""
    file_tool = tool_registry.get("file")
    shell_tool = tool_registry.get("shell")
    if file_tool is None:
        return ""

    root = _root_for_text(text)
    lower = (text or "").lower()
    query = _extract_folder_search_query(text)
    root_labels = {label for label, _ in _LOCAL_ROOTS if label in lower}

    # If the user asks for a known root or its projects/files, list it directly.
    if root_labels or re.search(r"\b(?:projects?|files?|folders?|directories)\b", lower):
        result = await file_tool.execute(action="list", path=root)
        result_dict = result.to_dict()
        rendered = _render_tool_result_answer(text, "file", result_dict)
        if rendered:
            return rendered

    # Otherwise search for a folder name across useful local roots.
    if query and shell_tool is not None:
        args = _coerce_tool_args("shell", {"query": query}, text)
        result = await shell_tool.execute(**args)
        result_dict = result.to_dict()
        rendered = _render_tool_result_answer(text, "shell", result_dict)
        if rendered:
            return rendered

    return ""


def _trim_tool_context(context: str) -> str:
    """Bound accumulated tool-context size to avoid prompt-eval blowups.

    Large tool outputs across iterations can force expensive full prompt
    re-processing and lead to client-side cancellations. Keep only the most
    recent tail, anchored to a tool/result boundary when possible.
    """
    if len(context) <= _MAX_TOOL_CONTEXT_CHARS:
        return context

    tail = context[-_MAX_TOOL_CONTEXT_CHARS :]
    anchor = tail.find("<tool_result>")
    if anchor == -1:
        anchor = tail.find("<tool_call>")
    if anchor > 0:
        return tail[anchor:]
    return tail


def _sanitize_memory_snippet(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    parts: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line or _MEMORY_META_LINE_RE.match(line):
            continue
        role_match = re.match(r"^(user|assistant)\s*:\s*(.*)$", line, re.IGNORECASE)
        if role_match:
            role = role_match.group(1).lower()
            content = role_match.group(2).strip()
            if content:
                parts.append(f"{role} said: {content}")
            continue
        parts.append(line)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _resolve_tool_name(
    requested_name: str,
    tool_args: dict,
    *,
    active_tools: list,
    user_input: str,
) -> str:
    """Best-effort resolution for malformed/placeholder tool names.

    Some models occasionally emit placeholders like "<name>" even when tools
    are provided. Resolve these to a concrete routed tool to keep execution
    grounded instead of failing the whole turn.
    """
    name = (requested_name or "").strip()
    active_names = [t.name for t in active_tools]
    if name in active_names:
        return name

    lower_name = name.lower()
    for candidate in active_names:
        if lower_name == candidate.lower():
            return candidate

    if lower_name in _FILESYSTEM_TOOL_ALIASES and "search" in lower_name and "shell" in active_names:
        return "shell"
    if lower_name in _FILESYSTEM_TOOL_ALIASES and "file" in active_names:
        return "file"
    if lower_name in _SHELL_TOOL_ALIASES and "shell" in active_names:
        return "shell"

    if name and not _PLACEHOLDER_TOOL_NAME_RE.match(name):
        # Keep the original unknown name so the caller can report a proper error.
        return name

    args = tool_args or {}
    if any(k in args for k in ("url", "urls")):
        if "web_fetch" in active_names:
            return "web_fetch"
    if any(k in args for k in ("query", "q", "keywords")):
        if "web_search" in active_names:
            return "web_search"
    if any(k in args for k in ("path", "file_path", "action")):
        if "file" in active_names:
            return "file"

    ui = (user_input or "").lower()
    if ("http://" in ui or "https://" in ui or "website" in ui or "web" in ui) and "web_fetch" in active_names:
        return "web_fetch"
    if ("search" in ui or "research" in ui or "look up" in ui or "lookup" in ui) and "web_search" in active_names:
        return "web_search"
    if _LOCAL_FILE_HINT_RE.search(user_input or "") and "file" in active_names:
        return "file"

    return active_names[0] if active_names else name


def _filter_agent_handoff_tools(tools: list, text: str) -> list:
    if _DELEGATION_INTENT_RE.search(text or ""):
        return tools
    return [tool for tool in tools if tool.name not in _AGENT_HANDOFF_TOOLS]


def _load_protected_prompt() -> str:
    """Load the main agent's system prompt from the protected file (read-only)."""
    try:
        return _PROTECTED_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning(
            "Could not read protected system prompt; using fallback."
        )
        return _FALLBACK_SYSTEM_PROMPT


def _parse_plan(response: str) -> dict:
    """Parse a plan_task response into approach and steps.

    Tries JSON parsing first (preferred — the prompt requests JSON output).
    Falls back to extracting a numbered list, then a ``STEPS:`` header, and
    finally treats the whole response as a single step.
    """
    stripped = response.strip()

    # ── Attempt 1: JSON parse ─────────────────────────────────────────────
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        json_candidate = fence_match.group(1)
    else:
        obj_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
        json_candidate = obj_match.group(0) if obj_match else stripped

    try:
        data = json.loads(json_candidate)
        approach = str(data.get("approach", "")).strip()
        raw_steps = data.get("steps", [])
        if isinstance(raw_steps, list) and raw_steps:
            steps = [str(s).strip() for s in raw_steps if str(s).strip()]
            if steps:
                return {"approach": approach, "steps": steps[:10]}
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # ── Attempt 2: Regex for numbered list items ──────────────────────────
    approach = ""
    approach_match = re.search(r"approach:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if approach_match:
        approach = approach_match.group(1).strip()

    steps = [
        s.strip()
        for s in re.findall(r"^\d+\.\s+(.+)", response, re.MULTILINE)
        if s.strip()
    ]

    if not steps:
        # ── Attempt 3: Lines after STEPS: header ─────────────────────────
        in_steps = False
        for line in response.splitlines():
            stripped_line = line.strip()
            if re.match(r"steps?\s*:", stripped_line, re.IGNORECASE):
                in_steps = True
                continue
            if in_steps and stripped_line:
                steps.append(stripped_line.lstrip("-•*").strip())

    if not steps:
        # ── Attempt 4: Last resort ────────────────────────────────────────
        steps = [response.strip()]

    return {"approach": approach, "steps": steps[:10]}


class MainAgent:
    """High-capability agent responsible for planning and executing complex tasks.

    The MainAgent runs on the main inference slot (higher context window) and
    provides three core capabilities:

    1. **Task planning** — ``plan_task`` breaks a natural-language task into an
       ordered list of actionable steps (JSON output for reliable parsing).
    2. **Step execution** — ``execute_step`` executes one step with accumulated
       context from previous steps, maintaining coherent long-running task state.
    3. **Sub-agent delegation** — ``spawn_sub_agent`` creates task-specialised
       ``SubAgent`` instances via the ``AgentRegistry`` for parallel or
       specialised work.

    Memory retrieval is performed before every ``process`` call so relevant
    past context is always available to the model.

    TODO: Add a tool-calling layer so the MainAgent can invoke tools (code
          execution, file I/O, web search) as part of execute_step.
    """

    def __init__(
        self,
        slot_manager: SlotManager,
        memory: Optional[LanceDBMemory],
        config: AppConfig,
    ) -> None:
        self._slot_manager = slot_manager
        self._memory = memory
        self._config = config
        self.slot_id = config.main_slot
        self.system_prompt = _load_protected_prompt()
        # Injected after construction so we avoid circular imports
        self._registry: AgentRegistry | None = None
        self._tool_registry: "ToolRegistry | None" = None
        summarize_at = max(1200, int((config.main_context_size or 4096) * 0.20))
        summary_timeout = min(
            2.0,
            max(0.25, float(getattr(config, "llama_completion_timeout_seconds", 600.0)) / 600.0),
        )
        self._direct_rolling_context = RollingContext(
            llama_client=self._slot_manager._client,
            slot_id=self.slot_id,
            max_tokens=config.main_context_size or 4096,
            summarize_at=summarize_at,
            summary_timeout_seconds=summary_timeout,
        )
        self._agentic_rolling_context = RollingContext(
            llama_client=self._slot_manager._client,
            slot_id=self.slot_id,
            max_tokens=config.main_context_size or 4096,
            summarize_at=summarize_at,
            summary_timeout_seconds=summary_timeout,
        )
        # Optional diagnostic recorder – injected after construction.
        self._diagnostic: "DiagnosticRecorder | None" = None

    def set_registry(self, registry: "AgentRegistry") -> None:
        """Inject the AgentRegistry so MainAgent can spawn sub-agents."""
        self._registry = registry

    def set_tool_registry(self, tool_registry: "ToolRegistry") -> None:
        """Inject the ToolRegistry."""
        self._tool_registry = tool_registry

    def set_diagnostic_recorder(self, recorder: "DiagnosticRecorder | None") -> None:
        """Inject an optional DiagnosticRecorder for offline run analysis."""
        self._diagnostic = recorder

    async def _append_rolling_context(self, ctx: RollingContext, role: str, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        try:
            await ctx.add_message(role, text)
        except Exception as exc:
            logger.warning("RollingContext add_message failed: %s", exc.__class__.__name__)

    def _schedule_direct_memory_store(
        self,
        user_input: str,
        response: str,
        tool_context: str = "",
    ) -> None:
        if self._memory is None:
            return
        text = f"user: {user_input}\nassistant: {response}"
        if tool_context.strip():
            text += f"\ncontext: {tool_context.strip()}"
        try:
            asyncio.create_task(
                self._memory.store(
                    text=text,
                    metadata={"agent": "main", "slot": self.slot_id, "mode": "direct", "kind": "turn"},
                )
            )
        except Exception:
            pass

    def _rolling_context_prompt(self, ctx: RollingContext, header: str) -> str:
        prompt = ctx.get_context_prompt().strip()
        if not prompt:
            return ""
        return f"{header}:\n{prompt}"

    def _augment_sub_agent_task(self, task: str, context_lines: list[str] | None = None) -> str:
        context_bits: list[str] = []
        if context_lines:
            context_bits.extend(line.strip() for line in context_lines[-6:] if line and line.strip())
        rolling = self._rolling_context_prompt(
            self._direct_rolling_context,
            "Recent main-agent conversation",
        )
        if rolling:
            context_bits.append(rolling)
        if not context_bits:
            return task
        context_text = "\n".join(context_bits)
        return (
            f"{task}\n\n"
            "Context handed off from the main agent:\n"
            f"{context_text}\n\n"
            "Use this context only as background; answer the delegated task directly."
        )

    # ------------------------------------------------------------------
    # Sub-agent delegation
    # ------------------------------------------------------------------

    def spawn_sub_agent(
        self,
        preset_id: str,
        task_description: str,
        max_tokens: int | None = None,
    ) -> "SubAgent":
        """Spawn a sub-agent with task-appropriate resource budgets.

        Sub-agents run on explicitly assigned non-main slots. Context size is
        controlled by llama.cpp launch settings and slot partitioning.

        Raises RuntimeError if the registry is not set or max_slots exceeded.
        """
        if self._registry is None:
            raise RuntimeError("AgentRegistry not injected into MainAgent")
        return self._registry.spawn(
            preset_id=preset_id,
            max_tokens_override=max_tokens,
            task_description=task_description,
        )

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    async def process(
        self,
        user_input: str,
    ) -> str:
        memory_results: list[dict] = []
        if self._memory is not None:
            try:
                memory_results = await self._memory.search_advanced(
                    query=user_input,
                    limit=5,
                    metadata_filter={"agent": "main", "mode": "agentic", "kind": "turn"},
                    candidate_pool=40,
                )
                if not memory_results:
                    memory_results = await self._memory.search_advanced(
                        query=user_input,
                        limit=5,
                        metadata_filter={"agent": "main", "kind": "turn"},
                        candidate_pool=40,
                    )
            except Exception as exc:
                logger.warning("MainAgent memory search failed: %s", exc.__class__.__name__)

        memory_context = ""
        if memory_results:
            sanitized: list[str] = []
            for r in memory_results:
                text = _sanitize_memory_snippet(str(r.get("text", "")))
                if text:
                    sanitized.append(f"- {text}")
            snippets = "\n".join(sanitized)
            memory_context = f"\n\nRelevant past context:\n{snippets}"

        full_prompt = (
            f"system: {self.system_prompt}"
            f"{memory_context}"
            "\n\n"
            f"user: {user_input}\nassistant:"
        )

        response = await self._slot_manager.send_to_main(
            full_prompt, max_tokens=self._config.main_context_size or 4096
        )

        if self._memory is not None:
            try:
                await self._memory.store(
                    text=f"user: {user_input}\nassistant: {response}",
                    metadata={"agent": "main", "slot": self.slot_id, "mode": "agentic", "kind": "turn"},
                )
            except Exception as exc:
                logger.warning("MainAgent memory store failed: %s", exc.__class__.__name__)

        return response

    async def process_direct(
        self,
        user_input: str,
        conversation_context: Optional[list[str]] = None,
    ) -> str:
        """Direct chat path with tool awareness.

        Tools are semantically routed from the registry and injected into the
        prompt so the model knows what it can call.  If the model emits a
        <tool_call> block the result is fed back and the model produces a
        final answer grounded in real data rather than hallucination.
        """
        latest_user_input = (user_input or "").strip()
        if _TOOL_QUERY_RE.search(latest_user_input):
            return self._describe_direct_capabilities()

        casual_chat = bool(_CASUAL_CHAT_RE.match(latest_user_input))
        web_target = _extract_url_or_domain(latest_user_input, conversation_context)
        current_web_target = _extract_url_or_domain(latest_user_input)
        web_summary_intent = bool(_WEB_SUMMARY_INTENT_RE.search(latest_user_input))
        explicit_tool_intent = bool(_EXPLICIT_TOOL_INTENT_RE.search(latest_user_input))
        delegation_intent = bool(_DELEGATION_INTENT_RE.search(latest_user_input))
        provenance_intent = bool(_PROVENANCE_INTENT_RE.search(latest_user_input))
        deeper_web_followup_intent = bool(_DEEPER_SUMMARY_INTENT_RE.search(latest_user_input)) and bool(web_target)
        web_intent = bool(web_target) and (
            bool(current_web_target)
            or (bool(_FETCH_INTENT_RE.search(latest_user_input)) and not provenance_intent)
            or web_summary_intent
            or explicit_tool_intent
            or deeper_web_followup_intent
        )
        local_file_intent = bool(_LOCAL_FILE_HINT_RE.search(latest_user_input)) and not web_intent
        schedule_intent = bool(_SCHEDULE_INTENT_RE.search(latest_user_input))
        short_followup_edit_intent = bool(_SHORT_FOLLOWUP_EDIT_RE.search(latest_user_input)) and len(latest_user_input) <= 120
        if (short_followup_edit_intent or deeper_web_followup_intent) and web_target:
            web_intent = True
            web_summary_intent = True
            local_file_intent = False
        tool_intent = web_intent or local_file_intent or explicit_tool_intent or schedule_intent or delegation_intent

        if local_file_intent:
            fast = await _try_local_filesystem_fast_path(self._tool_registry, latest_user_input)
            if fast:
                await self._append_rolling_context(self._direct_rolling_context, "user", latest_user_input)
                await self._append_rolling_context(self._direct_rolling_context, "assistant", fast)
                self._schedule_direct_memory_store(latest_user_input, fast)
                return fast

        # Tool-grounded direct requests should rely on fresh tool outputs rather
        # than semantically retrieved memory, which can leak stale facts into
        # summarization/fetch tasks and increase prompt size.
        allow_memory_context = not (casual_chat or web_intent or local_file_intent or short_followup_edit_intent)
        if conversation_context and short_followup_edit_intent:
            allow_memory_context = False

        memory_context = ""
        if allow_memory_context and self._memory is not None:
            try:
                memory_results = await self._memory.search_advanced(
                    query=latest_user_input,
                    limit=4,
                    metadata_filter={"agent": "main", "mode": "direct", "kind": "turn"},
                    candidate_pool=40,
                )
                snippets: list[str] = []
                for row in memory_results:
                    text = _sanitize_memory_snippet(str(row.get("text") or ""))
                    if not text:
                        continue
                    snippets.append(f"- {text}")
                if snippets:
                    memory_context = "Relevant long-term memory:\n" + "\n".join(snippets)
            except Exception as exc:
                logger.warning("MainAgent direct memory search failed: %s", exc.__class__.__name__)

        rolling_context = ""
        if not casual_chat:
            rolling_context = self._rolling_context_prompt(
                self._direct_rolling_context,
                "Rolling conversation summary",
            )

        # ── Tool injection ───────────────────────────────────────────────────
        # Route tools semantically based on the user input so the model always
        # knows what it can call.  We avoid desktop/vision tools unless the
        # request explicitly asks for them.
        tools_section = ""
        active_tools: list = []
        if self._tool_registry is not None and tool_intent:
            active_tools = self._tool_registry.get_tools(
                context=latest_user_input,
                semantic_routing_enabled=True,
            )
            active_tools = _filter_agent_handoff_tools(active_tools, latest_user_input)
            if web_intent:
                active_tools = [t for t in active_tools if t.name not in _DESKTOP_VISION_TOOLS]
                for must_have in ("web_fetch", "web_search"):
                    forced = self._tool_registry.get(must_have)
                    if forced is not None and all(t.name != must_have for t in active_tools):
                        active_tools.append(forced)
            if local_file_intent:
                active_tools = [
                    t for t in active_tools
                    if t.name not in _DESKTOP_VISION_TOOLS and t.name in {"file", "shell", "patch", "diff", "process"}
                ]
                for must_have in ("file", "shell", "patch", "diff", "process"):
                    forced = self._tool_registry.get(must_have)
                    if forced is not None and all(t.name != must_have for t in active_tools):
                        active_tools.append(forced)
            if schedule_intent:
                forced = self._tool_registry.get("scheduler")
                if forced is not None and all(t.name != "scheduler" for t in active_tools):
                    active_tools.append(forced)
            if active_tools:
                tools_block = self._tool_registry.render_tool_descriptions(active_tools)
                example_tool_name = active_tools[0].name
                tools_section = (
                    f"\n\n{tools_block}\n\n"
                    "To use a tool, emit a <tool_call> block with JSON:\n"
                    "<tool_call>\n"
                    f'{{"tool": "{example_tool_name}", "args": {{}}}}\n'
                    "</tool_call>\n"
                    "Use an exact tool name from the <tools> block above.\n"
                    "The tool result will be provided and you may continue.\n"
                    "During tool calls: output ONLY the next <tool_call> block, no prose.\n"
                    "Never invent file paths, URLs, or facts not present in tool outputs.\n"
                    "When the user asks about the content, purpose, location, or details of a URL/domain/site, "
                    "use web_fetch unless recent conversation already contains a successful fetched result for that same target.\n"
                    "When you have real data from tools: write the answer using actual results, not guesses."
                )

        sys_prompt = _DIRECT_SYSTEM_PROMPT
        prompt_parts = [f"system: {sys_prompt}"]
        if casual_chat:
            prompt_parts.append(
                "Style constraint: this is a simple casual check-in. "
                "Reply in one short friendly sentence. Do not mention identity, architecture, tools, model, or capabilities."
            )
        if local_file_intent:
            prompt_parts.append(_local_filesystem_context())
        if memory_context:
            prompt_parts.append(memory_context)
        if rolling_context:
            prompt_parts.append(rolling_context)
        if conversation_context:
            trimmed_context = [line.strip() for line in conversation_context if line and line.strip()]
            if trimmed_context:
                prompt_parts.append("Recent conversation:\n" + "\n".join(trimmed_context[-8:]))
        if tools_section:
            prompt_parts.append(tools_section.strip())
        prompt_parts.append(f"user: {latest_user_input}\nassistant:")
        prompt = "\n\n".join(prompt_parts)
        direct_max_tokens = 96 if casual_chat else max(
            128,
            min(int(self._config.main_context_size or 4096), _DIRECT_MAX_TOKENS_CAP),
        )

        # ── Tool-call loop ───────────────────────────────────────────────────
        accumulated_tool_context = ""
        response = ""
        last_tool_answer = ""
        executed_tool_notes: list[str] = []
        any_tool_error = False
        for iteration in range(_MAX_TOOL_ITERATIONS):
            full_prompt = prompt + accumulated_tool_context
            try:
                response = await self._slot_manager.send_to_main(
                    full_prompt, max_tokens=direct_max_tokens
                )
            except RuntimeError as exc:
                # llama.cpp can return HTTP 500 "context shift is disabled" when
                # prompt state grows too large for the slot. Retry once with an
                # aggressively trimmed tool context tail.
                if "context shift is disabled" in str(exc).lower() and accumulated_tool_context:
                    accumulated_tool_context = _trim_tool_context(accumulated_tool_context[-4000:])
                    full_prompt = prompt + accumulated_tool_context
                    response = await self._slot_manager.send_to_main(
                        full_prompt, max_tokens=direct_max_tokens
                    )
                else:
                    raise

            # No registry or no tools routed → single-shot answer
            if self._tool_registry is None or not active_tools:
                break

            tool_calls = self._tool_registry.parse_tool_calls(response)
            if not tool_calls:
                # Model attempted tool protocol but emitted malformed payload.
                # Ask for a corrected tool_call (or final answer) instead of
                # returning raw protocol text to the user.
                if "<tool_call>" in (response or "").lower() and iteration < (_MAX_TOOL_ITERATIONS - 1):
                    accumulated_tool_context += (
                        "\n\nYour previous <tool_call> payload was malformed. "
                        "Emit ONE valid <tool_call> JSON block with exact keys {\"tool\",\"args\"}, "
                        "or provide the final answer if no tool is needed.\nassistant:"
                    )
                    accumulated_tool_context = _trim_tool_context(accumulated_tool_context)
                    continue
                break  # Final answer; no more tool calls needed

            response_in_cache = _ASSISTANT_THINK_SKIP + response
            tool_result_lines: list[str] = [response_in_cache]
            for tc in tool_calls:
                tool_name = _resolve_tool_name(
                    tc["tool"],
                    tc.get("args", {}),
                    active_tools=active_tools,
                    user_input=latest_user_input,
                )
                tool_args = _coerce_tool_args(tool_name, tc.get("args", {}), latest_user_input)
                if tool_name == "sub_agent" and isinstance(tool_args, dict):
                    tool_args["task"] = self._augment_sub_agent_task(
                        str(tool_args.get("task") or latest_user_input),
                        conversation_context,
                    )
                tool = self._tool_registry.get(tool_name)
                if tool is None:
                    result_dict = {"ok": False, "error": f"Unknown tool: {tool_name!r}"}
                else:
                    try:
                        tool_result = await tool.execute(**tool_args)
                        result_dict = tool_result.to_dict()
                    except Exception as exc:
                        logger.warning("Direct tool %s raised: %s", tool_name, exc)
                        result_dict = {"ok": False, "error": "Tool execution failed"}
                rendered = _render_tool_result_answer(latest_user_input, tool_name, result_dict)
                if rendered:
                    last_tool_answer = rendered
                if not bool(result_dict.get("ok")):
                    any_tool_error = True
                note_parts = [f"tool={tool_name}", f"ok={bool(result_dict.get('ok'))}"]
                if isinstance(tool_args, dict) and tool_args:
                    note_parts.append(
                        "args="
                        + _truncate_text(
                            json.dumps(_sanitize_tool_payload_for_prompt(tool_args), ensure_ascii=False, default=str),
                            500,
                        )
                    )
                if rendered:
                    note_parts.append("result_summary=" + _truncate_text(rendered, 1000))
                elif result_dict.get("error"):
                    note_parts.append("error=" + _truncate_text(str(result_dict.get("error")), 500))
                executed_tool_notes.append("; ".join(note_parts))
                result_json = json.dumps(
                    {
                        "tool": tool_name,
                        "ok": bool(result_dict.get("ok")),
                        "output": _sanitize_tool_payload_for_prompt(result_dict.get("output")),
                        "error": _sanitize_tool_payload_for_prompt(result_dict.get("error")),
                    },
                    ensure_ascii=False,
                    default=str,
                )
                tool_result_lines.append(f"\n<tool_result>\n{result_json}\n</tool_result>")

            accumulated_tool_context += (
                "\n" + "\n".join(tool_result_lines)
                + "\n\nBased on the tool result(s) above, provide the actual answer. "
                + "Extract and use what the tool returned. Do NOT make up information.\nassistant:"
            )
            _before_trim = accumulated_tool_context
            accumulated_tool_context = _trim_tool_context(accumulated_tool_context)
            if self._memory is not None and len(accumulated_tool_context) < len(_before_trim):
                dropped = _before_trim[: len(_before_trim) - len(accumulated_tool_context)]
                compact = _truncate_text(" ".join(dropped.split()), 1200)
                try:
                    asyncio.create_task(
                        self._memory.store(
                            text=f"direct_tool_context_summary: {compact}",
                            metadata={"agent": "main", "slot": self.slot_id, "mode": "direct", "kind": "tool_context_trim"},
                        )
                    )
                except Exception:
                    pass

        if (
            (not (response or "").strip() or _needs_direct_repair(response, casual_chat=casual_chat) or any_tool_error)
            and last_tool_answer
        ):
            response = last_tool_answer

        if _needs_direct_repair(response, casual_chat=casual_chat):
            repair_prompt = (
                "system: Rewrite raw model/tool output into a concise Discord-ready answer. "
                "Do not include HTML, XML tags, tool protocol, or internal architecture. "
                "If the user asked a simple greeting, answer naturally in one short sentence.\n\n"
                f"user: {latest_user_input}\n\n"
                f"raw output:\n{_truncate_text(response, 5000)}\n\n"
                "assistant:"
            )
            try:
                response = await self._slot_manager.send_to_main(
                    repair_prompt,
                    max_tokens=min(direct_max_tokens, 384),
                )
            except Exception as exc:
                logger.warning("MainAgent direct repair failed: %s", exc.__class__.__name__)

        response = _sanitize_direct_response(response)
        if _needs_direct_repair(response, casual_chat=casual_chat) and _RAW_HTML_RE.search(response or ""):
            fallback = _htmlish_to_plain_summary(response)
            if fallback:
                response = fallback
        response = _dedupe_repeated_response(response)
        tool_context_note = ""
        if executed_tool_notes:
            tool_context_note = "Tool usage this turn:\n" + "\n".join(f"- {note}" for note in executed_tool_notes[-8:])
            await self._append_rolling_context(self._direct_rolling_context, "system", tool_context_note)
        await self._append_rolling_context(self._direct_rolling_context, "user", latest_user_input)
        await self._append_rolling_context(self._direct_rolling_context, "assistant", response)

        self._schedule_direct_memory_store(latest_user_input, response, tool_context_note)

        return response

    def _describe_direct_capabilities(self) -> str:
        """Return a deterministic capabilities summary from registered tools."""
        if self._tool_registry is None:
            return (
                "I currently do not have a tool registry attached, so only plain text chat is available right now."
            )

        tools = self._tool_registry.get_tools()
        if not tools:
            return (
                "I currently have no external tools enabled. "
                "I can still provide direct text answers."
            )

        lines = [
            f"I currently have access to {len(tools)} tool(s):"
        ]
        for tool in sorted(tools, key=lambda t: t.name):
            lines.append(f"- {tool.name}: {tool.description}")

        lines.append("Use /task <goal> to run agentic mode where tools can be invoked during step execution.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agentic-loop helpers
    # ------------------------------------------------------------------

    def _diag_fast_plan(
        self,
        run_id: str | None,
        approach: str,
        steps: list[str],
        source: str,
    ) -> dict:
        """Build a fast-path plan dict and optionally record it diagnostically."""
        plan = {"approach": approach, "steps": steps}
        if self._diagnostic is not None and run_id is not None:
            self._diagnostic.record(
                run_id,
                "plan_parsed",
                approach=approach,
                steps=steps,
                source=source,
            )
        return plan

    async def plan_task(self, task: str, *, run_id: str | None = None) -> dict:
        """Break a task into an ordered list of actionable steps.

        Requests JSON output for reliable parsing.  Falls back to regex
        extraction and then single-step execution on any failure.

        Returns a dict with keys:
        - ``approach``: brief description of the overall strategy.
        - ``steps``: list of step description strings (max 10).
        """
        # Fast-path: URL fetch tasks should be handled with web_fetch only,
        # not desktop UI automation steps.
        task_text = task or ""
        url_match = _URL_RE.search(task_text)
        domain_match = _DOMAIN_RE.search(task_text)
        local_file_task = bool(_LOCAL_FILE_HINT_RE.search(task_text))
        if local_file_task:
            return self._diag_fast_plan(
                run_id,
                approach="Use local filesystem tools directly and report concrete paths or file changes.",
                steps=[
                    "Resolve the user's local filesystem request using real file or shell tools, then report exact paths, files found, or changes made."
                ],
                source="fast_path_local_file",
            )

        if _FETCH_INTENT_RE.search(task_text) and (url_match or domain_match) and not local_file_task:
            if url_match:
                url = url_match.group(0)
            else:
                url = f"https://{domain_match.group(0)}"
            return self._diag_fast_plan(
                run_id,
                approach="Use web_fetch directly and summarize the response.",
                steps=[f"Use web_fetch to fetch {url} and report key findings."],
                source="fast_path_web_fetch",
            )

        # Fetch intent without a concrete URL/domain should ask for target
        # instead of letting the model invent a placeholder URL.
        if _FETCH_INTENT_RE.search(task_text) and not (url_match or domain_match) and not local_file_task:
            return self._diag_fast_plan(
                run_id,
                approach="Ask user for a specific URL before running web_fetch.",
                steps=[
                    "Ask the user to provide a full URL or domain to fetch (for example: https://example.com)."
                ],
                source="fast_path_no_url",
            )

        # NOTE: The plan prompt MUST share the same system-prompt prefix as
        # execute_step so llama.cpp can reuse the KV cache between planning and
        # execution calls on the same slot.  Using a different system prompt
        # (e.g. "You are a planning AI") invalidates the cache and forces a
        # full re-process of the entire prompt on every call.
        plan_prompt = (
            f"system: {self.system_prompt}\n\n"
            "Break the following task into clear, actionable steps that you can "
            "execute one at a time.\n\n"
            "Respond with a single valid JSON object only:\n"
            '{"approach": "<brief strategy description>", '
            '"steps": ["<step 1>", "<step 2>", ...]}\n\n'
            "Rules:\n"
            "- Maximum 10 steps.\n"
            "- Each step should be a single, concrete action.\n"
            "- Steps must be ordered and build on each other.\n\n"
            f"Task: {task}\nassistant:"
        )

        if self._diagnostic is not None and run_id is not None:
            self._diagnostic.record_planning_request(run_id, prompt=plan_prompt)

        try:
            # Plan JSON rarely needs more than a few hundred tokens, but give it
            # room in case the task has many steps.
            response = await self._slot_manager.send_to_main(
                plan_prompt, max_tokens=1024
            )
        except Exception as exc:
            if self._diagnostic is not None and run_id is not None:
                self._diagnostic.record_planning_response(
                    run_id, response="", error=exc.__class__.__name__
                )
            logger.warning("plan_task failed: %s", exc.__class__.__name__)
            return {"steps": [task], "approach": "Direct execution"}

        if self._diagnostic is not None and run_id is not None:
            self._diagnostic.record_planning_response(run_id, response=response)

        result = _parse_plan(response)

        if self._diagnostic is not None and run_id is not None:
            self._diagnostic.record(
                run_id, "plan_parsed",
                approach=result.get("approach", ""),
                steps=result.get("steps", []),
                source="model",
            )

        return result

    async def execute_step(
        self,
        step: str,
        task: str,
        context: list[str] | None = None,
        on_event: Callable[[dict], Awaitable[dict | None] | dict | None] | None = None,
        *,
        include_rolling_context: bool = True,
        tool_calls_enabled: bool = True,
        semantic_routing_enabled: bool = True,
        allowed_tool_names: list[str] | None = None,
        run_id: str | None = None,
        step_num: int | None = None,
    ) -> str:
        """Execute a single step within a larger task.

        If a ToolRegistry is attached, the model may issue ``<tool_call>``
        blocks in its response.  Each block is parsed, the tool executed, and
        the result injected back into context before the model is called again.
        This loop repeats until no tool calls remain or the iteration limit is
        reached.

        Parameters
        ----------
        step:
            The current step description.
        task:
            The overarching task so the agent maintains focus.
        context:
            Accumulated results from previous steps (most recent last).
        """
        context_section = ""
        if include_rolling_context and context:
            recent = context[-5:]
            context_section = "\n\nContext from previous steps:\n" + "\n".join(recent)
        if include_rolling_context:
            rolling_context_section = self._rolling_context_prompt(
                self._agentic_rolling_context,
                "Rolling task summary",
            )
            if rolling_context_section:
                context_section += "\n\n" + rolling_context_section

        # Build tool descriptions block if tools are available
        tools_section = ""
        tools: list = []
        local_file_intent = bool(_LOCAL_FILE_HINT_RE.search(f"{task}\n{step}"))
        if local_file_intent:
            fast = await _try_local_filesystem_fast_path(self._tool_registry, f"{task}\n{step}")
            if fast:
                await self._append_rolling_context(self._agentic_rolling_context, "user", step)
                await self._append_rolling_context(self._agentic_rolling_context, "assistant", fast)
                return fast
        if self._tool_registry is not None and tool_calls_enabled:
            routing_context_parts = [
                f"task: {task}",
                f"step: {step}",
            ]
            if include_rolling_context and context:
                routing_context_parts.append("recent_context:")
                routing_context_parts.extend(context[-3:])
            routing_context = "\n".join(routing_context_parts)

            tools = self._tool_registry.get_tools(
                context=routing_context,
                semantic_routing_enabled=semantic_routing_enabled,
                allow_tool_names=allowed_tool_names,
            )
            tools = _filter_agent_handoff_tools(tools, routing_context)

            # Guardrail: web-fetch intent should not route through desktop/vision
            # tools, which can trigger expensive screenshots and unrelated UI actions.
            web_intent = bool(_FETCH_INTENT_RE.search(routing_context)) and bool(
                _URL_RE.search(routing_context) or _DOMAIN_RE.search(routing_context)
            )
            if web_intent:
                tools = [t for t in tools if t.name not in _DESKTOP_VISION_TOOLS]

            local_file_intent = bool(_LOCAL_FILE_HINT_RE.search(routing_context)) and not web_intent
            if local_file_intent:
                tools = [
                    t for t in tools
                    if t.name not in _DESKTOP_VISION_TOOLS and t.name in {"file", "shell", "patch", "diff", "process"}
                ]
                # Always expose core filesystem tools for local-file requests,
                # even if semantic routing did not include them in top-k.
                for must_have in ("file", "shell", "patch", "diff", "process"):
                    forced = self._tool_registry.get(must_have)
                    if forced is not None and all(t.name != must_have for t in tools):
                        tools.append(forced)

            if tools:
                tools_block = self._tool_registry.render_tool_descriptions(tools)
                example_tool_name = tools[0].name
                tools_section = (
                    f"\n\n{tools_block}\n\n"
                    "To use a tool, emit a <tool_call> block with JSON:\n"
                    "<tool_call>\n"
                    f'{{"tool": "{example_tool_name}", "args": {{}}}}\n'
                    "</tool_call>\n"
                    "Use an exact tool name from the <tools> block above.\n"
                    "The tool result will be provided and you may continue.\n"
                    "During tool calls: output ONLY the next <tool_call> block, no prose.\n"
                    "When you have enough real data: write the complete final answer with\n"
                    "actual findings, facts, and details — NOT a description of what you plan to write."
                )

        step_prompt = (
            f"system: {self.system_prompt}\n\n"
            "You are executing a multi-step task one step at a time.\n"
            f"Overall task: {task}\n"
            f"Current step: {step}"
            f"{context_section}"
            f"{chr(10) + chr(10) + _local_filesystem_context() if local_file_intent else ''}"
            f"{tools_section}\n\n"
            "Execute this step and return ACTUAL results now.\n"
            "Do NOT say what you plan to do or will compile later — output the real content directly.\n"
            "assistant:"
        )

        # Tool-calling loop
        accumulated_tool_context = ""
        last_tool_answer = ""
        _diag = self._diagnostic  # local alias for brevity
        _tool_names = [t.name for t in tools] if tools else []
        for iteration in range(_MAX_TOOL_ITERATIONS):
            prompt = step_prompt + accumulated_tool_context
            logger.info(
                "execute_step iteration=%d tool_calls_enabled=%s semantic_routing_enabled=%s allowed_tools=%s",
                iteration + 1,
                tool_calls_enabled,
                semantic_routing_enabled,
                allowed_tool_names or [],
            )
            # Give each iteration ample room. The model stops naturally when
            # done; we don't need to force-truncate the output.
            iter_max_tokens = self._config.main_context_size or 4096

            if _diag is not None and run_id is not None:
                _diag.record_model_request(
                    run_id,
                    step_num=step_num,
                    iteration=iteration + 1,
                    prompt=prompt,
                    tools_available=_tool_names,
                    slot_id=self.slot_id,
                )

            try:
                response = await self._slot_manager.send_to_main(prompt, max_tokens=iter_max_tokens)
            except Exception as exc:
                if _diag is not None and run_id is not None:
                    _diag.record_model_response(
                        run_id,
                        step_num=step_num,
                        iteration=iteration + 1,
                        response="",
                        error=exc.__class__.__name__,
                        slot_id=self.slot_id,
                    )
                if on_event is not None:
                    maybe = on_event(
                        {
                            "type": "model_error",
                            "error": exc.__class__.__name__,
                            "message": str(exc),
                        }
                    )
                    intervention = await maybe if inspect.isawaitable(maybe) else maybe
                    if intervention:
                        return intervention.get("message", "Execution paused by SafetySupervisor.")
                raise

            if _diag is not None and run_id is not None:
                _diag.record_model_response(
                    run_id,
                    step_num=step_num,
                    iteration=iteration + 1,
                    response=response,
                    slot_id=self.slot_id,
                )

            if on_event is not None:
                maybe = on_event({"type": "model_output", "output": response})
                intervention = await maybe if inspect.isawaitable(maybe) else maybe
                if intervention:
                    return intervention.get("message", "Execution paused by SafetySupervisor.")

            logger.info("execute_step model_output iteration=%d preview=%s", iteration + 1, _truncate_text(response, 160))

            # No tool registry or no tools → return directly
            if self._tool_registry is None or not tools:
                await self._append_rolling_context(self._agentic_rolling_context, "user", step)
                await self._append_rolling_context(self._agentic_rolling_context, "assistant", response)
                return response

            tool_calls = self._tool_registry.parse_tool_calls(response)
            logger.info("execute_step parsed_tool_calls iteration=%d count=%d", iteration + 1, len(tool_calls))
            if not tool_calls:
                # No tool calls — final answer
                await self._append_rolling_context(self._agentic_rolling_context, "user", step)
                await self._append_rolling_context(self._agentic_rolling_context, "assistant", response)
                return response

            # Execute each tool call and accumulate results.
            #
            # IMPORTANT – cache coherence for recurrent/SWA models:
            # The LlamaClient injects "\n<think>\n\n</think>\n\n" into every
            # prompt that ends with "assistant:".  The llama.cpp server caches
            # the FULL token sequence including that injected prefix + the model
            # response.  If we do NOT echo that prefix here, the next iteration's
            # prompt diverges from the cache immediately after "assistant:",
            # causing a forced full-context re-evaluation on every tool call.
            # By reconstructing the context as:
            #   \n<think>\n\n</think>\n\n<response>\n<tool_result>...\nassistant:
            # we match what llama.cpp cached, so only the new tokens (tool result
            # and the fresh assistant turn) need to be evaluated.
            # Echo the full response so the accumulated context matches the
            # exact token sequence llama.cpp cached.  The overall
            # _MAX_TOOL_CONTEXT_CHARS guard trims from the left if needed.
            response_in_cache = _ASSISTANT_THINK_SKIP + response
            tool_result_lines: list[str] = [response_in_cache]
            for tc in tool_calls:
                tool_name = _resolve_tool_name(
                    tc["tool"],
                    tc.get("args", {}),
                    active_tools=tools,
                    user_input=f"{task}\n{step}",
                )
                tool_args = _coerce_tool_args(tool_name, tc.get("args", {}), f"{task}\n{step}")
                if tool_name == "sub_agent" and isinstance(tool_args, dict):
                    handoff_lines = list(context or [])
                    handoff_lines.append(f"Overall task: {task}")
                    handoff_lines.append(f"Current step: {step}")
                    tool_args["task"] = self._augment_sub_agent_task(
                        str(tool_args.get("task") or step),
                        handoff_lines,
                    )

                if _diag is not None and run_id is not None:
                    _diag.record_tool_call_parsed(
                        run_id,
                        step_num=step_num,
                        iteration=iteration + 1,
                        tool_name=tool_name,
                        args=tool_args if isinstance(tool_args, dict) else {},
                    )

                if on_event is not None:
                    safe_args = _sanitize_tool_payload(tool_args)
                    maybe = on_event(
                        {
                            "type": "tool_call",
                            "tool": tool_name,
                            "args": safe_args,
                        }
                    )
                    intervention = await maybe if inspect.isawaitable(maybe) else maybe
                    if intervention:
                        return intervention.get("message", "Execution paused by SafetySupervisor.")

                tool = self._tool_registry.get(tool_name)
                if tool is None:
                    result_dict = {"ok": False, "error": f"Unknown tool: {tool_name!r}"}
                else:
                    try:
                        tool_result = await tool.execute(**tool_args)
                        result_dict = tool_result.to_dict()
                    except Exception as exc:
                        logger.warning("Tool %s raised: %s", tool_name, exc)
                        # Avoid leaking internal paths or credentials to the model.
                        result_dict = {"ok": False, "error": "Tool execution failed"}

                if _diag is not None and run_id is not None:
                    _diag.record_tool_executed(
                        run_id,
                        step_num=step_num,
                        tool_name=tool_name,
                        ok=bool(result_dict.get("ok")),
                        output=result_dict.get("output"),
                        error=result_dict.get("error"),
                    )

                rendered = _render_tool_result_answer(f"{task}\n{step}", tool_name, result_dict)
                if rendered:
                    last_tool_answer = rendered

                if on_event is not None:
                    safe_output = _sanitize_tool_payload(result_dict.get("output"))
                    safe_error = _sanitize_tool_payload(result_dict.get("error"))
                    maybe = on_event(
                        {
                            "type": "tool_result",
                            "tool": tool_name,
                            "args": safe_args,
                            "ok": bool(result_dict.get("ok")),
                            "output": safe_output,
                            "error": safe_error,
                        }
                    )
                    intervention = await maybe if inspect.isawaitable(maybe) else maybe
                    if intervention:
                        return intervention.get("message", "Execution paused by SafetySupervisor.")

                result_json = json.dumps(
                    {
                        "tool": tool_name,
                        "ok": bool(result_dict.get("ok")),
                        "output": _sanitize_tool_payload_for_prompt(result_dict.get("output")),
                        "error": _sanitize_tool_payload_for_prompt(result_dict.get("error")),
                    },
                    ensure_ascii=False,
                    default=str,
                )
                tool_result_lines.append(
                    f"\n<tool_result>\n{result_json}\n</tool_result>"
                )

            # After tool results, add an explicit synthesis directive to force the model
            # to extract real findings instead of hallucinating. This prevents responses
            # that ignore the actual tool output and make things up.
            accumulated_tool_context += (
                "\n" + "\n".join(tool_result_lines)
                + "\n\nBased on the tool result(s) above, provide the actual findings, data, or answer. "
                + "Do NOT make up information. Extract and cite what the tool returned. "
                + "If the tool returned an error, explain that. If it succeeded, summarize the real result.\nassistant:"
            )
            _before_trim = accumulated_tool_context
            accumulated_tool_context = _trim_tool_context(accumulated_tool_context)
            if self._memory is not None and len(accumulated_tool_context) < len(_before_trim):
                dropped = _before_trim[: len(_before_trim) - len(accumulated_tool_context)]
                compact = _truncate_text(" ".join(dropped.split()), 1200)
                try:
                    asyncio.create_task(
                        self._memory.store(
                            text=f"agentic_tool_context_summary: {compact}",
                            metadata={"agent": "main", "slot": self.slot_id, "mode": "agentic", "kind": "tool_context_trim"},
                        )
                    )
                except Exception:
                    pass

        # Iteration limit reached — return last response
        await self._append_rolling_context(self._agentic_rolling_context, "user", step)
        await self._append_rolling_context(self._agentic_rolling_context, "assistant", response)
        return response
