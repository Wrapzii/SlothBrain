"""SlothBrain TUI — Textual-based desktop interface.

Run with:
    python -m tui.app
  or
    textual run tui/app.py
"""

from __future__ import annotations

import asyncio
import re
from typing import ClassVar, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

import tui.api as api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(val: object) -> str:
    return str(val) if val is not None else "—"


_TOOL_INTENT_RE = re.compile(
    r"(\bweb_fetch\b|\buse\b.{0,20}\btool\b|\btry\b.{0,20}\btool\b|\bfetch\b\s+https?://)",
    re.IGNORECASE,
)


def _should_use_agentic_chat(message: str) -> bool:
    msg = (message or "").strip().lower()
    return msg.startswith("/task ") or bool(_TOOL_INTENT_RE.search(message or ""))


# ---------------------------------------------------------------------------
# Confirm modal
# ---------------------------------------------------------------------------


class ConfirmScreen(ModalScreen[bool]):
    """A simple yes/no confirmation dialog."""

    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label(self._message, id="confirm-msg")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("No", variant="primary", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    DEFAULT_CSS = """
    ConfirmScreen {
        align: center middle;
    }
    #confirm-dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $error;
        padding: 1 2;
    }
    #confirm-msg { margin-bottom: 1; }
    #confirm-buttons { align: center middle; }
    #confirm-buttons Button { margin: 0 1; }
    """


# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------


class DashboardTab(Container):
    cpu: reactive[float] = reactive(0.0)
    ram_mb: reactive[float] = reactive(0.0)
    ram_pct: reactive[float] = reactive(0.0)
    mode: reactive[str] = reactive("unknown")

    def compose(self) -> ComposeResult:
        yield Label("[b]System Status[/b]", id="dash-title")
        with Horizontal(id="dash-stats"):
            with Vertical(classes="stat-card"):
                yield Label("CPU", classes="stat-label")
                yield Label("0 %", id="cpu-val")
                yield ProgressBar(total=100, id="cpu-bar", show_eta=False)
            with Vertical(classes="stat-card"):
                yield Label("RAM", classes="stat-label")
                yield Label("0 MB (0 %)", id="ram-val")
                yield ProgressBar(total=100, id="ram-bar", show_eta=False)
            with Vertical(classes="stat-card"):
                yield Label("Mode", classes="stat-label")
                yield Label("unknown", id="mode-val")
                with Horizontal():
                    yield Button("Idle", id="set-idle", variant="default")
                    yield Button("Active", id="set-active", variant="success")
        yield Label("[b]Slots[/b]", id="slots-title")
        yield DataTable(id="slots-table", show_cursor=False)
        yield Label("[b]Audit Log[/b]", id="audit-title")
        with Horizontal(id="audit-controls"):
            yield Button("Refresh", id="audit-refresh", variant="default")
            yield Select(
                [("25", "25"), ("50", "50"), ("100", "100"), ("200", "200")],
                value="50",
                id="audit-limit",
                allow_blank=False,
            )
        yield DataTable(id="audit-table", show_cursor=False)

    def on_mount(self) -> None:
        st = self.query_one("#slots-table", DataTable)
        st.add_columns("ID", "Role", "State")
        at = self.query_one("#audit-table", DataTable)
        at.add_columns("Time", "Action", "Actor", "Details")
        self.app.call_later(self._load_audit)

    def update_stats(self, data: dict) -> None:
        cpu = data.get("cpu_percent", 0.0)
        self.query_one("#cpu-val", Label).update(f"{cpu:.1f} %")
        bar = self.query_one("#cpu-bar", ProgressBar)
        bar.progress = cpu
        self.cpu = cpu  # keep reactive in sync for potential watchers

        # Status may be flat (ram_used_mb) or nested (ram.used_mb)
        ram_nested = data.get("ram", {})
        mb = ram_nested.get("used_mb") if ram_nested else None
        pct = ram_nested.get("percent") if ram_nested else None
        if mb is None:
            mb = data.get("ram_used_mb", 0)
        if pct is None:
            total = data.get("ram_total_mb", 1) or 1
            pct = (mb / total * 100) if total else 0
        self.query_one("#ram-val", Label).update(f"{mb:.0f} MB ({pct:.1f} %)")
        rbar = self.query_one("#ram-bar", ProgressBar)
        rbar.progress = pct
        self.ram_pct = pct  # keep reactive in sync

    def update_slots(self, slots: list) -> None:
        t = self.query_one("#slots-table", DataTable)
        t.clear()
        for s in slots:
            t.add_row(_fmt(s.get("id")), _fmt(s.get("role")), _fmt(s.get("state")))

    def update_mode(self, mode: str) -> None:
        self.query_one("#mode-val", Label).update(mode)

    async def _load_audit(self) -> None:
        limit_sel = self.query_one("#audit-limit", Select)
        n = int(limit_sel.value)
        try:
            entries = await api.get_audit_log(n)
        except Exception as exc:
            self.app.notify(f"Audit log error: {exc}", severity="error")
            return
        t = self.query_one("#audit-table", DataTable)
        t.clear()
        for e in entries:
            t.add_row(
                _fmt(e.get("timestamp", "")[:19]),
                _fmt(e.get("action")),
                _fmt(e.get("actor")),
                _fmt(e.get("details") or e.get("changes")),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "audit-refresh":
            self.app.call_later(self._load_audit)
        elif event.button.id == "set-idle":
            self.app.call_later(self._set_mode, "idle")
        elif event.button.id == "set-active":
            self.app.call_later(self._set_mode, "active")

    async def _set_mode(self, mode: str) -> None:
        try:
            await api.set_mode(mode)
            self.update_mode(mode)
        except Exception as exc:
            self.app.notify(f"Mode error: {exc}", severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "audit-limit":
            self.app.call_later(self._load_audit)

    DEFAULT_CSS = """
    DashboardTab { padding: 1; }
    #dash-stats { height: auto; }
    .stat-card { width: 1fr; border: round $primary; padding: 1; margin: 0 1; }
    .stat-label { text-style: bold; }
    #slots-title, #audit-title { margin-top: 1; text-style: bold; }
    #audit-controls { height: auto; }
    #audit-limit { width: 10; margin-left: 1; }
    """


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------


class ChatTab(Container):
    def compose(self) -> ComposeResult:
        yield Label("[b]Chat[/b]")
        yield Label("Direct by default. Use /task <goal> for full agentic execution.", id="agentic-label")
        yield Log(id="chat-log", highlight=True)
        with Horizontal(id="chat-input-row"):
            yield Input(placeholder="Message (or /task <goal>)", id="chat-input")
            yield Button("Run", variant="primary", id="chat-send")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-send":
            self._do_send()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._do_send()

    def _do_send(self) -> None:
        inp = self.query_one("#chat-input", Input)
        msg = inp.value.strip()
        if not msg:
            return
        inp.value = ""
        inp.disabled = True
        send_btn = self.query_one("#chat-send", Button)
        send_btn.disabled = True
        log = self.query_one("#chat-log", Log)
        use_agentic = _should_use_agentic_chat(msg)
        mode_label = "agentic" if use_agentic else "direct"
        log.write_line(f"[You → {mode_label}] {msg}")
        log.write_line("[thinking…]")
        self.run_worker(self._send_async(msg, log), exclusive=False, name="chat-request")

    async def _send_async(self, msg: str, log: Log) -> None:
        use_agentic = _should_use_agentic_chat(msg)
        if not use_agentic:
            try:
                result = await api.send_chat(msg, max_steps=1, mode="direct")
                response = str(result.get("response", "")).strip() or "(no response returned)"
                log.write_line(f"[direct] {response}")
            except Exception as exc:
                log.write_line(f"[error] {exc}")
            finally:
                inp = self.query_one("#chat-input", Input)
                inp.disabled = False
                send_btn = self.query_one("#chat-send", Button)
                send_btn.disabled = False
                inp.focus()
            return

        task_msg = msg[6:].strip() if msg.lower().startswith("/task ") else msg
        task_msg = task_msg or msg
        try:
            result: dict | None = None
            async for event in api.stream_agentic_chat(task_msg):
                et = event.get("type")
                if et == "planning":
                    log.write_line("[agentic] planning steps...")
                elif et == "plan_ready":
                    total = event.get("total_steps", 0)
                    log.write_line(f"[agentic] plan ready: {total} step(s)")
                elif et == "step_start":
                    sn = event.get("step_num")
                    total = event.get("total_steps")
                    desc = str(event.get("description", "")).strip()
                    log.write_line(f"[step {sn}/{total}] {desc}")
                elif et == "tool_call":
                    tool = event.get("tool", "unknown")
                    args = event.get("args", {})
                    log.write_line(f"[tool → {tool}] args={args}")
                elif et == "tool_result":
                    tool = event.get("tool", "unknown")
                    ok = bool(event.get("ok"))
                    if ok:
                        out = str(event.get("output", ""))[:220]
                        log.write_line(f"[tool ✓ {tool}] {out}")
                    else:
                        err = str(event.get("error", "tool failed"))
                        log.write_line(f"[tool ✗ {tool}] {err}")
                elif et == "model_error":
                    err = str(event.get("error", "ModelError"))
                    msg_txt = str(event.get("message", ""))
                    log.write_line(f"[model error] {err}: {msg_txt}")
                elif et == "step_retry":
                    sn = event.get("step_num")
                    attempt = event.get("attempt")
                    fb = str(event.get("feedback", ""))
                    log.write_line(f"[step {sn}] retry #{attempt}: {fb}")
                elif et == "step_monitored":
                    sn = event.get("step_num")
                    action = event.get("action", "continue")
                    fb = str(event.get("feedback", ""))
                    if fb:
                        log.write_line(f"[watcher step {sn}] {action}: {fb}")
                elif et == "step_complete":
                    sn = event.get("step_num")
                    status = event.get("status", "complete")
                    log.write_line(f"[step {sn}] {status}")
                elif et == "verifying":
                    log.write_line("[agentic] verifying completion...")
                elif et == "complete":
                    verified = bool(event.get("verified", False))
                    summary = str(event.get("summary", ""))
                    log.write_line(f"[agentic] complete; verified={verified} | {summary}")
                elif et == "result":
                    result = event
                elif et == "error":
                    raise RuntimeError(str(event.get("message", "Unknown websocket error")))

            if result is None:
                result = await api.send_agentic_chat(task_msg)

            summary = result.get("summary") or "(no summary returned)"
            completed = result.get("completed", False)
            verified = result.get("completion_verified", False)
            status_bits = [
                "completed" if completed else "incomplete",
                "verified" if verified else "unverified",
            ]
            log.write_line(f"[agentic:{', '.join(status_bits)}] {summary}")
        except Exception as exc:
            log.write_line(f"[error] {exc}")
            try:
                result = await api.send_agentic_chat(task_msg)
                summary = result.get("summary") or "(no summary returned)"
                completed = result.get("completed", False)
                verified = result.get("completion_verified", False)
                status_bits = [
                    "completed" if completed else "incomplete",
                    "verified" if verified else "unverified",
                ]
                log.write_line(f"[agentic:{', '.join(status_bits)}] {summary}")
            except Exception as fallback_exc:
                log.write_line(f"[error] fallback request failed: {fallback_exc}")
        finally:
            inp = self.query_one("#chat-input", Input)
            inp.disabled = False
            send_btn = self.query_one("#chat-send", Button)
            send_btn.disabled = False
            inp.focus()

    DEFAULT_CSS = """
    ChatTab { padding: 1; }
    #agentic-label { color: $accent; margin-top: 1; }
    #chat-log { height: 1fr; border: round $primary; margin-top: 1; }
    #chat-input-row { height: auto; margin-top: 1; }
    #chat-input { width: 1fr; }
    #chat-send { width: 10; }
    """


# ---------------------------------------------------------------------------
# Agents tab
# ---------------------------------------------------------------------------


class AgentsTab(Container):
    def compose(self) -> ComposeResult:
        yield Label("[b]Agent Presets[/b]")
        yield DataTable(id="presets-table")
        with Horizontal(id="preset-actions"):
            yield Button("Refresh", id="presets-refresh", variant="default")
            yield Button("New Preset", id="preset-new", variant="primary")
            yield Button("Spawn Selected", id="preset-spawn", variant="success")
            yield Button("Delete Selected", id="preset-delete", variant="error")
        yield Label("[b]Running Agents[/b]", id="agents-title")
        yield DataTable(id="agents-table")
        with Horizontal(id="agent-actions"):
            yield Button("Refresh", id="agents-refresh", variant="default")
            yield Button("Kill Selected", id="agent-kill", variant="error")

    def on_mount(self) -> None:
        pt = self.query_one("#presets-table", DataTable)
        pt.add_columns("ID", "Name", "Description", "Temperature", "Max Tokens")
        at = self.query_one("#agents-table", DataTable)
        at.add_columns("ID", "Name", "Preset")
        self.app.call_later(self._load_presets)
        self.app.call_later(self._load_agents)

    async def _load_presets(self) -> None:
        try:
            presets = await api.list_presets()
        except Exception as exc:
            self.app.notify(f"Presets error: {exc}", severity="error")
            return
        t = self.query_one("#presets-table", DataTable)
        t.clear()
        for p in presets:
            t.add_row(
                _fmt(p.get("id")),
                _fmt(p.get("name")),
                _fmt(p.get("description")),
                _fmt(p.get("temperature")),
                _fmt(p.get("max_tokens")),
                key=_fmt(p.get("id")),
            )

    async def _load_agents(self) -> None:
        try:
            agents = await api.list_agents()
        except Exception as exc:
            self.app.notify(f"Agents error: {exc}", severity="error")
            return
        t = self.query_one("#agents-table", DataTable)
        t.clear()
        for a in agents:
            t.add_row(
                _fmt(a.get("id")),
                _fmt(a.get("name")),
                _fmt(a.get("preset_id") or a.get("preset")),
                key=_fmt(a.get("id")),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "presets-refresh":
            self.app.call_later(self._load_presets)
        elif bid == "agents-refresh":
            self.app.call_later(self._load_agents)
        elif bid == "preset-spawn":
            self.app.call_later(self._spawn_selected)
        elif bid == "preset-delete":
            self.app.call_later(self._delete_selected_preset)
        elif bid == "agent-kill":
            self.app.call_later(self._kill_selected_agent)
        elif bid == "preset-new":
            self.app.push_screen(NewPresetScreen(), self._on_preset_created)

    async def _on_preset_created(self, result: dict | None) -> None:
        if result:
            try:
                await api.create_preset(result)
                await self._load_presets()
                self.app.notify("Preset created.")
            except Exception as exc:
                self.app.notify(f"Create error: {exc}", severity="error")

    async def _spawn_selected(self) -> None:
        t = self.query_one("#presets-table", DataTable)
        if t.cursor_row < 0:
            self.app.notify("Select a preset first.", severity="warning")
            return
        row = t.get_row_at(t.cursor_row)
        preset_id = row[0]
        try:
            await api.spawn_agent(preset_id)
            await self._load_agents()
            self.app.notify(f"Agent spawned from preset {preset_id}.")
        except Exception as exc:
            self.app.notify(f"Spawn error: {exc}", severity="error")

    async def _delete_selected_preset(self) -> None:
        t = self.query_one("#presets-table", DataTable)
        if t.cursor_row < 0:
            self.app.notify("Select a preset first.", severity="warning")
            return
        row = t.get_row_at(t.cursor_row)
        preset_id = row[0]
        confirmed = await self.app.push_screen_wait(ConfirmScreen(f"Delete preset '{row[1]}'?"))
        if confirmed:
            try:
                await api.delete_preset(preset_id)
                await self._load_presets()
                self.app.notify("Preset deleted.")
            except Exception as exc:
                self.app.notify(f"Delete error: {exc}", severity="error")

    async def _kill_selected_agent(self) -> None:
        t = self.query_one("#agents-table", DataTable)
        if t.cursor_row < 0:
            self.app.notify("Select an agent first.", severity="warning")
            return
        row = t.get_row_at(t.cursor_row)
        agent_id = row[0]
        confirmed = await self.app.push_screen_wait(ConfirmScreen(f"Kill agent '{row[1]}'?"))
        if confirmed:
            try:
                await api.destroy_agent(agent_id)
                await self._load_agents()
                self.app.notify("Agent killed.")
            except Exception as exc:
                self.app.notify(f"Kill error: {exc}", severity="error")

    DEFAULT_CSS = """
    AgentsTab { padding: 1; }
    #presets-table { height: 12; }
    #preset-actions { height: auto; }
    #preset-actions Button { margin-right: 1; }
    #agents-title { margin-top: 1; text-style: bold; }
    #agents-table { height: 8; }
    #agent-actions { height: auto; }
    #agent-actions Button { margin-right: 1; }
    """


# ---------------------------------------------------------------------------
# New Preset modal
# ---------------------------------------------------------------------------


class NewPresetScreen(ModalScreen):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def compose(self) -> ComposeResult:
        with Container(id="np-dialog"):
            yield Label("[b]New Preset[/b]")
            yield Input(placeholder="Name", id="np-name")
            yield Input(placeholder="Description", id="np-desc")
            yield TextArea(id="np-prompt", language=None)
            yield Label("^ System Prompt")
            yield Input(placeholder="Temperature (e.g. 0.7)", id="np-temp")
            yield Input(placeholder="Max tokens (e.g. 512)", id="np-maxtok")
            with Horizontal(id="np-buttons"):
                yield Button("Create", variant="primary", id="np-create")
                yield Button("Cancel", variant="default", id="np-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "np-cancel":
            self.dismiss(None)
        elif event.button.id == "np-create":
            data = {
                "name": self.query_one("#np-name", Input).value,
                "description": self.query_one("#np-desc", Input).value,
                "system_prompt": self.query_one("#np-prompt", TextArea).text,
                "temperature": float(self.query_one("#np-temp", Input).value or "0.7"),
                "max_tokens": int(self.query_one("#np-maxtok", Input).value or "512"),
            }
            self.dismiss(data)

    DEFAULT_CSS = """
    NewPresetScreen { align: center middle; }
    #np-dialog {
        width: 70;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    #np-dialog Input, #np-dialog TextArea { margin-bottom: 1; }
    #np-prompt { height: 6; }
    #np-buttons { align: center middle; }
    #np-buttons Button { margin: 0 1; }
    """


# ---------------------------------------------------------------------------
# Settings tab
# ---------------------------------------------------------------------------


class SettingsTab(Container):
    def compose(self) -> ComposeResult:
        yield Label("[b]Settings[/b]")
        yield Label("Server Host")
        yield Input(id="s-host", placeholder="127.0.0.1")
        yield Label("Server Port")
        yield Input(id="s-port", placeholder="8080")
        yield Label("Main Slot")
        yield Input(id="s-main-slot", placeholder="1")
        yield Label("Main Context Size")
        yield Input(id="s-ctx", placeholder="32768")
        with Horizontal(id="settings-buttons"):
            yield Button("Load", id="s-load", variant="default")
            yield Button("Save", id="s-save", variant="primary")
            yield Button("Restart Server", id="s-restart", variant="warning")
        yield Label("", id="s-status")

    def on_mount(self) -> None:
        self.app.call_later(self._load)

    async def _load(self) -> None:
        try:
            cfg = await api.get_settings()
        except Exception as exc:
            self.query_one("#s-status", Label).update(f"Error: {exc}")
            return
        self.query_one("#s-host", Input).value = _fmt(cfg.get("llama_host", "127.0.0.1"))
        self.query_one("#s-port", Input).value = _fmt(cfg.get("llama_port", "8080"))
        self.query_one("#s-main-slot", Input).value = _fmt(cfg.get("main_slot", "1"))
        self.query_one("#s-ctx", Input).value = _fmt(cfg.get("main_context_size", "32768"))
        self.query_one("#s-status", Label).update("Settings loaded.")

    async def _save(self) -> None:
        data: dict = {
            "llama_host": self.query_one("#s-host", Input).value,
            "llama_port": int(self.query_one("#s-port", Input).value or "8080"),
            "main_slot": int(self.query_one("#s-main-slot", Input).value or "1"),
            "main_context_size": int(self.query_one("#s-ctx", Input).value or "32768"),
        }
        try:
            await api.update_settings(data)
            self.query_one("#s-status", Label).update("Saved.")
        except Exception as exc:
            self.query_one("#s-status", Label).update(f"Error: {exc}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "s-load":
            self.app.call_later(self._load)
        elif event.button.id == "s-save":
            self.app.call_later(self._save)
        elif event.button.id == "s-restart":
            self.app.call_later(self._restart)

    async def _restart(self) -> None:
        confirmed = await self.app.push_screen_wait(ConfirmScreen("Restart llama-server?"))
        if confirmed:
            try:
                await api.restart_server()
                self.query_one("#s-status", Label).update("Server restarting…")
            except Exception as exc:
                self.query_one("#s-status", Label).update(f"Error: {exc}")

    DEFAULT_CSS = """
    SettingsTab { padding: 1; }
    SettingsTab Label { margin-top: 1; }
    SettingsTab Input { margin-bottom: 0; }
    #settings-buttons { height: auto; margin-top: 1; }
    #settings-buttons Button { margin-right: 1; }
    #s-status { color: $success; }
    """


# ---------------------------------------------------------------------------
# Benchmarks tab
# ---------------------------------------------------------------------------


class BenchmarksTab(Container):
    def compose(self) -> ComposeResult:
        yield Label("[b]Benchmarks[/b]")
        with Horizontal(id="bench-buttons"):
            yield Button("Inference Speed", id="bench-speed", variant="primary")
            yield Button("VRAM", id="bench-vram", variant="primary")
            yield Button("Slot Interference", id="bench-slots", variant="primary")
            yield Button("Run All", id="bench-all", variant="warning")
        yield Log(id="bench-log", highlight=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        kind_map = {
            "bench-speed": "speed",
            "bench-vram": "vram",
            "bench-slots": "slots",
            "bench-all": "all",
        }
        kind = kind_map.get(event.button.id)
        if kind:
            self.app.call_later(self._run, kind)

    async def _run(self, kind: str) -> None:
        log = self.query_one("#bench-log", Log)
        log.write_line(f"Running {kind} benchmark…")
        try:
            result = await api.run_benchmark(kind)
            import json
            log.write_line(json.dumps(result, indent=2))
        except Exception as exc:
            log.write_line(f"Error: {exc}")

    DEFAULT_CSS = """
    BenchmarksTab { padding: 1; }
    #bench-buttons { height: auto; margin-bottom: 1; }
    #bench-buttons Button { margin-right: 1; }
    #bench-log { height: 1fr; border: round $primary; }
    """


# ---------------------------------------------------------------------------
# Approvals tab
# ---------------------------------------------------------------------------


class ApprovalsTab(Container):
    def compose(self) -> ComposeResult:
        yield Label("[b]Approval Queue[/b]")
        with Horizontal(id="appr-actions"):
            yield Button("Refresh", id="appr-refresh", variant="default")
            yield Button("Approve Selected", id="appr-approve", variant="success")
            yield Button("Reject Selected", id="appr-reject", variant="error")
        yield DataTable(id="appr-table")

    def on_mount(self) -> None:
        t = self.query_one("#appr-table", DataTable)
        t.add_columns("ID", "Action", "Description", "Timestamp")
        self.app.call_later(self._load)

    async def _load(self) -> None:
        try:
            items = await api.list_approvals()
        except Exception as exc:
            self.app.notify(f"Approvals error: {exc}", severity="error")
            return
        t = self.query_one("#appr-table", DataTable)
        t.clear()
        for item in items:
            t.add_row(
                _fmt(item.get("id")),
                _fmt(item.get("action")),
                _fmt(item.get("description")),
                _fmt(item.get("timestamp", "")[:19]),
                key=_fmt(item.get("id")),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "appr-refresh":
            self.app.call_later(self._load)
        elif event.button.id == "appr-approve":
            self.app.call_later(self._approve_selected)
        elif event.button.id == "appr-reject":
            self.app.call_later(self._reject_selected)

    async def _approve_selected(self) -> None:
        t = self.query_one("#appr-table", DataTable)
        if t.cursor_row < 0:
            self.app.notify("Select an item first.", severity="warning")
            return
        row = t.get_row_at(t.cursor_row)
        try:
            await api.approve_action(row[0])
            await self._load()
            self.app.notify("Approved.")
        except Exception as exc:
            self.app.notify(f"Error: {exc}", severity="error")

    async def _reject_selected(self) -> None:
        t = self.query_one("#appr-table", DataTable)
        if t.cursor_row < 0:
            self.app.notify("Select an item first.", severity="warning")
            return
        row = t.get_row_at(t.cursor_row)
        try:
            await api.reject_action(row[0])
            await self._load()
            self.app.notify("Rejected.")
        except Exception as exc:
            self.app.notify(f"Error: {exc}", severity="error")

    DEFAULT_CSS = """
    ApprovalsTab { padding: 1; }
    #appr-actions { height: auto; margin-bottom: 1; }
    #appr-actions Button { margin-right: 1; }
    #appr-table { height: 1fr; }
    """


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class SlothBrainApp(App):
    TITLE = "SlothBrain"
    CSS_PATH = None  # inline CSS only

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+e", "emergency_stop", "Emergency Stop"),
    ]

    CSS = """
    Screen { background: $background; }
    TabbedContent { height: 1fr; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="tabs"):
            with TabPane("Dashboard", id="tab-dashboard"):
                yield DashboardTab(id="dashboard")
            with TabPane("Chat", id="tab-chat"):
                yield ChatTab(id="chat")
            with TabPane("Agents", id="tab-agents"):
                yield AgentsTab(id="agents")
            with TabPane("Settings", id="tab-settings"):
                with ScrollableContainer():
                    yield SettingsTab(id="settings")
            with TabPane("Benchmarks", id="tab-benchmarks"):
                yield BenchmarksTab(id="benchmarks")
            with TabPane("Approvals", id="tab-approvals"):
                yield ApprovalsTab(id="approvals")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._status_loop(), exclusive=True, name="status-loop")

    async def _status_loop(self) -> None:
        async for data in api.status_stream():
            try:
                dash = self.query_one("#dashboard", DashboardTab)
                dash.update_stats(data)
                slots_raw = data.get("slots") or {}
                if isinstance(slots_raw, dict):
                    slots = slots_raw.get("slots") or []
                else:
                    slots = slots_raw
                if slots:
                    dash.update_slots(slots)
                mode = data.get("mode")
                if mode:
                    dash.update_mode(mode)
            except Exception:
                pass  # widget may not be mounted yet

    async def action_emergency_stop(self) -> None:
        confirmed = await self.push_screen_wait(ConfirmScreen("EMERGENCY STOP — kill server and all agents?"))
        if confirmed:
            try:
                await api.emergency_stop()
                self.notify("Emergency stop executed.", severity="error")
            except Exception as exc:
                self.notify(f"Error: {exc}", severity="error")


def main() -> None:
    SlothBrainApp().run()


if __name__ == "__main__":
    main()
