"""Dynamic plugin loader for user-supplied tools.

Place a ``.py`` file in ``backend/tools/plugins/`` that defines one or more
:class:`~backend.tools.base.Tool` subclasses.  Call :func:`load_plugins` at
startup to discover and register them automatically.

Example plugin file (``backend/tools/plugins/my_tool.py``)::

    from backend.tools.base import Tool, ToolResult

    class GreetTool(Tool):
        name = "greet"
        description = "Returns a greeting."
        parameters_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        async def execute(self, name: str = "world", **_) -> ToolResult:
            return ToolResult(ok=True, output=f"Hello, {name}!")
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from backend.tools.base import Tool

if TYPE_CHECKING:
    from backend.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_PLUGINS_DIR = Path(__file__).parent / "plugins"


def load_plugins(registry: "ToolRegistry", plugins_dir: Path | None = None) -> int:
    """Discover and register Tool subclasses from ``plugins_dir``.

    Parameters
    ----------
    registry:
        The :class:`~backend.tools.registry.ToolRegistry` to register tools into.
    plugins_dir:
        Directory to scan.  Defaults to ``backend/tools/plugins/``.

    Returns
    -------
    int
        Number of tools successfully registered.
    """
    directory = Path(plugins_dir) if plugins_dir else _PLUGINS_DIR
    if not directory.exists():
        logger.debug("Plugins directory does not exist: %s", directory)
        return 0

    loaded = 0
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue  # skip __init__.py and private files
        try:
            spec = importlib.util.spec_from_file_location(
                f"slothbrain_plugin_{path.stem}", path
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("Failed to load plugin %s: %s", path.name, exc)
            continue

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj is Tool:
                continue
            if issubclass(obj, Tool) and obj.name:
                try:
                    # Plugin classes must be instantiable without arguments
                    instance = obj()
                    registry.register(instance)
                    loaded += 1
                    logger.info("Plugin tool registered: %s (from %s)", obj.name, path.name)
                except Exception as exc:
                    logger.error(
                        "Failed to instantiate plugin tool %s from %s: %s",
                        obj.name, path.name, exc,
                    )

    return loaded
