"""
plugins/plugin_loader.py
─────────────────────────
Dynamic plugin system for loading additional agents and tools at runtime.

Plugins are Python packages (or plain modules) that expose:
  • ``register_agents() -> dict[str, type[BaseAgent]]``
  • ``register_tools()  -> list[BaseTool]``

They are discovered via two mechanisms:
  1. ``plugins/`` directory in the project root — drop a .py file or package.
  2. Installed packages that declare the entry-point group
     ``virtual_assistant.plugins`` in their pyproject.toml.

This allows third parties to extend the assistant without modifying core code
(Open/Closed Principle).

Example plugin (plugins/weather_plugin.py):

    from langchain_core.tools import BaseTool
    from agents.base_agent import BaseAgent, AgentResponse

    class WeatherTool(BaseTool):
        name = "get_weather"
        description = "Get current weather for a city."
        def _run(self, city: str) -> str:
            return f"Weather in {city}: 22°C, partly cloudy."
        def _arun(self, **kw): raise NotImplementedError

    class WeatherAgent(BaseAgent):
        @property
        def name(self): return "weather_agent"
        @property
        def description(self): return "Get weather information."
        def get_tools(self): return [WeatherTool()]
        def run(self, query, **kw):
            return AgentResponse(output=self.get_tools()[0]._run(query), agent_name=self.name)

    def register_agents():
        return {"weather_agent": WeatherAgent}

    def register_tools():
        return [WeatherTool()]
"""
from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_PLUGINS_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
#  Discovery helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_module_from_path(path: Path):
    """Import a .py file as a module without installing it."""
    module_name = f"_va_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)   # type: ignore[union-attr]
    return module


def _discover_file_plugins() -> list:
    """Find all .py files in the plugins/ directory (excluding __init__ and loader)."""
    modules = []
    for path in sorted(_PLUGINS_DIR.glob("*.py")):
        if path.name.startswith("_") or path.name == "plugin_loader.py":
            continue
        try:
            mod = _load_module_from_path(path)
            modules.append(mod)
            logger.info("Loaded file plugin: %s", path.name)
        except Exception as exc:
            logger.warning("Failed to load plugin '%s': %s", path.name, exc)
    return modules


def _discover_entrypoint_plugins() -> list:
    """Load plugins registered as setuptools entry-points."""
    modules = []
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="virtual_assistant.plugins")
        for ep in eps:
            try:
                mod = ep.load()
                modules.append(mod)
                logger.info("Loaded entry-point plugin: %s", ep.name)
            except Exception as exc:
                logger.warning("Failed to load entry-point '%s': %s", ep.name, exc)
    except Exception as exc:
        logger.debug("Entry-point discovery error: %s", exc)
    return modules


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_all_plugins() -> tuple[dict[str, type], list]:
    """
    Discover and load all plugins.

    Returns:
        agents: dict mapping agent name → agent class.
        tools:  flat list of extra BaseTool instances.
    """
    all_modules = _discover_file_plugins() + _discover_entrypoint_plugins()

    agent_classes: dict[str, type] = {}
    extra_tools: list = []

    for mod in all_modules:
        # Collect agent registrations
        if hasattr(mod, "register_agents"):
            try:
                new_agents: dict = mod.register_agents()
                for name, cls in new_agents.items():
                    if name in agent_classes:
                        logger.warning("Plugin agent '%s' overrides existing entry.", name)
                    agent_classes[name] = cls
                    logger.info("  + agent: %s", name)
            except Exception as exc:
                logger.warning("register_agents() failed in %s: %s", mod.__name__, exc)

        # Collect tool registrations
        if hasattr(mod, "register_tools"):
            try:
                new_tools: list = mod.register_tools()
                extra_tools.extend(new_tools)
                for t in new_tools:
                    logger.info("  + tool:  %s", getattr(t, "name", type(t).__name__))
            except Exception as exc:
                logger.warning("register_tools() failed in %s: %s", mod.__name__, exc)

    logger.info(
        "Plugin loading complete: %d agent(s), %d tool(s).",
        len(agent_classes), len(extra_tools),
    )
    return agent_classes, extra_tools


def inject_into_orchestrator(orchestrator) -> None:
    """
    Load plugins and inject discovered agents into a running Orchestrator.
    New agent names are added to ``orchestrator._agents`` using a synthetic
    Intent value so the router can reach them.

    Args:
        orchestrator: A live ``Orchestrator`` instance.
    """
    agent_classes, extra_tools = load_all_plugins()
    if not agent_classes:
        return

    for name, cls in agent_classes.items():
        try:
            agent_instance = cls(memory=orchestrator.memory)
            # Use string keys for plugin agents (the router will need updating
            # to handle them; for now they're reachable via forced intent)
            orchestrator._agents[name] = agent_instance   # type: ignore[index]
            logger.info("Injected plugin agent '%s' into orchestrator.", name)
        except Exception as exc:
            logger.warning("Could not instantiate plugin agent '%s': %s", name, exc)
