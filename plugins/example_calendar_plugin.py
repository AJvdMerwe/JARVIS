"""
plugins/example_calendar_plugin.py
────────────────────────────────────
Example plugin demonstrating the plugin system.

Drop any .py file into the plugins/ directory and it will be
auto-discovered when the assistant starts.

This plugin adds:
  • CalendarTool     – returns today's date / day-of-week.
  • CalendarAgent    – answers date/time questions.

To activate: this file is discovered automatically.
To deactivate: rename to _example_calendar_plugin.py (prefix with _).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agents.base_agent import AgentResponse, BaseAgent


# ─────────────────────────────────────────────────────────────────────────────
#  Tool
# ─────────────────────────────────────────────────────────────────────────────

class DateTimeInput(BaseModel):
    timezone_name: str = Field(
        default="UTC",
        description="IANA timezone name, e.g. 'America/New_York', 'Europe/London', 'UTC'",
    )


class CalendarTool(BaseTool):
    """Return current date, time, day-of-week, and week number."""

    name: str = "get_datetime"
    description: str = (
        "Get the current date and time. "
        "Optionally pass a timezone name (IANA format). Defaults to UTC."
    )
    args_schema: Type[BaseModel] = DateTimeInput

    def _run(self, timezone_name: str = "UTC") -> str:
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(timezone_name)
        except Exception:
            tz = timezone.utc
            timezone_name = "UTC"

        now = datetime.now(tz)
        return (
            f"Current date/time ({timezone_name}):\n"
            f"  Date:        {now.strftime('%A, %d %B %Y')}\n"
            f"  Time:        {now.strftime('%H:%M:%S')}\n"
            f"  Week number: {now.isocalendar().week}\n"
            f"  ISO:         {now.isoformat()}"
        )

    async def _arun(self, **kwargs) -> str:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

class CalendarAgent(BaseAgent):
    """Agent that answers date/time/calendar questions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._executor = self._build_react_agent(
            system_prompt=(
                "You answer questions about dates, times, days, and calendars. "
                "Always use the get_datetime tool to get the actual current time."
            )
        )

    @property
    def name(self) -> str:
        return "calendar_agent"

    @property
    def description(self) -> str:
        return (
            "Answer date, time, and calendar questions. "
            "Use for: what day is it, what week, timezone conversions."
        )

    def get_tools(self) -> list[BaseTool]:
        return [CalendarTool()]

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        try:
            result = self._executor.invoke({"input": query})
            return AgentResponse(
                output=result.get("output", ""),
                agent_name=self.name,
                tool_calls=self._extract_tool_calls(
                    result.get("intermediate_steps", [])
                ),
            )
        except Exception as exc:
            return AgentResponse(
                output=f"Calendar error: {exc}",
                agent_name=self.name,
                error=str(exc),
            )


# ─────────────────────────────────────────────────────────────────────────────
#  Plugin registration
# ─────────────────────────────────────────────────────────────────────────────

def register_agents() -> dict[str, type]:
    return {"calendar_agent": CalendarAgent}


def register_tools() -> list[BaseTool]:
    return [CalendarTool()]
