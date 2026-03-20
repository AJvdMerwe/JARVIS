"""
main.py
────────
Virtual Personal Assistant – main entry point.

Modes:
  • Interactive REPL  – rich terminal chat loop (default).
  • Voice loop        – Whisper STT → agent → TTS.
  • Single query      – run one query and exit (--query flag).
  • Ingest            – index a document directly (--ingest flag).
"""
from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.theme import Theme

from agents.orchestrator import Intent, Orchestrator
from config import settings

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.log_path, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Rich console ─────────────────────────────────────────────────────────────
_THEME = Theme({
    "user":      "bold cyan",
    "assistant": "bold green",
    "agent":     "dim italic",
    "reference": "dim blue",
    "error":     "bold red",
    "info":      "yellow",
})
console = Console(theme=_THEME)

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          Virtual Personal Assistant  v1.0                    ║
║  Agents: Code · News · Search · Documents                    ║
║  Backend: {backend:<10}  Voice: {voice}                      ║
║  Type  :quit  to exit  |  :help  for commands                ║
╚══════════════════════════════════════════════════════════════╝
"""

_HELP = """
## Commands

| Command | Description |
|---------|-------------|
| `:quit` / `:exit` | Exit the assistant |
| `:clear` | Clear conversation memory |
| `:docs` | List ingested documents |
| `:ingest <path>` | Ingest a document into the knowledge base |
| `:delete <title>` | Remove a document from the knowledge base |
| `:agent <name>` | Force next query to a specific agent (code/news/search/document) |
| `:history` | Show conversation history |
| `:voice` | Toggle voice input/output |
| `:help` | Show this help |

## Agent routing (automatic)
The assistant auto-detects intent from your message:
- **Code tasks** → CodeAgent  (write, debug, review, execute Python)
- **News tasks** → NewsAgent  (headlines, topic news, briefings)
- **Document Q&A** → DocumentAgent  (PDF/DOCX/XLSX/PPTX chat + references)
- **Everything else** → SearchAgent  (web search, Wikipedia, calculations)
"""


# ─────────────────────────────────────────────────────────────────────────────
#  REPL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_response(response, show_refs: bool = True, show_tools: bool = False) -> None:
    """Render an AgentResponse to the console."""
    agent_label = f"[{response.agent_name}]"
    console.print(f"\n[agent]{agent_label}[/agent]")

    if response.error:
        console.print(f"[error]Error: {response.error}[/error]")
        return

    console.print(Markdown(response.output))

    if show_refs and response.has_references:
        console.print()
        console.print("[reference]Sources:[/reference]")
        for i, ref in enumerate(response.references, 1):
            console.print(f"[reference]  [{i}] {ref}[/reference]")

    if show_tools and response.tool_calls:
        console.print()
        for tool_name, tool_in, tool_out in response.tool_calls:
            console.print(
                f"[dim]  ⚙ {tool_name}({str(tool_in)[:60]}) → {str(tool_out)[:100]}[/dim]"
            )


def _handle_command(
    cmd: str,
    orchestrator: Orchestrator,
    voice_enabled: list[bool],
    forced_intent: list[Optional[str]],
) -> bool:
    """
    Handle a ':command'. Returns True if the REPL should continue, False to quit.
    """
    parts = cmd.strip().split(None, 1)
    verb = parts[0].lower()
    arg  = parts[1] if len(parts) > 1 else ""

    if verb in (":quit", ":exit"):
        console.print("\n[info]Goodbye![/info]")
        return False

    elif verb == ":help":
        console.print(Markdown(_HELP))

    elif verb == ":clear":
        orchestrator.clear_memory()
        console.print("[info]Memory cleared.[/info]")

    elif verb == ":docs":
        docs = orchestrator.list_documents()
        if not docs:
            console.print("[info]No documents in the knowledge base.[/info]")
        else:
            console.print(f"[info]{len(docs)} document(s):[/info]")
            for d in docs:
                console.print(f"  • [bold]{d['doc_title']}[/bold] ({d.get('doctype','?').upper()})  {d['doc_path']}")

    elif verb == ":ingest":
        if not arg:
            console.print("[error]Usage: :ingest <file_path>[/error]")
        else:
            with console.status(f"[info]Ingesting {arg}…[/info]"):
                try:
                    result = orchestrator.ingest_document(arg)
                    console.print(f"[info]{result}[/info]")
                except Exception as exc:
                    console.print(f"[error]Ingest failed: {exc}[/error]")

    elif verb == ":delete":
        if not arg:
            console.print("[error]Usage: :delete <document_title>[/error]")
        else:
            response = orchestrator.run(f"delete document '{arg}'", intent=Intent.DOCUMENT)
            console.print(f"[info]{response.output}[/info]")

    elif verb == ":agent":
        valid = {i.value for i in Intent} - {"unknown"}
        if arg.lower() in valid:
            forced_intent[0] = arg.lower()
            console.print(f"[info]Next query forced to: {arg.lower()}[/info]")
        else:
            console.print(f"[error]Unknown agent. Choose from: {', '.join(sorted(valid))}[/error]")

    elif verb == ":history":
        msgs = orchestrator.memory.messages
        if not msgs:
            console.print("[info]No history yet.[/info]")
        else:
            for msg in msgs[-20:]:  # last 10 turns
                from langchain_core.messages import HumanMessage
                role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                console.print(f"[dim]{role}: {str(msg.content)[:200]}[/dim]")

    elif verb == ":voice":
        voice_enabled[0] = not voice_enabled[0]
        state = "ON" if voice_enabled[0] else "OFF"
        console.print(f"[info]Voice: {state}[/info]")

    else:
        console.print(f"[error]Unknown command: {verb}. Type :help for help.[/error]")

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Voice loop
# ─────────────────────────────────────────────────────────────────────────────

def _voice_listen() -> Optional[str]:
    """Capture one utterance from the microphone and transcribe it."""
    try:
        from core.voice import MicrophoneListener
        listener = MicrophoneListener()
        return listener.listen()
    except ImportError:
        console.print("[error]Voice input dependencies not installed.[/error]")
        return None


def _voice_speak(text: str) -> None:
    """Speak the response text using TTS."""
    try:
        from core.voice import speak
        speak(text)
    except Exception as exc:
        logger.debug("TTS failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  Main REPL
# ─────────────────────────────────────────────────────────────────────────────

def run_repl(
    session_id: Optional[str] = None,
    voice: bool = False,
) -> None:
    """
    Start the interactive REPL.

    Args:
        session_id: Persistent session identifier (auto-generated if None).
        voice:      Start with voice I/O enabled.
    """
    session_id = session_id or str(uuid.uuid4())[:8]
    orchestrator = Orchestrator(session_id=session_id)

    voice_enabled  = [voice and settings.voice_enabled]
    forced_intent: list[Optional[str]] = [None]

    console.print(
        _BANNER.format(
            backend=settings.llm_backend,
            voice="ON" if voice_enabled[0] else "OFF",
        )
    )
    console.print(f"[dim]Session: {session_id}[/dim]\n")

    while True:
        try:
            # ── Get input ───────────────────────────────────────────────────
            if voice_enabled[0]:
                console.print("[info]🎙  Listening (speak now)…[/info]")
                user_input = _voice_listen()
                if user_input:
                    console.print(f"[user]You (voice):[/user] {user_input}")
                else:
                    console.print("[dim]Nothing heard, try again.[/dim]")
                    continue
            else:
                user_input = Prompt.ask("\n[user]You[/user]").strip()

            if not user_input:
                continue

            # ── Command? ────────────────────────────────────────────────────
            if user_input.startswith(":"):
                should_continue = _handle_command(
                    user_input, orchestrator, voice_enabled, forced_intent
                )
                if not should_continue:
                    break
                continue

            # ── Run agent ───────────────────────────────────────────────────
            kwargs: dict = {}
            if forced_intent[0]:
                kwargs["intent"] = forced_intent[0]
                forced_intent[0] = None

            with console.status("[info]Thinking…[/info]"):
                response = orchestrator.run(user_input, **kwargs)

            console.print(Rule(style="dim"))
            _print_response(response, show_refs=True)

            # ── TTS ─────────────────────────────────────────────────────────
            if voice_enabled[0] and response.output:
                _voice_speak(response.output)

        except KeyboardInterrupt:
            console.print("\n[info]Interrupted. Type :quit to exit.[/info]")
        except EOFError:
            break


# ─────────────────────────────────────────────────────────────────────────────
#  Single-shot helpers (used by CLI)
# ─────────────────────────────────────────────────────────────────────────────

def run_query(
    query: str,
    session_id: str = "cli",
    intent: Optional[str] = None,
) -> str:
    """Run a single query and return the full response string."""
    orch = Orchestrator(session_id=session_id)
    kwargs = {"intent": intent} if intent else {}
    response = orch.run(query, **kwargs)
    return response.full_response()


def run_ingest(file_path: str) -> str:
    """Ingest a document and return a status string."""
    orch = Orchestrator(session_id="ingest")
    return orch.ingest_document(file_path)


if __name__ == "__main__":
    run_repl()
