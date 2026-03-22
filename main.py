"""
main.py
────────
Virtual Personal Assistant – main entry point.

Modes:
  • Interactive REPL  – rich terminal chat loop (default).
  • Voice loop        – Whisper STT → agent → TTS.
  • Single query      – run one query and exit (--query flag).
"""
from __future__ import annotations

import logging
import sys
import uuid
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.theme import Theme

from agents import Intent, Orchestrator
from config import settings

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.log_path, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Rich console ──────────────────────────────────────────────────────────────
_THEME = Theme({
    "user":      "bold cyan",
    "assistant": "bold green",
    "agent":     "dim italic",
    "reference": "dim blue",
    "error":     "bold red",
    "info":      "yellow",
    "rag":       "dim magenta",
})
console = Console(theme=_THEME)

_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          Virtual Personal Assistant  v1.0                        ║
║  Chat · Code · News · Search · Docs · Finance · Research         ║
║  Backend: {backend:<10}  Embed: {embed:<16}  Voice: {voice}     ║
║  Type :help for commands  ·  :quit to exit                       ║
╚══════════════════════════════════════════════════════════════════╝
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
| `:agent <name>` | Force next query to a specific agent |
| `:history` | Show recent conversation turns |
| `:voice` | Toggle voice input/output |
| `:rag on/off` | Toggle the RAG pre-check |
| `:help` | Show this help |

## Agents

| Agent | Trigger keywords |
|-------|-----------------|
| `chat` | Greetings, opinions, creative writing, explanations |
| `code` | write / debug / review / execute code |
| `news` | headlines, latest news, RSS feeds |
| `search` | factual questions, web search, Wikipedia, calculations |
| `document` | PDF/DOCX/XLSX/PPTX Q&A, knowledge base |
| `finance` | stock price, P/E ratio, earnings, financial analysis |
| `research` | deep dive, comprehensive report, literature review |

## Agent routing (automatic)
Intent is detected in two stages: keyword regex (fast) then LLM classifier
(ambiguous queries only). Force a specific agent with `:agent <name>`.

## RAG pre-check (automatic)
After intent detection the vector store is scanned for a direct answer.
If a sufficiently relevant answer is found it is returned immediately
without invoking any agent. Disable with `:rag off`.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Response rendering
# ─────────────────────────────────────────────────────────────────────────────

def _print_response(response, show_refs: bool = True, show_tools: bool = False) -> None:
    """Render an AgentResponse to the console."""
    agent_label = f"[{response.agent_name}]"
    is_rag      = response.agent_name == "rag_precheck"
    style       = "rag" if is_rag else "agent"
    prefix      = "📚 " if is_rag else ""

    console.print(f"\n[{style}]{prefix}{agent_label}[/{style}]")

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

    # Show research metadata if present
    if response.agent_name == "deep_research_agent" and response.metadata:
        m = response.metadata
        console.print(
            f"\n[dim]  Research: {m.get('sources_found', 0)} sources · "
            f"{m.get('iterations', 0)} iterations · "
            f"{m.get('elapsed_s', 0):.1f}s · "
            f"model: {m.get('model', '?')}[/dim]"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Command handler
# ─────────────────────────────────────────────────────────────────────────────

_VALID_AGENTS = {i.value for i in Intent} - {"unknown"}


def _handle_command(
    cmd:           str,
    orchestrator:  Orchestrator,
    voice_enabled: list[bool],
    forced_intent: list[Optional[str]],
    rag_enabled:   list[bool],
) -> bool:
    """Handle a ':command'. Returns True to continue, False to quit."""
    parts = cmd.strip().split(None, 1)
    verb  = parts[0].lower()
    arg   = parts[1] if len(parts) > 1 else ""

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
                dtype = d.get("doctype", "?").upper()
                console.print(
                    f"  • [bold]{d['doc_title']}[/bold]  "
                    f"[dim]({dtype})[/dim]  {d['doc_path']}"
                )

    elif verb == ":ingest":
        if not arg:
            console.print("[error]Usage: :ingest <file_path>[/error]")
        else:
            from pathlib import Path
            from document_processing import MassUploader
            p = Path(arg.strip())
            uploader = MassUploader(max_workers=1)
            with console.status(f"[info]Ingesting {p.name}…[/info]"):
                try:
                    report = uploader.upload_files([p])
                    if report.ok_count:
                        console.print(
                            f"[info]✓ Ingested '{p.name}' "
                            f"({report.total_chunks_added} chunks).[/info]"
                        )
                    elif report.duplicate_count:
                        console.print(f"[info]⧉ '{p.name}' already in KB.[/info]")
                    else:
                        err = report.failed_outcomes[0].error if report.failed_outcomes else "?"
                        console.print(f"[error]✗ Ingest failed: {err}[/error]")
                except Exception as exc:
                    console.print(f"[error]✗ Ingest failed: {exc}[/error]")

    elif verb == ":delete":
        if not arg:
            console.print("[error]Usage: :delete <document_title>[/error]")
        else:
            response = orchestrator.run(f"delete document '{arg}'", intent=Intent.DOCUMENT)
            console.print(f"[info]{response.output}[/info]")

    elif verb == ":agent":
        if arg.lower() in _VALID_AGENTS:
            forced_intent[0] = arg.lower()
            console.print(f"[info]Next query forced to: {arg.lower()}[/info]")
        else:
            console.print(
                f"[error]Unknown agent '{arg}'. "
                f"Choose from: {', '.join(sorted(_VALID_AGENTS))}[/error]"
            )

    elif verb == ":history":
        msgs = orchestrator.memory.messages
        if not msgs:
            console.print("[info]No history yet.[/info]")
        else:
            for msg in msgs[-20:]:
                from langchain_core.messages import HumanMessage
                role = "You" if isinstance(msg, HumanMessage) else "Assistant"
                console.print(f"[dim]{role}: {str(msg.content)[:200]}[/dim]")

    elif verb == ":voice":
        voice_enabled[0] = not voice_enabled[0]
        state = "ON" if voice_enabled[0] else "OFF"
        console.print(f"[info]Voice: {state}[/info]")

    elif verb == ":rag":
        if arg.lower() in ("on", "off"):
            rag_enabled[0] = (arg.lower() == "on")
            orchestrator._enable_rag_precheck = rag_enabled[0]
            console.print(f"[info]RAG pre-check: {'ON' if rag_enabled[0] else 'OFF'}[/info]")
        else:
            console.print("[error]Usage: :rag on|off[/error]")

    else:
        console.print(f"[error]Unknown command: {verb}. Type :help for help.[/error]")

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Voice helpers
# ─────────────────────────────────────────────────────────────────────────────

def _voice_listen() -> Optional[str]:
    try:
        from core.voice import MicrophoneListener
        return MicrophoneListener().listen()
    except ImportError:
        console.print("[error]Voice input dependencies not installed.[/error]")
        return None


def _voice_speak(text: str) -> None:
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

    Parameters
    ----------
    session_id : str, optional
        Persistent session identifier (auto-generated if None).
    voice : bool
        Start with voice I/O enabled.
    """
    session_id   = session_id or str(uuid.uuid4())[:8]
    orchestrator = Orchestrator(session_id=session_id)

    voice_enabled:  list[bool]          = [voice and settings.voice_enabled]
    forced_intent:  list[Optional[str]] = [None]
    rag_enabled:    list[bool]          = [True]

    console.print(
        _BANNER.format(
            backend=settings.llm_backend,
            embed=settings.embedding_backend + ":" + (
                settings.ollama_embedding_model
                if settings.embedding_backend == "ollama"
                else settings.embedding_model.split("/")[-1]
            ),
            voice="ON" if voice_enabled[0] else "OFF",
        )
    )
    console.print(f"[dim]Session: {session_id}[/dim]\n")

    while True:
        try:
            # ── Get input ─────────────────────────────────────────────────────
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

            # ── Command? ──────────────────────────────────────────────────────
            if user_input.startswith(":"):
                should_continue = _handle_command(
                    user_input, orchestrator,
                    voice_enabled, forced_intent, rag_enabled,
                )
                if not should_continue:
                    break
                continue

            # ── Run agent ─────────────────────────────────────────────────────
            kwargs: dict = {}
            if forced_intent[0]:
                kwargs["intent"] = forced_intent[0]
                forced_intent[0] = None

            # Research agent gives progress indication
            is_research = (
                kwargs.get("intent") == "research"
                or orchestrator.route_only(user_input) == Intent.RESEARCH
            )
            status_msg = (
                "[info]Researching (this may take 30–120s)…[/info]"
                if is_research
                else "[info]Thinking…[/info]"
            )

            with console.status(status_msg):
                response = orchestrator.run(user_input, **kwargs)

            console.print(Rule(style="dim"))
            _print_response(response, show_refs=True)

            # ── TTS ───────────────────────────────────────────────────────────
            if voice_enabled[0] and response.output:
                _voice_speak(response.output)

        except KeyboardInterrupt:
            console.print("\n[info]Interrupted. Type :quit to exit.[/info]")
        except EOFError:
            break


# ─────────────────────────────────────────────────────────────────────────────
#  Single-shot helpers (used by cli.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_query(
    query:              str,
    session_id:         str  = "cli",
    intent:             Optional[str] = None,
    enable_rag_precheck: bool = True,
) -> tuple[str, list[str]]:
    """
    Run a single query and return (output_text, references).

    Parameters
    ----------
    query              : The user's question.
    session_id         : Session identifier for memory persistence.
    intent             : Force a specific agent (e.g. 'research', 'finance').
    enable_rag_precheck: Whether to run the vector-store pre-check.

    Returns
    -------
    tuple[str, list[str]]
        (full_response_markdown, list_of_source_references)
    """
    orch = Orchestrator(
        session_id=session_id,
        enable_rag_precheck=enable_rag_precheck,
    )
    kwargs = {"intent": intent} if intent else {}
    resp   = orch.run(query, **kwargs)
    return resp.full_response(), resp.references


if __name__ == "__main__":
    run_repl()
