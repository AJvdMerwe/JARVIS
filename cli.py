"""
cli.py
───────
Typer-based CLI for the Virtual Personal Assistant.

Usage examples:
    python cli.py chat                        # interactive REPL
    python cli.py chat --voice                # voice-enabled REPL
    python cli.py ask "What is quantum computing?"
    python cli.py ask "Write a quicksort" --agent code
    python cli.py ingest reports/Q3.pdf
    python cli.py ingest ./docs/ --directory
    python cli.py docs list
    python cli.py docs search "revenue figures"
    python cli.py docs delete "Q3 Report"
    python cli.py transcribe audio.wav
    python cli.py config show
"""
from __future__ import annotations

from typing import Optional

import typer
import logging

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

app = typer.Typer(
    name="assistant",
    help="Virtual Personal Assistant – Code · News · Search · Documents",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
docs_app = typer.Typer(help="Manage the document knowledge base")
app.add_typer(docs_app, name="docs")

console = Console()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  chat
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def chat(
    voice: bool = typer.Option(False, "--voice", "-v", help="Enable voice I/O"),
    session: str = typer.Option("", "--session", "-s", help="Session ID for persistent memory"),
) -> None:
    """Start the interactive REPL chat."""
    from main import run_repl
    run_repl(session_id=session or None, voice=voice)


# ─────────────────────────────────────────────────────────────────────────────
#  ask
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def ask(
    query: str = typer.Argument(..., help="The question or task"),
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a",
        help="Force agent: code | news | search | document",
    ),
    session: str = typer.Option("cli", "--session", "-s"),
    refs: bool = typer.Option(True, "--refs/--no-refs", help="Show source references"),
) -> None:
    """Send a single query and print the response."""
    from main import run_query
    logger.info(f"Agent called: {agent}")
    with console.status("[yellow]Thinking…[/yellow]"):
        response_text = run_query(query, session_id=session, intent=agent)
    logger.debug(f"agent response: {response_text}")
    console.print(Markdown(response_text))


# ─────────────────────────────────────────────────────────────────────────────
#  ingest
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory path to ingest"),
    directory: bool = typer.Option(False, "--directory", "-d", help="Ingest all docs in a directory"),
) -> None:
    """Ingest a document (or directory of documents) into the knowledge base."""
    from agents import Orchestrator

    orch = Orchestrator(session_id="ingest_cli")

    if directory:
        from pathlib import Path
        from document_processing import SUPPORTED_SUFFIXES

        dir_path = Path(path)
        if not dir_path.is_dir():
            console.print(f"[red]Not a directory: {path}[/red]")
            raise typer.Exit(1)

        files = [f for f in dir_path.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES]
        if not files:
            console.print(f"[yellow]No supported documents found in {path}[/yellow]")
            return

        results = {}
        for f in files:
            with console.status(f"[yellow]Ingesting {f.name}…[/yellow]"):
                try:
                    msg = orch.ingest_document(str(f))
                    results[f.name] = ("✓", msg)
                except Exception as exc:
                    results[f.name] = ("✗", str(exc))

        table = Table(title="Ingestion Results")
        table.add_column("File", style="cyan")
        table.add_column("Status")
        table.add_column("Details")
        for fname, (status, msg) in results.items():
            color = "green" if status == "✓" else "red"
            table.add_row(fname, f"[{color}]{status}[/{color}]", msg)
        console.print(table)

    else:
        with console.status(f"[yellow]Ingesting {path}…[/yellow]"):
            try:
                msg = orch.ingest_document(path)
                console.print(f"[green]✓[/green] {msg}")
            except Exception as exc:
                console.print(f"[red]✗ Failed: {exc}[/red]")
                raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  docs subcommands
# ─────────────────────────────────────────────────────────────────────────────

@docs_app.command("list")
def docs_list() -> None:
    """List all documents in the knowledge base."""
    from agents import Orchestrator

    orch = Orchestrator(session_id="docs_cli")
    docs = orch.list_documents()

    if not docs:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    table = Table(title=f"Knowledge Base ({len(docs)} documents)")
    table.add_column("Title", style="bold cyan")
    table.add_column("Type", style="green")
    table.add_column("Path", style="dim")
    for d in docs:
        table.add_row(d["doc_title"], d.get("doctype", "?").upper(), d["doc_path"])
    console.print(table)


@docs_app.command("search")
def docs_search(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, "--k", help="Number of results"),
    doc: Optional[str] = typer.Option(None, "--doc", "-d", help="Restrict to a document"),
) -> None:
    """Semantic search across the knowledge base."""
    from document_processing import DocumentManager

    dm = DocumentManager()

    with console.status("[yellow]Searching…[/yellow]"):
        results = dm.search(query, k=k, doc_title=doc)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(Markdown(dm.format_search_results(results)))


@docs_app.command("delete")
def docs_delete(
    title: str = typer.Argument(..., help="Exact document title to remove"),
) -> None:
    """Remove a document from the knowledge base."""
    from document_processing import DocumentManager

    if not typer.confirm(f"Delete '{title}' from the knowledge base?"):
        raise typer.Abort()

    dm = DocumentManager()
    removed = dm.delete_document(title)
    if removed:
        console.print(f"[green]Removed '{title}' ({removed} chunks).[/green]")
    else:
        console.print(f"[yellow]Document '{title}' not found.[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
#  transcribe
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def transcribe(
    audio_file: str = typer.Argument(..., help="Path to audio file (WAV/MP3/M4A)"),
) -> None:
    """Transcribe an audio file using Whisper."""
    from core.voice import transcribe_file
    from pathlib import Path

    p = Path(audio_file)
    if not p.exists():
        console.print(f"[red]File not found: {audio_file}[/red]")
        raise typer.Exit(1)

    with console.status("[yellow]Transcribing…[/yellow]"):
        text = transcribe_file(p)

    console.print(Markdown(f"**Transcription:**\n\n{text}"))


# ─────────────────────────────────────────────────────────────────────────────
#  config
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def config(
    show: bool = typer.Option(True, "--show", help="Display current configuration"),
) -> None:
    """Show the current configuration."""
    from config import settings

    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    rows = [
        ("LLM Backend",     settings.llm_backend),
        ("Ollama Model",    settings.ollama_model),
        ("Ollama URL",      settings.ollama_base_url),
        ("vLLM Model",      settings.vllm_model),
        ("vLLM URL",        settings.vllm_base_url),
        ("Whisper Model",   settings.whisper_model),
        ("Voice Enabled",   str(settings.voice_enabled)),
        ("Embedding Model", settings.embedding_model),
        ("Vector Store",    str(settings.vector_store_path)),
        ("Uploads Path",    str(settings.uploads_path)),
        ("Chunk Size",      str(settings.chunk_size)),
        ("Chunk Overlap",   str(settings.chunk_overlap)),
        ("Agent Iterations",str(settings.agent_max_iterations)),
        ("Log Level",       settings.log_level),
    ]
    for name, value in rows:
        table.add_row(name, value)

    console.print(table)


if __name__ == "__main__":
    app()
