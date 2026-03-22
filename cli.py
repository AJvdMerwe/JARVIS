"""
cli.py
───────
Typer-based CLI for the Virtual Personal Assistant.

Usage examples:
    python cli.py chat                              # interactive REPL
    python cli.py chat --voice                      # voice-enabled REPL
    python cli.py ask "What is quantum computing?"
    python cli.py ask "Write a quicksort" --agent code
    python cli.py ask "Research climate change"  --agent research
    python cli.py ask "Apple P/E ratio"          --agent finance
    python cli.py ingest reports/Q3.pdf
    python cli.py ingest ./docs/ --directory --workers 4 --dry-run
    python cli.py docs list
    python cli.py docs search "revenue figures"
    python cli.py docs delete "Q3 Report"
    python cli.py transcribe audio.wav
    python cli.py config show
"""
from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

app = typer.Typer(
    name="assistant",
    help="Virtual Personal Assistant – Chat · Code · News · Search · Documents · Finance · Research",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
docs_app = typer.Typer(help="Manage the document knowledge base")
app.add_typer(docs_app, name="docs")

console = Console()

_VALID_AGENTS = ("chat", "code", "news", "search", "document", "finance", "research")


# ─────────────────────────────────────────────────────────────────────────────
#  chat
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def chat(
    voice: bool = typer.Option(False, "--voice", "-v", help="Enable voice I/O (Whisper STT + TTS)"),
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
        help=f"Force a specific agent: {' | '.join(_VALID_AGENTS)}",
    ),
    session: str = typer.Option("cli", "--session", "-s", help="Session ID"),
    refs: bool = typer.Option(True, "--refs/--no-refs", help="Show source references"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable RAG pre-check for this query"),
) -> None:
    """Send a single query and print the response."""
    if agent and agent not in _VALID_AGENTS:
        console.print(
            f"[red]Unknown agent '{agent}'. Valid options: {', '.join(_VALID_AGENTS)}[/red]"
        )
        raise typer.Exit(1)

    from main import run_query

    with console.status("[yellow]Thinking…[/yellow]"):
        response_text, references = run_query(
            query,
            session_id=session,
            intent=agent,
            enable_rag_precheck=not no_rag,
        )

    console.print(Markdown(response_text))

    if refs and references:
        console.print()
        console.print("[dim]Sources:[/dim]")
        for i, ref in enumerate(references, 1):
            console.print(f"[dim]  [{i}] {ref}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
#  ingest
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory path to ingest"),
    directory: bool = typer.Option(
        False, "--directory", "-d",
        help="Ingest all supported documents in a directory (recursive)",
    ),
    workers: int = typer.Option(
        4, "--workers", "-w",
        help="Parallel workers for text/structured files (PDF/DOCX always sequential)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Detect and plan ingestion without writing to the vector store",
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive",
        help="Recurse into sub-directories (only with --directory)",
    ),
) -> None:
    """
    Ingest a document or directory into the knowledge base.

    PDF, DOCX, XLSX, PPTX files are processed sequentially (pdfium safety).
    TXT, MD, CSV, HTML, JSON, XML files are processed in parallel.
    All content is converted to UTF-8 Markdown before chunking.
    """
    from pathlib import Path

    p = Path(path)

    if directory or p.is_dir():
        # ── Mass upload (directory) ───────────────────────────────────────────
        if not p.is_dir():
            console.print(f"[red]Not a directory: {path}[/red]")
            raise typer.Exit(1)

        from document_processing import MassUploader

        if dry_run:
            console.print(f"[yellow]Dry-run mode — nothing will be written.[/yellow]")

        progress_rows: list[dict] = []

        def _on_progress(outcome, done: int, total: int) -> None:
            status_color = {
                "ok": "green", "duplicate": "yellow",
                "error": "red", "skipped": "dim", "unsupported": "dim",
            }.get(outcome.status, "white")
            progress_rows.append({
                "file": outcome.filename[:45],
                "status": f"[{status_color}]{outcome.status}[/{status_color}]",
                "chunks": str(outcome.chunks_added) if outcome.chunks_added else "—",
                "time": f"{outcome.elapsed_ms:.0f}ms",
                "error": (outcome.error or "")[:60],
            })
            console.print(
                f"  [{done:3d}/{total}] [{status_color}]{outcome.status:<11}[/{status_color}]"
                f"  {outcome.filename[:50]}",
                end="\r" if done < total else "\n",
            )

        uploader = MassUploader(max_workers=workers, on_progress=_on_progress)

        with console.status(f"[yellow]Scanning {path}…[/yellow]"):
            report = uploader.upload_directory(
                p, dry_run=dry_run, recursive=recursive
            )

        # Results table
        table = Table(
            title=f"{'[DRY RUN] ' if dry_run else ''}Ingestion Results — {p.name}",
            show_footer=True,
        )
        table.add_column("File",    style="cyan",  footer="TOTAL")
        table.add_column("Status",  style="white")
        table.add_column("Chunks",  style="green",  footer=str(report.total_chunks_added))
        table.add_column("Time",    style="dim",    footer=f"{report.total_elapsed_ms:.0f}ms")
        table.add_column("Error",   style="red dim")

        for row in progress_rows:
            table.add_row(
                row["file"], row["status"], row["chunks"], row["time"], row["error"]
            )

        console.print()
        console.print(table)
        console.print()
        console.print(
            f"  [green]✓ {report.ok_count} ok[/green]  "
            f"[yellow]⧉ {report.duplicate_count} duplicate[/yellow]  "
            f"[red]✗ {report.error_count} error[/red]  "
            f"[dim]⊘ {report.unsupported_count} unsupported[/dim]  "
            f"  {report.total_chunks_added} chunks added"
        )

    else:
        # ── Single file ───────────────────────────────────────────────────────
        if not p.exists():
            console.print(f"[red]File not found: {path}[/red]")
            raise typer.Exit(1)

        from document_processing import MassUploader

        if dry_run:
            console.print("[yellow]Dry-run mode — nothing will be written.[/yellow]")

        uploader = MassUploader(max_workers=1)
        with console.status(f"[yellow]Ingesting {p.name}…[/yellow]"):
            report = uploader.upload_files([p], dry_run=dry_run)

        if report.ok_count:
            console.print(
                f"[green]✓[/green] Ingested '{p.name}' "
                f"({report.total_chunks_added} chunks added)"
            )
        elif report.duplicate_count:
            console.print(f"[yellow]⧉ Duplicate — '{p.name}' already in knowledge base.[/yellow]")
        elif report.error_count:
            err = report.failed_outcomes[0].error if report.failed_outcomes else "unknown"
            console.print(f"[red]✗ Failed: {err}[/red]")
            raise typer.Exit(1)
        else:
            console.print(f"[dim]Skipped '{p.name}' (unsupported or empty).[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
#  docs subcommands
# ─────────────────────────────────────────────────────────────────────────────

@docs_app.command("list")
def docs_list() -> None:
    """List all documents in the knowledge base."""
    from document_processing import DocumentManager

    dm   = DocumentManager()
    docs = dm.list_documents()

    if not docs:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    table = Table(title=f"Knowledge Base ({len(docs)} documents)")
    table.add_column("Title",   style="bold cyan")
    table.add_column("Type",    style="green")
    table.add_column("Path",    style="dim")
    for d in docs:
        table.add_row(
            d["doc_title"],
            d.get("doctype", "?").upper(),
            d["doc_path"],
        )
    console.print(table)


@docs_app.command("search")
def docs_search(
    query: str = typer.Argument(..., help="Natural-language search query"),
    k: int = typer.Option(5, "--k", help="Number of results to return"),
    doc: Optional[str] = typer.Option(
        None, "--doc", "-d", help="Restrict search to a single document title"
    ),
    threshold: float = typer.Option(
        0.0, "--threshold", "-t",
        help="Minimum cosine similarity (0.0 = return all, 0.5+ = relevant only)",
    ),
) -> None:
    """Semantic search across the knowledge base."""
    from document_processing import DocumentManager

    dm = DocumentManager()

    with console.status("[yellow]Searching…[/yellow]"):
        results = dm.search(query, k=k, doc_title=doc, similarity_threshold=threshold or None)

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

    dm      = DocumentManager()
    removed = dm.delete_document(title)
    if removed:
        console.print(f"[green]Removed '{title}' ({removed} chunks).[/green]")
    else:
        console.print(f"[yellow]Document '{title}' not found.[/yellow]")


@docs_app.command("stats")
def docs_stats() -> None:
    """Show knowledge base statistics."""
    from document_processing import DocumentManager

    dm    = DocumentManager()
    stats = dm.get_stats()

    table = Table(title="Knowledge Base Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="green")
    table.add_row("Total documents", str(stats.total_documents))
    table.add_row("Total chunks",    str(stats.total_chunks))
    table.add_row("Store path",      str(stats.store_path))
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
#  transcribe
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def transcribe(
    audio_file: str = typer.Argument(..., help="Path to audio file (WAV / MP3 / M4A)"),
) -> None:
    """Transcribe an audio file using Whisper (offline)."""
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
    """Show the current configuration (reads from .env)."""
    from config import settings

    table = Table(title="Configuration")
    table.add_column("Setting",  style="cyan")
    table.add_column("Value",    style="green")
    table.add_column("Section",  style="dim")

    rows = [
        # LLM
        ("LLM Backend",            settings.llm_backend,           "LLM"),
        ("Ollama URL",             settings.ollama_base_url,        "LLM"),
        ("Ollama Chat Model",      settings.ollama_model,           "LLM"),
        ("Ollama Reasoning Model", settings.ollama_reasoning_model, "LLM"),
        ("vLLM Model",             settings.vllm_model,             "LLM"),
        ("vLLM URL",               settings.vllm_base_url,          "LLM"),
        # Embeddings
        ("Embedding Backend",      settings.embedding_backend,      "Embeddings"),
        ("Ollama Embed Model",     settings.ollama_embedding_model, "Embeddings"),
        ("HuggingFace Model",      settings.embedding_model,        "Embeddings"),
        ("Embedding Device",       settings.embedding_device,       "Embeddings"),
        ("Embedding Batch Size",   str(settings.embedding_batch_size), "Embeddings"),
        # Vector store
        ("Vector Store Path",      str(settings.vector_store_path), "Vector Store"),
        ("Chunk Size",             str(settings.chunk_size),        "Vector Store"),
        ("Chunk Overlap",          str(settings.chunk_overlap),     "Vector Store"),
        # Research
        ("Research Max Iterations",str(settings.research_max_iterations), "Research"),
        ("Research Max Sources",   str(settings.research_max_sources),    "Research"),
        ("Research Chunk Budget",  str(settings.research_chunk_budget),   "Research"),
        # Voice
        ("Voice Enabled",          str(settings.voice_enabled),    "Voice"),
        ("Whisper Model",          settings.whisper_model,         "Voice"),
        ("Voice Language",         settings.voice_language,        "Voice"),
        # Misc
        ("Agent Max Iterations",   str(settings.agent_max_iterations), "Agent"),
        ("Uploads Path",           str(settings.uploads_path),         "Agent"),
        ("Log Level",              settings.log_level,                 "Agent"),
    ]
    for name, value, section in rows:
        table.add_row(name, str(value), section)

    console.print(table)


if __name__ == "__main__":
    app()
