"""
admin/admin_cli.py
───────────────────
Operational admin CLI for the Virtual Personal Assistant.

Commands:
  sessions list              – List all active sessions and their memory size
  sessions clear <id>        – Clear a specific session's memory
  sessions clear-all         – Wipe all sessions

  kb stats                   – Knowledge base statistics
  kb list                    – List all ingested documents
  kb ingest <path>           – Ingest a document
  kb delete <title>          – Remove a document
  kb vacuum                  – Remove duplicate/stale chunks

  traces list [--n N]        – Show recent traces
  traces stats               – Aggregated trace statistics
  traces export <out.jsonl>  – Export all trace data

  cache stats                – Tool cache hit/miss statistics
  cache clear                – Flush the entire tool cache

  prefs show <user>          – Show user preferences
  prefs reset <user>         – Reset user preferences to defaults
  prefs set <user> <k> <v>   – Set a preference value

  scheduler list             – Show registered tasks and last run times
  scheduler run <task>       – Trigger a task immediately
  scheduler enable <task>    – Enable a disabled task
  scheduler disable <task>   – Disable a running task

  health                     – Run startup validation checks
  config                     – Show current configuration

Usage:
  python admin/admin_cli.py sessions list
  python admin/admin_cli.py kb stats
  python admin/admin_cli.py traces list --n 10
  python admin/admin_cli.py cache clear
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

app     = typer.Typer(name="admin", help="Virtual Assistant admin CLI", add_completion=False)
console = Console()

sessions_app  = typer.Typer(help="Manage conversation sessions")
kb_app        = typer.Typer(help="Manage the knowledge base")
traces_app    = typer.Typer(help="Inspect request traces")
cache_app     = typer.Typer(help="Manage the tool cache")
prefs_app     = typer.Typer(help="Manage user preferences")
scheduler_app = typer.Typer(help="Manage scheduled tasks")

app.add_typer(sessions_app,  name="sessions")
app.add_typer(kb_app,        name="kb")
app.add_typer(traces_app,    name="traces")
app.add_typer(cache_app,     name="cache")
app.add_typer(prefs_app,     name="prefs")
app.add_typer(scheduler_app, name="scheduler")


# ─────────────────────────────────────────────────────────────────────────────
#  Sessions
# ─────────────────────────────────────────────────────────────────────────────

@sessions_app.command("list")
def sessions_list():
    """List all persisted session files."""
    from config import settings
    sessions_dir = settings.log_path.parent / "sessions"
    files = sorted(sessions_dir.glob("*.json")) if sessions_dir.exists() else []

    if not files:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(title=f"Sessions ({len(files)})")
    table.add_column("Session ID",  style="cyan")
    table.add_column("Messages",    style="green")
    table.add_column("Size",        style="dim")
    table.add_column("Modified",    style="dim")

    for f in files:
        try:
            data = json.loads(f.read_text())
            import os, datetime
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
            table.add_row(
                f.stem,
                str(len(data)),
                f"{f.stat().st_size} B",
                mtime,
            )
        except Exception:
            table.add_row(f.stem, "?", "?", "?")

    console.print(table)


@sessions_app.command("clear")
def sessions_clear(session_id: str = typer.Argument(..., help="Session ID to clear")):
    """Clear a specific session's conversation memory."""
    from config import settings
    path = settings.log_path.parent / "sessions" / f"{session_id}.json"
    if path.exists():
        path.unlink()
        console.print(f"[green]Cleared session '{session_id}'.[/green]")
    else:
        console.print(f"[yellow]Session '{session_id}' not found.[/yellow]")


@sessions_app.command("clear-all")
def sessions_clear_all():
    """Delete all session memory files."""
    from config import settings
    sessions_dir = settings.log_path.parent / "sessions"
    if not sessions_dir.exists():
        console.print("[yellow]No sessions directory found.[/yellow]")
        return
    if not typer.confirm("Delete ALL session memory files?"):
        raise typer.Abort()
    count = 0
    for f in sessions_dir.glob("*.json"):
        f.unlink()
        count += 1
    console.print(f"[green]Deleted {count} session file(s).[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

@kb_app.command("stats")
def kb_stats():
    """Show knowledge base statistics."""
    from document_processing import DocumentManager
    dm = DocumentManager()
    docs = dm.list_documents()
    console.print(Panel(
        f"[bold]Documents:[/bold] {len(docs)}\n"
        f"[bold]Total chunks:[/bold] {dm.total_chunks}\n"
        f"[bold]Store path:[/bold] [dim]{dm._store._persist_dir}[/dim]",
        title="Knowledge Base Stats",
    ))


@kb_app.command("list")
def kb_list():
    """List all ingested documents."""
    from document_processing import DocumentManager
    dm = DocumentManager()
    docs = dm.list_documents()
    if not docs:
        console.print("[yellow]Knowledge base is empty.[/yellow]")
        return

    table = Table(title=f"Knowledge Base ({len(docs)} documents)")
    table.add_column("Title",  style="cyan bold")
    table.add_column("Type",   style="green")
    table.add_column("Path",   style="dim")
    for d in docs:
        table.add_row(d["doc_title"], d.get("doctype","?").upper(), d.get("doc_path",""))
    console.print(table)


@kb_app.command("ingest")
def kb_ingest(path: str = typer.Argument(..., help="Path to document or directory")):
    """Ingest a document into the knowledge base."""
    from document_processing import DocumentManager, SUPPORTED_SUFFIXES
    dm = DocumentManager()
    p = Path(path)

    if p.is_dir():
        results = dm.ingest_directory(p)
        for name, count in results.items():
            status = "[green]✓[/green]" if count >= 0 else "[red]✗[/red]"
            console.print(f"  {status} {name}: {count} chunks")
    else:
        with console.status(f"[yellow]Ingesting {p.name}…[/yellow]"):
            added = dm.ingest(p)
        console.print(f"[green]✓[/green] {p.name}: {added} new chunks added.")


@kb_app.command("delete")
def kb_delete(title: str = typer.Argument(..., help="Exact document title")):
    """Remove a document from the knowledge base."""
    from document_processing import DocumentManager
    if not typer.confirm(f"Remove '{title}' from knowledge base?"):
        raise typer.Abort()
    dm = DocumentManager()
    removed = dm.delete_document(title)
    if removed:
        console.print(f"[green]Removed '{title}' ({removed} chunks).[/green]")
    else:
        console.print(f"[yellow]Document '{title}' not found.[/yellow]")


@kb_app.command("vacuum")
def kb_vacuum():
    """Re-ingest modified documents and remove orphaned chunks."""
    from core.scheduler import get_scheduler
    scheduler = get_scheduler()
    with console.status("[yellow]Running KB vacuum…[/yellow]"):
        scheduler.run_now("kb_reindex")
    console.print("[green]✓ KB vacuum complete.[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  Traces
# ─────────────────────────────────────────────────────────────────────────────

@traces_app.command("list")
def traces_list(n: int = typer.Option(20, "--n", help="Number of traces to show")):
    """Show the N most recent request traces."""
    from core.tracing import get_tracer
    tracer = get_tracer()
    recent = tracer.store.recent(n)
    if not recent:
        console.print("[yellow]No traces yet.[/yellow]")
        return

    table = Table(title=f"Recent Traces ({len(recent)})")
    table.add_column("Trace ID",   style="dim")
    table.add_column("Session",    style="cyan")
    table.add_column("Query",      style="white")
    table.add_column("Agent",      style="green")
    table.add_column("Latency",    style="yellow")
    table.add_column("Error",      style="red")

    for t in reversed(recent):
        table.add_row(
            t.trace_id,
            t.session_id,
            (t.query[:50] + "…") if len(t.query) > 50 else t.query,
            t.agent_name or "—",
            f"{t.total_ms:.0f}ms",
            t.error[:30] if t.error else "",
        )
    console.print(table)


@traces_app.command("stats")
def traces_stats():
    """Show aggregated trace statistics."""
    from core.tracing import get_tracer
    stats = get_tracer().stats()
    console.print(Panel(
        f"[bold]Total traces:[/bold]  {stats['total']}\n"
        f"[bold]Avg latency:[/bold]   {stats['avg_ms']}ms\n"
        f"[bold]Error rate:[/bold]    {stats['error_rate']:.1%}\n"
        f"[bold]By agent:[/bold]\n" +
        "\n".join(f"  {k}: {v}" for k, v in (stats.get("agents") or {}).items()),
        title="Trace Statistics",
    ))


@traces_app.command("export")
def traces_export(output: str = typer.Argument(..., help="Output .jsonl file path")):
    """Export all trace data to a JSONL file."""
    from config import settings
    src = settings.log_path.parent / "traces.jsonl"
    if not src.exists():
        console.print("[yellow]No trace log found.[/yellow]")
        return
    import shutil
    shutil.copy2(src, output)
    lines = Path(output).read_text().count("\n")
    console.print(f"[green]Exported {lines} traces → {output}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  Cache
# ─────────────────────────────────────────────────────────────────────────────

@cache_app.command("stats")
def cache_stats():
    """Show tool cache statistics."""
    from core.cache import get_cache
    stats = get_cache().stats()
    console.print(Panel(
        "\n".join(f"[bold]{k}:[/bold] {v}" for k, v in stats.items()),
        title="Tool Cache Stats",
    ))


@cache_app.command("clear")
def cache_clear():
    """Flush the entire tool cache."""
    from core.cache import get_cache
    removed = get_cache().clear()
    console.print(f"[green]Cache cleared ({removed} entries removed).[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  User Preferences
# ─────────────────────────────────────────────────────────────────────────────

@prefs_app.command("show")
def prefs_show(user_id: str = typer.Argument(default="default")):
    """Show preferences for a user."""
    from core.user_prefs import get_preferences
    prefs = get_preferences(user_id)
    table = Table(title=f"Preferences: {user_id}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value",   style="green")
    for field_name, value in prefs.model_dump(exclude={"user_id"}).items():
        table.add_row(field_name, str(value))
    console.print(table)


@prefs_app.command("reset")
def prefs_reset(user_id: str = typer.Argument(default="default")):
    """Reset a user's preferences to defaults."""
    from core.user_prefs import get_preferences, invalidate_cache
    if not typer.confirm(f"Reset preferences for '{user_id}'?"):
        raise typer.Abort()
    prefs = get_preferences(user_id)
    prefs.reset()
    invalidate_cache(user_id)
    console.print(f"[green]Reset preferences for '{user_id}'.[/green]")


@prefs_app.command("set")
def prefs_set(
    user_id: str = typer.Argument(...),
    key:     str = typer.Argument(..., help="Preference key"),
    value:   str = typer.Argument(..., help="New value"),
):
    """Set a single preference for a user."""
    from core.user_prefs import get_preferences, invalidate_cache
    prefs = get_preferences(user_id)
    if not hasattr(prefs, key):
        console.print(f"[red]Unknown preference: '{key}'[/red]")
        raise typer.Exit(1)
    try:
        # Attempt type coercion
        current = getattr(prefs, key)
        if isinstance(current, bool):
            coerced = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            coerced = int(value)
        elif isinstance(current, list):
            coerced = [v.strip() for v in value.split(",")]
        else:
            coerced = value
        setattr(prefs, key, coerced)
        prefs.save()
        invalidate_cache(user_id)
        console.print(f"[green]Set {key} = {coerced} for '{user_id}'.[/green]")
    except Exception as exc:
        console.print(f"[red]Failed to set '{key}': {exc}[/red]")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  Scheduler
# ─────────────────────────────────────────────────────────────────────────────

@scheduler_app.command("list")
def scheduler_list():
    """Show all registered tasks."""
    from core.scheduler import get_scheduler
    stats = get_scheduler().stats()
    table = Table(title="Scheduled Tasks")
    table.add_column("Task",        style="cyan")
    table.add_column("Enabled",     style="green")
    table.add_column("Interval",    style="yellow")
    table.add_column("Run Count",   style="white")
    table.add_column("Errors",      style="red")
    table.add_column("Last Run",    style="dim")
    for t in stats:
        table.add_row(
            t["name"],
            "[green]✓[/green]" if t["enabled"] else "[red]✗[/red]",
            f"{t['interval_m']}m",
            str(t["run_count"]),
            str(t["error_count"]),
            (t["last_run"] or "never")[:19],
        )
    console.print(table)


@scheduler_app.command("run")
def scheduler_run(task: str = typer.Argument(..., help="Task name")):
    """Trigger a task immediately."""
    from core.scheduler import get_scheduler
    with console.status(f"[yellow]Running '{task}'…[/yellow]"):
        try:
            success = get_scheduler().run_now(task)
            if success:
                console.print(f"[green]✓ Task '{task}' completed.[/green]")
            else:
                console.print(f"[red]✗ Task '{task}' failed (check logs).[/red]")
        except KeyError:
            console.print(f"[red]Task '{task}' not found.[/red]")
            raise typer.Exit(1)


@scheduler_app.command("enable")
def scheduler_enable(task: str = typer.Argument(...)):
    from core.scheduler import get_scheduler
    get_scheduler().enable(task)
    console.print(f"[green]Enabled task '{task}'.[/green]")


@scheduler_app.command("disable")
def scheduler_disable(task: str = typer.Argument(...)):
    from core.scheduler import get_scheduler
    get_scheduler().disable(task)
    console.print(f"[yellow]Disabled task '{task}'.[/yellow]")


# ─────────────────────────────────────────────────────────────────────────────
#  Health + Config
# ─────────────────────────────────────────────────────────────────────────────

@app.command()
def health():
    """Run startup validation checks."""
    from api.startup_validator import run_startup_checks
    report = run_startup_checks()
    if report.all_passed:
        console.print("[green]All checks passed.[/green]")
    else:
        console.print(f"[red]{len(report.failed)} check(s) failed.[/red]")
        raise typer.Exit(1)


@app.command()
def config():
    """Display current configuration."""
    from config import settings
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value",   style="green")
    for k, v in settings.model_dump().items():
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    app()
