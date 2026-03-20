"""
api/startup_validator.py
─────────────────────────
Pre-flight validation that runs at server startup before FastAPI begins
accepting requests. Catches misconfiguration early rather than at
request time.

Checks:
  1. LLM connectivity  — can we reach the Ollama / vLLM server?
  2. Embedding model   — can we load the embedding model?
  3. Vector store      — can we open the ChromaDB path?
  4. Data directories  — uploads and logs directories are writable.
  5. .env completeness — warn about missing optional settings.

Results are logged at INFO/WARNING/ERROR and also returned as a dict
so the /health endpoint can report them.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name:    str
    passed:  bool
    message: str
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def log(self) -> None:
        level = logging.INFO if self.passed else logging.WARNING
        mark  = "✓" if self.passed else "✗"
        logger.log(level, "  %s  %s: %s (%.0fms)", mark, self.name, self.message, self.latency_ms)


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "checks": [
                {
                    "name":       c.name,
                    "passed":     c.passed,
                    "message":    c.message,
                    "latency_ms": round(c.latency_ms, 1),
                }
                for c in self.checks
            ],
        }

    def print_summary(self) -> None:
        logger.info("─" * 50)
        logger.info("  Startup validation")
        logger.info("─" * 50)
        for c in self.checks:
            c.log()
        logger.info("─" * 50)
        if self.all_passed:
            logger.info("  All checks passed. Server is ready.")
        else:
            failed_names = [c.name for c in self.failed]
            logger.warning("  %d check(s) failed: %s", len(self.failed), ", ".join(failed_names))
        logger.info("─" * 50)


# ─────────────────────────────────────────────────────────────────────────────
#  Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_llm_connectivity() -> CheckResult:
    """Ping the configured LLM backend."""
    t0 = time.monotonic()
    try:
        import httpx
        if settings.llm_backend == "ollama":
            url = f"{settings.ollama_base_url}/api/tags"
        else:
            url = f"{settings.vllm_base_url}/models"

        resp = httpx.get(url, timeout=5.0)
        resp.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="llm_connectivity",
            passed=True,
            message=f"{settings.llm_backend} reachable at {url}",
            latency_ms=latency,
        )
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="llm_connectivity",
            passed=False,
            message=f"Cannot reach {settings.llm_backend}: {exc}",
            latency_ms=latency,
        )


def _check_embeddings() -> CheckResult:
    """Verify the embedding model can be loaded (cached after first load)."""
    t0 = time.monotonic()
    try:
        from core.llm_manager import get_embeddings
        emb = get_embeddings()
        _ = emb.embed_query("startup check")
        latency = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="embeddings",
            passed=True,
            message=f"Embedding model ready ({settings.embedding_model})",
            latency_ms=latency,
        )
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="embeddings",
            passed=False,
            message=f"Embedding model failed: {exc}",
            latency_ms=latency,
        )


def _check_vector_store() -> CheckResult:
    """Verify the ChromaDB path is accessible and writable."""
    t0 = time.monotonic()
    path = settings.vector_store_path
    try:
        path.mkdir(parents=True, exist_ok=True)
        # Test write
        probe = path / ".startup_probe"
        probe.write_text("ok")
        probe.unlink()
        latency = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="vector_store",
            passed=True,
            message=f"Vector store path writable ({path})",
            latency_ms=latency,
            details={"path": str(path)},
        )
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return CheckResult(
            name="vector_store",
            passed=False,
            message=f"Vector store path not writable: {exc}",
            latency_ms=latency,
        )


def _check_data_directories() -> CheckResult:
    """Verify all required data directories exist and are writable."""
    dirs = {
        "uploads":     settings.uploads_path,
        "logs":        settings.log_path.parent,
        "sessions":    settings.log_path.parent / "sessions",
    }
    failed = []
    for name, path in dirs.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".probe"
            probe.write_text("ok")
            probe.unlink()
        except Exception as exc:
            failed.append(f"{name}: {exc}")

    if failed:
        return CheckResult(
            name="data_directories",
            passed=False,
            message=f"Directory issues: {'; '.join(failed)}",
        )
    return CheckResult(
        name="data_directories",
        passed=True,
        message=f"{len(dirs)} directories writable",
    )


def _check_env_completeness() -> CheckResult:
    """Warn about settings that are at their default placeholder values."""
    warnings = []

    if settings.llm_backend == "vllm" and settings.vllm_api_key == "EMPTY":
        pass  # EMPTY is the correct vLLM value

    if settings.ollama_model == "llama3.1:8b" and settings.llm_backend == "ollama":
        pass  # valid default

    # Check .env exists
    if not Path(".env").exists():
        warnings.append("No .env file found — using defaults from .env.example")

    if warnings:
        return CheckResult(
            name="env_completeness",
            passed=True,   # warnings only, not failures
            message=f"{len(warnings)} advisory note(s): {'; '.join(warnings)}",
        )
    return CheckResult(
        name="env_completeness",
        passed=True,
        message="Configuration looks complete",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_startup_checks(
    check_llm: bool = True,
    check_embeddings: bool = True,
    check_storage: bool = True,
) -> ValidationReport:
    """
    Run all pre-flight checks and return a ValidationReport.

    Args:
        check_llm:        Ping the LLM server.
        check_embeddings: Load and test the embedding model.
        check_storage:    Verify vector store + data directories.

    Returns:
        ValidationReport with all check results.
    """
    report = ValidationReport()

    if check_storage:
        report.checks.append(_check_data_directories())
        report.checks.append(_check_vector_store())

    report.checks.append(_check_env_completeness())

    if check_llm:
        report.checks.append(_check_llm_connectivity())

    if check_embeddings:
        report.checks.append(_check_embeddings())

    report.print_summary()
    return report
