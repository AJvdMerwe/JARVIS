"""
api/server.py
──────────────
FastAPI REST + WebSocket server exposing all assistant capabilities
over HTTP/WS for web UI and external API consumers.

Endpoints:
  POST   /chat                – single query, JSON response
  WS     /ws/{session_id}     – streaming bidirectional chat
  POST   /documents/ingest    – upload + ingest a document
  GET    /documents           – list knowledge base
  DELETE /documents/{title}   – remove a document
  GET    /documents/search    – semantic search
  GET    /health              – liveness probe
  GET    /metrics             – cache stats + circuit state

Run with:
    uvicorn api.server:app --reload --port 8080
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    Depends, FastAPI, File, Form, HTTPException, Query,
    UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents import AgentResponse, Orchestrator
from api.auth import (
    TokenResponse, UserCreate, UserOut,
    authenticate_user, create_access_token, create_user,
    decode_token, get_user_by_id, init_db,
)
from api.rate_limiter import RateLimitMiddleware
from api.sse import router as sse_router
from config import settings
from core.cache import get_cache
from core.logging import setup_logging
from core.tracing import get_tracer

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)

# ── App setup ────────────────────────────────────────────────────────────────
setup_logging()

app = FastAPI(
    title="Virtual Personal Assistant API",
    description="Local AI assistant with Code, News, Search, and Document agents.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.include_router(sse_router)


# ── Startup validation ───────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    """Run pre-flight checks and initialise auth DB on server start."""
    init_db()   # create users.db + seed admin if empty
    from api.startup_validator import run_startup_checks
    loop = __import__("asyncio").get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: run_startup_checks(
            check_llm=False,          # skip in dev; enable in prod
            check_embeddings=False,   # skip in dev; enable in prod
            check_storage=True,
        ),
    )

# Mount UI static files (built separately)
_UI_DIR = Path(__file__).parent.parent / "ui"

@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
async def serve_ui():
    """Serve the SPA entry point."""
    from fastapi.responses import HTMLResponse
    html_path = _UI_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1><p>Make sure ui/index.html exists.</p>", status_code=404)

if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

# ── Session store (in-memory; swap for Redis in production) ──────────────────
_orchestrators: dict[str, Orchestrator] = {}


def _get_or_create_orchestrator(session_id: str) -> Orchestrator:
    if session_id not in _orchestrators:
        _orchestrators[session_id] = Orchestrator(session_id=session_id)
        logger.info("Created orchestrator for session '%s'.", session_id)
    return _orchestrators[session_id]


# ─────────────────────────────────────────────────────────────────────────────
#  Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:      str         = Field(..., min_length=1, max_length=4096)
    session_id: str         = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    intent:     Optional[str] = Field(None, description="Force agent: code|news|search|document")
    doc_title:  Optional[str] = Field(None, description="Restrict document search to this doc")


class ChatResponse(BaseModel):
    session_id:  str
    output:      str
    agent_name:  str
    references:  list[str]
    tool_calls:  list[dict]
    latency_ms:  float
    error:       Optional[str] = None


class DocumentInfo(BaseModel):
    doc_title: str
    doc_path:  str
    doctype:   str


class SearchResult(BaseModel):
    reference:  str
    text:       str
    score:      float
    page:       Optional[int]
    section:    str


class HealthResponse(BaseModel):
    status:     str
    backend:    str
    session_count: int
    kb_chunks:  int


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _response_to_dict(resp: AgentResponse, latency_ms: float) -> ChatResponse:
    return ChatResponse(
        session_id="",
        output=resp.output,
        agent_name=resp.agent_name,
        references=resp.references,
        tool_calls=[
            {"tool": t, "input": i[:200], "output": o[:400]}
            for t, i, o in resp.tool_calls
        ],
        latency_ms=round(latency_ms, 1),
        error=resp.error,
    )


from contextlib import asynccontextmanager
from typing import AsyncGenerator as _AsyncGen


@asynccontextmanager
async def _chat_trace(tracer, session_id: str, query: str) -> _AsyncGen:
    """Async context manager shim around the synchronous Tracer.trace()."""
    from core.tracing.tracer import Trace
    import time as _time
    t = Trace(
        trace_id=tracer._next_id(),
        session_id=session_id,
        query=query,
    )
    try:
        yield t
    except Exception as exc:
        t.error = str(exc)
        raise
    finally:
        tracer.store.record(t)


# ─────────────────────────────────────────────────────────────────────────────
#  REST endpoints
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
#  Authentication endpoints
# =============================================================================

def _get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(_bearer),
) -> UserOut:
    """FastAPI dependency — validate Bearer JWT and return the user."""
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(creds.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user_by_id(int(payload["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(data: UserCreate) -> TokenResponse:
    """Register a new user account and receive a JWT."""
    try:
        user = create_user(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    token = create_access_token(user)
    return TokenResponse(access_token=token, username=user.username, user_id=user.id)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(data: LoginRequest) -> TokenResponse:
    """Authenticate with username + password and receive a JWT."""
    user = authenticate_user(data.username, data.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(user)
    return TokenResponse(access_token=token, username=user.username, user_id=user.id)


@app.get("/auth/me", response_model=UserOut, tags=["Auth"])
async def me(current_user: UserOut = Depends(_get_current_user)) -> UserOut:
    """Return the currently authenticated user's profile."""
    return current_user


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe."""
    from document_processing import DocumentManager
    try:
        dm = DocumentManager()
        kb_chunks = dm.total_chunks
    except Exception:
        kb_chunks = -1

    return HealthResponse(
        status="ok",
        backend=settings.llm_backend,
        session_count=len(_orchestrators),
        kb_chunks=kb_chunks,
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Cache, rate limiter, and tracer stats."""
    from api.rate_limiter import get_rate_limiter
    return {
        "cache":        get_cache().stats(),
        "rate_limiter": get_rate_limiter().stats(),
        "traces":       get_tracer().stats(),
        "sessions":     len(_orchestrators),
    }


@app.get("/traces", tags=["System"])
async def traces(n: int = 20, session_id: Optional[str] = None):
    """Return recent request traces for observability."""
    tracer = get_tracer()
    if session_id:
        recent = tracer.store.for_session(session_id)[-n:]
    else:
        recent = tracer.store.recent(n)
    return [t.to_dict() for t in recent]


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Send a query to the assistant and receive a complete response.
    For streaming, use the /stream (SSE) or /ws (WebSocket) endpoints.
    """
    orch = _get_or_create_orchestrator(req.session_id)
    kwargs = {}
    if req.intent:
        kwargs["intent"] = req.intent
    if req.doc_title:
        kwargs["doc_title"] = req.doc_title

    tracer = get_tracer()
    t0 = time.monotonic()

    async with _chat_trace(tracer, req.session_id, req.query) as trace:
        try:
            loop = asyncio.get_event_loop()
            with trace.span("routing"):
                detected = orch.route_only(req.query)

            with trace.span("agent_run", agent=detected.value):
                response = await loop.run_in_executor(
                    None, lambda: orch.run(req.query, **kwargs)
                )

            trace.set_outcome(response.output, error=response.error, agent_name=response.agent_name)
        except Exception as exc:
            logger.error("Chat error for session '%s': %s", req.session_id, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    latency = (time.monotonic() - t0) * 1000
    result = _response_to_dict(response, latency)
    result.session_id = req.session_id
    return result


@app.delete("/chat/{session_id}", tags=["Chat"])
async def clear_memory(session_id: str):
    """Clear conversation memory for a session."""
    if session_id in _orchestrators:
        _orchestrators[session_id].clear_memory()
        return {"message": f"Memory cleared for session '{session_id}'."}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")


# ── Document endpoints ───────────────────────────────────────────────────────

@app.post("/documents/ingest", tags=["Documents"])
async def ingest_document(
    file: UploadFile = File(...),
    session_id: str = Form(default="api"),
):
    """Upload and ingest a document (PDF, DOCX, XLSX, PPTX)."""
    from document_processing import SUPPORTED_SUFFIXES

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_SUFFIXES)}",
        )

    # Save upload to the uploads directory
    dest = settings.uploads_path / file.filename
    content = await file.read()
    dest.write_bytes(content)
    logger.info("Uploaded '%s' (%d bytes).", file.filename, len(content))

    # Ingest in a thread (CPU-bound)
    orch = _get_or_create_orchestrator(session_id)
    loop = asyncio.get_event_loop()
    try:
        msg = await loop.run_in_executor(None, lambda: orch.ingest_document(str(dest)))
        return {"message": msg, "filename": file.filename, "path": str(dest)}
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all documents in the knowledge base."""
    from document_processing import DocumentManager
    dm = DocumentManager()
    docs = dm.list_documents()
    return [
        DocumentInfo(
            doc_title=d["doc_title"],
            doc_path=d.get("doc_path", ""),
            doctype=d.get("doctype", ""),
        )
        for d in docs
    ]




@app.post("/documents/bulk-ingest")
async def bulk_ingest(
    directory: Optional[str] = None,
    paths: Optional[str] = None,
    recursive: bool = True,
    dry_run: bool = False,
):
    """
    Bulk-ingest documents into the knowledge base.

    Pass either:
      - ``directory``: path to a folder (all supported files ingested)
      - ``paths``: comma-separated list of file paths

    Set ``dry_run=true`` to inspect without writing.
    """
    from document_processing.mass_uploader import MassUploader
    uploader = MassUploader()
    try:
        if directory:
            report = uploader.upload_directory(
                directory, recursive=recursive, dry_run=dry_run
            )
        elif paths:
            path_list = [p.strip() for p in paths.split(",") if p.strip()]
            report = uploader.upload_files(path_list, dry_run=dry_run)
        else:
            return {"error": "Provide 'directory' or 'paths'"}

        return {
            "ok": report.ok_count,
            "duplicate": report.duplicate_count,
            "error": report.error_count,
            "unsupported": report.unsupported_count,
            "chunks_added": report.total_chunks_added,
            "elapsed_ms": report.total_elapsed_ms,
            "dry_run": report.is_dry_run,
            "outcomes": [
                {
                    "file": o.filename,
                    "status": o.status,
                    "doc_type": o.doc_type.value,
                    "strategy": o.strategy.value,
                    "chunks_added": o.chunks_added,
                    "error": o.error or None,
                }
                for o in report.outcomes
            ],
        }
    except NotADirectoryError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"Bulk ingest failed: {exc}"}

@app.delete("/documents/{doc_title}", tags=["Documents"])
async def delete_document(doc_title: str):
    """Remove a document from the knowledge base."""
    from document_processing import DocumentManager
    dm = DocumentManager()
    removed = dm.delete_document(doc_title)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{doc_title}' not found.")
    return {"message": f"Removed '{doc_title}' ({removed} chunks deleted)."}


@app.get("/documents/search", response_model=list[SearchResult], tags=["Documents"])
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    k: int = Query(default=5, ge=1, le=20),
    doc_title: Optional[str] = Query(None),
):
    """Semantic search across the knowledge base."""
    from document_processing import DocumentManager
    dm = DocumentManager()
    results = dm.search(q, k=k, doc_title=doc_title)
    return [
        SearchResult(
            reference=r.reference,
            text=r.chunk.text,
            score=round(r.score, 4),
            page=r.chunk.page_number,
            section=" › ".join(r.chunk.section_path),
        )
        for r in results
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  WebSocket streaming endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    Bidirectional WebSocket chat with token-level streaming.

    Client → Server  (JSON):
        {"query": "...", "intent": "code|news|search|document" (optional)}

    Server → Client  (JSON, one message per event):
        {"type": "start",     "agent": "code_agent"}
        {"type": "token",     "content": "Here "}
        {"type": "token",     "content": "is "}
        {"type": "reference", "ref": "Doc › Page 3 › Section"}
        {"type": "done",      "latency_ms": 1234}
        {"type": "error",     "message": "..."}
    """
    await websocket.accept()
    orch = _get_or_create_orchestrator(session_id)
    logger.info("WebSocket connected: session='%s'", session_id)

    try:
        while True:
            raw = await websocket.receive_json()
            query   = raw.get("query", "").strip()
            intent  = raw.get("intent")

            if not query:
                await websocket.send_json({"type": "error", "message": "Empty query."})
                continue

            kwargs = {}
            if intent:
                kwargs["intent"] = intent

            # Detect which agent will handle this (no LLM call yet)
            detected = orch.route_only(query)
            await websocket.send_json({"type": "start", "agent": detected.value})

            t0 = time.monotonic()
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: orch.run(query, **kwargs)
                )
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
                continue

            latency = (time.monotonic() - t0) * 1000

            # Stream the output word-by-word to simulate token streaming
            # (Real token streaming requires LangChain streaming callbacks)
            words = response.output.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                await websocket.send_json({"type": "token", "content": chunk})
                await asyncio.sleep(0.005)  # pacing

            # Send references
            for ref in response.references:
                await websocket.send_json({"type": "reference", "ref": ref})

            await websocket.send_json({
                "type": "done",
                "agent": response.agent_name,
                "latency_ms": round(latency, 1),
                "tool_call_count": len(response.tool_calls),
            })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session='%s'", session_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
