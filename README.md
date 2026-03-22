# Virtual Personal Assistant

A fully local, privacy-first AI assistant with deep multi-agent capabilities. No cloud APIs. No data leaving your machine.

```
╔═══════════════════════════════════════════════════════════════╗
║         Virtual Personal Assistant  v1.0                      ║
║  Chat · Code · News · Search · Documents · Finance · Voice    ║
╚═══════════════════════════════════════════════════════════════╝

You> Hello! Can you help me brainstorm a startup idea?
[chat_agent] Sure! Let's start with your interests…

You> Write a binary search in Python and test it
[code_agent] Here's a clean implementation with tests…

You> What are today's AI headlines?
[news_agent] ## Latest Headlines…

You> Summarise the Q3 report I uploaded
[document_agent] Based on Q3_Report.pdf › Page 4 › Revenue…

You> What is Apple's current stock price?
[financial_agent] Apple Inc (AAPL): $178.50  ▲ $3.30 (1.89%)…
```

---

## Quick Start

```bash
git clone https://github.com/yourname/virtual-assistant.git
cd virtual-assistant
bash start.sh
```

`start.sh` handles everything on first run: creates a virtualenv, installs all Python dependencies, copies `.env`, creates data directories, starts Ollama, and pulls the configured model. Subsequent runs detect completed steps and skip them.

```bash
bash start.sh --help                                   # all options
bash start.sh voice                                    # mic + TTS
bash start.sh api                                      # FastAPI server on :8080
bash start.sh docker                                   # full Docker Compose stack
bash start.sh test                                     # run the 693-test suite
bash start.sh --query "Explain recursion like I'm five"
bash start.sh --ingest ./reports/
bash start.sh --model llama3.2:3b                      # lighter model
bash start.sh --backend vllm api                       # GPU + API server
```

---

## Features

| Capability | Detail |
|---|---|
| **6 specialist agents** | Chat · Code · News · Search · Document · Finance — auto-routed by intent |
| **Fallback loop** | If the primary agent fails, the orchestrator tries fallback agents then synthesises |
| **RAG pre-check** | Vector-store scan after intent detection; serves KB answers instantly, skipping agents |
| **Direct LLM chat** | ChatAgent talks straight to the model with full conversation history |
| **Document Q&A** | Ingest PDF, DOCX, XLSX, PPTX, TXT, MD, CSV, HTML, JSON, XML |
| **Mass document upload** | Concurrent batch ingestion with type detection, deduplication, and dry-run |
| **Financial analysis** | Stock quotes, ratios, statements, price history, comparisons via Yahoo Finance |
| **Voice I/O** | Whisper STT (offline, 99 languages) + pyttsx3 TTS |
| **Dual LLM backend** | Ollama (CPU) or vLLM (GPU) — swap with one env var |
| **FastAPI server** | REST + WebSocket streaming + Server-Sent Events |
| **Dark-mode Web UI** | Streaming chat, document upload sidebar, agent selector, reference display |
| **Resilience** | Circuit breaker · retry with exponential backoff · automatic LLM failover |
| **Tool cache** | TTL cache with `@cached_tool` decorator, disk persistence |
| **Long-term memory** | Episodic store — recalls relevant facts from past sessions |
| **Conversation summariser** | Auto-compresses old turns so context never overflows |
| **Async fan-out** | Run multiple agents in parallel with `asyncio.gather` |
| **User preferences** | Per-user style, language, timezone, news topics — persisted |
| **Plugin system** | Drop a `.py` in `plugins/` to add agents and tools without touching core |
| **Evaluation harness** | 10 criteria types, 8 built-in suites, LLM judge, JSON reports |
| **Admin CLI** | Inspect sessions, traces, KB, cache, scheduler from the terminal |
| **Structured logging** | JSON log lines with session tags and per-request span tracing |
| **Scheduled tasks** | Background cron jobs: cache purge, KB re-index, news digest |
| **Rate limiting** | Sliding-window per-IP limiter with `X-RateLimit-*` headers |
| **Startup validator** | Pre-flight checks (dirs, vector store, LLM, env) before serving |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            Interfaces                                    │
│   CLI (Typer)  ·  Interactive REPL  ·  Voice loop  ·  Web UI             │
│   FastAPI REST ·  WebSocket /ws     ·  SSE /stream  ·  OpenAPI /docs     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                           Orchestrator                                   │
│                                                                          │
│  Intent Router (2-stage)                                                 │
│    Stage 1: keyword regex + phrase patterns    < 1ms, no LLM call        │
│    Stage 2: LLM classification                 ambiguous queries only    │
│                                                                          │
│  ┌──────┬──────┬──────┬──────────┬──────────┬──────────┐                 │
│  │ Chat │ Code │ News │  Search  │ Document │ Finance  │                 │
│  │Agent │Agent │Agent │  Agent   │  Agent   │  Agent   │                 │
│  └──────┴──────┴──────┴──────────┴──────────┴──────────┘                 │
│                                                                          │
│  RAG pre-check → quality gate → fallback chain → synthesis               │
│  Post-processing: memory · episodic · summariser · tracing               │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                             Core Layer                                   │
│                                                                          │
│  llm_manager       Ollama / vLLM factory · @lru_cache singleton          │
│  resilience        ResilientLLM · CircuitBreaker · retry + failover      │ 
│  memory            ConversationMemory · PersistentMemory (JSON)          │
│  summariser        ConversationSummariser (rolling LLM compression)      │
│  long_term_memory  EpisodicMemory (ChromaDB, cross-session recall)       │
│  async_runner      AsyncAgentRunner · fan-out · streaming callbacks      │
│  cache             ToolCache (TTL, LRU, disk) · @cached_tool             │
│  tracing           Tracer · Span · TraceStore (JSONL sink)               │
│  user_prefs        UserPreferences (Pydantic, per-user JSON)             │
│  scheduler         TaskScheduler (daemon thread, 4 built-in tasks)       │
│  logging           JsonFormatter · AssistantLogger · agent_call()        │
│  voice             Whisper STT · MicrophoneListener VAD · pyttsx3 TTS    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                       Document Processing                                │
│                                                                          │
│  TypeDetector      12 types · 17 extensions · magic-byte sniffing        │
│  DoclingProcessor  PDF/DOCX/XLSX/PPTX → DocumentChunk with references    │
│  MassUploader      concurrent batch · dedup · dry-run · progress hooks   │
│  DocumentManager   search · stats · delete · LangChain retriever         │
│  VectorStore       ChromaDB · cosine similarity · idempotent ingest      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Agents

### Intent Routing

Two-stage pipeline — Stage 1 is regex (< 1 ms, no LLM call); Stage 2 calls the LLM only when Stage 1 is ambiguous:

```
"Hello, how are you?"             →  CHAT     (greeting keyword)
"Write a haiku about autumn"      →  CHAT     (creative "write a haiku")
"Write a quicksort in Python"     →  CODE     (code noun after "write")
"Debug this TypeError"            →  CODE     (debug keyword)
"What are today's headlines?"    →  NEWS     (headlines keyword)
"Summarise the uploaded PDF"      →  DOCUMENT (document keyword)
"What is Apple's stock price?"   →  FINANCE  (stock price keyword)
"AAPL P/E ratio"                  →  FINANCE  (ratio keyword)
"What is the capital of France?"  →  SEARCH   (no other signals)
"Anything ambiguous…"             →  LLM classifies → defaults to CHAT
```

### Chat Agent

Talks directly to the LLM — no external tools. Chosen for greetings, explanations, creative writing, brainstorming, opinions, follow-up questions, and anything conversational. Full sliding-window history is injected so context persists across many turns. Supports `stream()` for live-typing REPL output.

### Code Agent

Five task types detected automatically:

| Task | Trigger example |
|---|---|
| `write` | "Implement a REST API in TypeScript" |
| `write_and_verify` | "Write a quicksort in Python and test it" |
| `review` | "Refactor this function" + pasted code |
| `debug` | "Why does this raise TypeError?" |
| `execute` | "Run this" + pasted Python snippet |

For `write_and_verify`, code is generated, executed in a subprocess sandbox, and automatically fixed by the reviewer if execution fails — then re-run.

### Document Agent

Seven tools available:

| Tool | Description |
|---|---|
| `ingest_document` | Parse and index a single file |
| `bulk_ingest_directory` | Ingest an entire folder (all supported types) |
| `bulk_ingest_files` | Ingest a comma-separated list of files |
| `search_documents` | Semantic search with relevance scores |
| `list_documents` | List all indexed documents |
| `get_full_document` | Retrieve a full document's text |
| `delete_document` | Remove a document from the KB |

Every answer cites chunks by page and section: `Q3_Report › Page 4 › Revenue`.

### Financial Agent

Eight task types with direct tool routing (no ReAct loop for known patterns):

| Task | Tools invoked |
|---|---|
| `quote` | `stock_quote` |
| `profile` | `company_info` |
| `ratios` | `financial_ratios` |
| `history` | `price_history` |
| `statements` | `financial_statements` |
| `compare` | `stock_comparison` |
| `analysis` | `stock_quote` + `company_info` + `financial_ratios` + `price_history` |
| `unknown` | ReAct executor (all 6 tools) |

Company names resolve to tickers automatically: `"Apple"` → `AAPL`, `"Microsoft"` → `MSFT`, `"Bitcoin"` → `BTC-USD` (70+ entries in the lookup map). Data comes from Yahoo Finance via yfinance; the LLM's training knowledge is used as a fallback when the network is restricted.

---

## RAG Pre-check

After intent is detected and before any agent is invoked, the orchestrator runs a
lightweight vector-store scan to check whether the knowledge base already contains
a sufficient answer. When it does, the answer is returned immediately — no agent
is ever called.

```
intent detected
      │
      ▼
RAG pre-check  ──► search KB (< 200 ms)
      │
      ├── relevant chunks found + LLM answers from context
      │       └── SUFFICIENT ──────────────────────────► return KB answer
      │
      └── no relevant chunks  /  LLM says INSUFFICIENT
              └────────────────────────────────────────► primary agent
```

### How it works

1. **Skip check** — CODE and CHAT intents are bypassed immediately (not retrieval tasks).
   Empty knowledge bases are also skipped with zero latency.
2. **Similarity search** — `DocumentManager.search()` is called with the user query.
   Only chunks above `rag_similarity_threshold` (default `0.55`) are kept.
3. **Strict LLM prompt** — the model is given only the retrieved excerpts and asked to
   answer or return the single word `INSUFFICIENT`.
4. **Quality gate** — the same `_is_sufficient_response()` check used by the fallback
   loop validates the answer before returning it.
5. **Fall-through** — if any step fails (empty KB, low scores, INSUFFICIENT verdict,
   LLM error), execution continues to the normal agent routing without interruption.

### Response shape

A RAG pre-check answer is a standard `AgentResponse` with:

```python
AgentResponse(
    output    = "Q3 revenue was $4.2B according to Q3 Report…",
    agent_name= "rag_precheck",
    references= ["Q3 Report › Page 4 › Revenue Summary"],
    metadata  = {
        "rag_chunks":    3,      # number of chunks retrieved
        "rag_top_score": 0.87,   # highest cosine similarity
        "rag_elapsed_ms": 143.2, # total pre-check time
    },
)
```

### Configuration

```python
from agents.orchestrator import Orchestrator

# Default: pre-check enabled, threshold 0.55, retrieve 4 chunks
orch = Orchestrator(
    enable_rag_precheck      = True,
    rag_similarity_threshold = 0.55,   # 0.65+ = more selective
    rag_k                    = 4,      # chunks to retrieve
)

# Disable for deterministic / unit-test pipelines
orch = Orchestrator(enable_rag_precheck=False)

# More aggressive retrieval for deep knowledge bases
orch = Orchestrator(rag_similarity_threshold=0.45, rag_k=8)
```

### Intents that participate

| Intent | Pre-check runs? | Reason |
|---|---|---|
| `DOCUMENT` | ✓ | Core retrieval use-case |
| `SEARCH` | ✓ | KB may hold the factual answer |
| `NEWS` | ✓ | Ingested news articles searched first |
| `FINANCE` | ✓ | Financial reports in KB checked first |
| `UNKNOWN` | ✓ | Best-effort KB scan |
| `CHAT` | ✗ | Conversational, not retrieval |
| `CODE` | ✗ | Code generation is not a lookup task |

## Orchestrator — Fallback Loop

When the primary agent does not produce a satisfactory answer, the orchestrator automatically iterates through a per-intent fallback chain until a sufficient response is found or all options are exhausted.

### Quality gates

After every agent call the response is evaluated against fast-path gates (no LLM call, no latency penalty):

| Signal | Fail condition |
|---|---|
| Error flag | `response.error` is set |
| Too short | Output is fewer than 40 characters |
| Failure phrase | Output contains "I don't know", "cannot find", "no results found", "failed to retrieve", or 16 other phrases |

When `enable_llm_quality_check=True` the orchestrator also asks the LLM `"SUFFICIENT or INSUFFICIENT?"` after each agent call. This is more accurate but incurs a latency and token cost — off by default.

### Fallback chains

| Primary intent | Fallback order |
|---|---|
| `DOCUMENT` | SEARCH → CHAT |
| `SEARCH` | DOCUMENT → CHAT |
| `NEWS` | SEARCH → CHAT |
| `FINANCE` | SEARCH → CHAT |
| `CODE` | SEARCH → CHAT |
| `CHAT` | SEARCH |
| `UNKNOWN` | CHAT |

### Execution flow

```
query
  │
  ▼
Primary agent  (intent-routed)
  │
  ├── SUFFICIENT ──────────────────────────────────► post-process & return
  │
  └── INSUFFICIENT
        │
        ├── Fallback agent 1 ──► quality check
        │       ├── SUFFICIENT ──────────────────► post-process & return
        │       └── INSUFFICIENT
        │
        ├── Fallback agent 2 ──► quality check
        │       ├── SUFFICIENT ──────────────────► post-process & return
        │       └── INSUFFICIENT
        │
        └── Synthesis (LLM merges all evidence) ► post-process & return
```

### Synthesis

When every agent in the chain fails, `_synthesise_from_attempts()` combines all non-empty partial outputs, tool calls, and references into a single prompt and asks the LLM to produce the best possible answer from the available evidence. The synthesised response carries `agent_name="synthesised"`.

### Configuration

```python
from agents.orchestrator import Orchestrator

# Default: up to 2 fallback agents, fast quality gates only
orch = Orchestrator(
    session_id               = "alice",
    max_fallback_attempts    = 2,      # cap retries (0 = disable fallback)
    enable_llm_quality_check = False,  # opt-in LLM judge (slower, more accurate)
)

# Disable fallback entirely for deterministic pipelines
orch = Orchestrator(max_fallback_attempts=0)

# Maximum accuracy: LLM judge + 3 fallback attempts
orch = Orchestrator(enable_llm_quality_check=True, max_fallback_attempts=3)
```

---

## Mass Document Upload

Batch-ingest any mix of document types in a single call:

```bash
# Directory
bash start.sh --ingest ./reports/

# CLI
python cli.py ingest ./documents/ --directory

# Admin
python admin/admin_cli.py kb bulk ./documents/ --workers 8

# API
curl -X POST "http://localhost:8080/documents/bulk-ingest?directory=./reports/"

# Agent (REPL or API)
:agent document
> Index all files in the ./quarterly_reports/ directory
```

### Type detection

Every file is classified before parsing. Two-stage detection: extension-first (fast), then magic-byte content sniffing for files with wrong or missing extensions.

| Format | Extensions | Strategy |
|---|---|---|
| PDF | `.pdf` | Docling (OCR, tables, heading detection) |
| Word | `.docx` · `.doc` | Docling → python-docx fallback |
| Excel | `.xlsx` · `.xls` | Docling → openpyxl fallback |
| PowerPoint | `.pptx` · `.ppt` | Docling → python-pptx fallback |
| Plain text | `.txt` · `.md` | Text splitter |
| Tabular | `.csv` · `.tsv` | Text splitter (column headers preserved) |
| Web | `.html` · `.htm` | HTML tag stripper + text splitter |
| Structured | `.json` · `.jsonl` · `.xml` | JSON pretty-printer / XML reader |

### Deduplication

Content-hash (SHA-256) deduplication prevents re-ingesting unchanged files, even when the filename changes. Pre-populate `dedup_hashes` to skip files already known from a previous session.

### Dry run

```bash
python admin/admin_cli.py kb bulk ./documents/ --dry-run
```

Scans, detects types, and prints a full plan — nothing is written to the store.

### UploadReport

Every batch returns a structured `UploadReport`:

```python
from document_processing import MassUploader

uploader = MassUploader(max_workers=4)
report   = uploader.upload_directory("./reports/")
print(report.summary())

# report.ok_count          — files successfully ingested
# report.duplicate_count   — files skipped (same content hash)
# report.error_count       — files that failed
# report.total_chunks_added
# report.total_elapsed_ms
# report.outcomes          — list[FileOutcome] with per-file detail
```

---

## CLI Reference

### `start.sh`

```bash
bash start.sh [MODE] [OPTIONS]

Modes
  chat      Interactive text REPL (default)
  voice     REPL with Whisper STT + TTS
  api       FastAPI server on :8080
  docker    Full Docker Compose stack
  test      Run the 693-test suite

Options
  --query  TEXT   Single query, print response, exit
  --ingest PATH   Ingest document or directory, then start REPL
  --backend NAME  ollama | vllm
  --model  TAG    Ollama model (e.g. llama3.2:3b)
  --port   INT    API port (default: 8080)
  --fresh         Force full re-install of dependencies
  --help          Show all options
```

### `cli.py`

```bash
python cli.py chat
python cli.py chat --voice
python cli.py ask "Explain SOLID"
python cli.py ask "Write a sort"           --agent code
python cli.py ask "Hello!"                 --agent chat
python cli.py ask "Apple stock P/E ratio"  --agent finance

python cli.py ingest report.pdf
python cli.py ingest ./docs/ --directory

python cli.py docs list
python cli.py docs search "revenue Q3"
python cli.py docs search "costs" --doc "Q3 Report"
python cli.py docs delete "Q3 Report"

python cli.py transcribe meeting.wav
python cli.py config
```

### REPL commands

| Command | Description |
|---|---|
| `:help` | Show all commands |
| `:quit` / `:exit` | Exit |
| `:clear` | Clear conversation memory |
| `:docs` | List ingested documents |
| `:ingest <path>` | Ingest a document |
| `:delete <title>` | Remove a document |
| `:agent <n>` | Force next query to: `chat` `code` `news` `search` `document` `finance` |
| `:history` | Show recent conversation |
| `:voice` | Toggle voice I/O |

### `admin/admin_cli.py`

```bash
python admin/admin_cli.py sessions list
python admin/admin_cli.py sessions clear <id>

python admin/admin_cli.py kb stats
python admin/admin_cli.py kb list
python admin/admin_cli.py kb ingest ./reports/
python admin/admin_cli.py kb bulk ./documents/ --workers 8 --dry-run
python admin/admin_cli.py kb delete "Q3 Report"
python admin/admin_cli.py kb vacuum

python admin/admin_cli.py traces list --n 20
python admin/admin_cli.py traces stats
python admin/admin_cli.py traces export traces.jsonl

python admin/admin_cli.py cache stats
python admin/admin_cli.py cache clear

python admin/admin_cli.py prefs show alice
python admin/admin_cli.py prefs set alice response_style technical
python admin/admin_cli.py scheduler list
python admin/admin_cli.py scheduler run cache_purge
python admin/admin_cli.py health
```

---

## API Reference

Start with `bash start.sh api` or `uvicorn api.server:app --port 8080`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Single query → full JSON response |
| `DELETE` | `/chat/{session_id}` | Clear session memory |
| `GET` | `/stream` | SSE token-by-token streaming |
| `WS` | `/ws/{session_id}` | WebSocket bidirectional streaming |
| `POST` | `/documents/ingest` | Upload + ingest a single document |
| `POST` | `/documents/bulk-ingest` | Batch-ingest a directory or file list |
| `GET` | `/documents` | List knowledge base |
| `DELETE` | `/documents/{title}` | Remove a document |
| `GET` | `/documents/search?q=...` | Semantic search |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Cache, rate limiter, tracer stats |
| `GET` | `/traces` | Recent request traces |
| `GET` | `/docs` | Swagger UI |

### Bulk ingest

```bash
# Ingest a directory
curl -X POST "http://localhost:8080/documents/bulk-ingest?directory=./reports/"

# Ingest specific files
curl -X POST "http://localhost:8080/documents/bulk-ingest?paths=report.pdf,data.xlsx"

# Dry run
curl -X POST "http://localhost:8080/documents/bulk-ingest?directory=./docs&dry_run=true"
```

Response:

```json
{
  "ok": 12,
  "duplicate": 3,
  "error": 0,
  "unsupported": 1,
  "chunks_added": 847,
  "elapsed_ms": 14230,
  "dry_run": false,
  "outcomes": [
    {
      "file": "Q3.pdf",
      "status": "ok",
      "doc_type": "pdf",
      "strategy": "docling",
      "chunks_added": 64,
      "error": null
    }
  ]
}
```

### WebSocket event types

```json
{"type": "start",     "agent": "financial_agent"}
{"type": "token",     "content": "Apple Inc "}
{"type": "reference", "ref": "Q3_Report › Page 4 › Revenue"}
{"type": "done",      "latency_ms": 823, "tool_call_count": 3}
{"type": "error",     "message": "..."}
```

---

## Configuration

`.env` is created automatically from `.env.example` on first run.

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `vllm` |
| `OLLAMA_MODEL` | `llama3.1:8b` | Chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `VOICE_ENABLED` | `false` | Enable mic + TTS |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `VOICE_LANGUAGE` | `en` | ISO 639-1 language code |
| `EMBEDDING_BACKEND` | `huggingface` | `huggingface` \| `ollama` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model (backend=huggingface) |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` \| `cuda` \| `mps` \| `auto` (HF backend only) |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model (backend=ollama) |
| `EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding call (lower = less VRAM) |
| `CHUNK_SIZE` | `512` | Document chunk size (chars) |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `VECTOR_STORE_PATH` | `./data/vector_store` | ChromaDB path |
| `AGENT_MAX_ITERATIONS` | `15` | Max ReAct loop steps |
| `AGENT_VERBOSE` | `false` | Print tool reasoning to stdout |
| `RATE_LIMIT_REQUESTS` | `60` | API requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Window in seconds |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Recommended models

| RAM / VRAM | Model |
|---|---|
| 4 GB | `llama3.2:3b` · `phi3:mini` |
| 8 GB | `llama3.1:8b` · `mistral:7b` ← **default** |
| 16 GB | `llama3.1:70b-q4` · `qwen2.5:14b` |
| GPU 16 GB+ | vLLM: `mistralai/Mistral-7B-Instruct-v0.2` |

---

## Embedding Backends

The assistant supports two embedding backends, selectable via `EMBEDDING_BACKEND` in `.env`.

### HuggingFace (default)

Loads a sentence-transformers model in-process. Fast and self-contained, but the model shares GPU memory with Docling and the LLM.

```bash
EMBEDDING_BACKEND=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu    # safe default — no GPU competition
# EMBEDDING_DEVICE=cuda # only if you have headroom after Docling + LLM
```

### Ollama (recommended for low-VRAM GPUs)

Sends embedding requests to the Ollama HTTP server. The model runs inside the Ollama process — **zero application VRAM**. This is the correct fix for `torch.OutOfMemoryError` during document ingestion.

```bash
EMBEDDING_BACKEND=ollama
OLLAMA_EMBEDDING_MODEL=nomic-embed-text   # pull first: ollama pull nomic-embed-text
EMBEDDING_BATCH_SIZE=16                   # conservative for 6 GB GPU
```

Supported Ollama embedding models:

| Model | Params | Dims | Notes |
|---|---|---|---|
| `nomic-embed-text` | 137 M | 768 | Fast, good default |
| `nomic-embed-text-v2-moe` | MoE | 768 | Higher quality, same speed |
| `mxbai-embed-large` | 335 M | 1024 | Best quality |
| `all-minilm` | 23 M | 384 | Smallest, fastest |

Pull a model before switching:

```bash
ollama pull nomic-embed-text
ollama pull nomic-embed-text-v2-moe
```

### Switching backends at runtime

```python
from core.llm_manager import clear_embeddings_cache
# After changing settings:
clear_embeddings_cache()
# Next call to get_embeddings() rebuilds from current settings
```

### Batch size tuning

`EMBEDDING_BATCH_SIZE` controls how many texts are embedded per call. Lower values reduce peak memory at the cost of throughput:

| GPU VRAM | Recommended batch size |
|---|---|
| ≥ 16 GB | 64–128 |
| 8–16 GB | 32 (default) |
| 4–8 GB | 8–16 |
| < 4 GB | 1–4 (use Ollama backend instead) |

## Adding a New Agent

```python
# agents/my_agent.py
from agents.base_agent import AgentResponse, BaseAgent
from tools.my_tools import get_my_tools

class MyAgent(BaseAgent):
    @property
    def name(self): return "my_agent"

    @property
    def description(self): return "Does X. Use for: A, B, C."

    def get_tools(self): return get_my_tools()

    def run(self, query, **kwargs):
        executor = self._build_react_agent()
        result   = executor.invoke({"input": query})
        return AgentResponse(
            output=result["output"],
            agent_name=self.name,
            tool_calls=self._extract_tool_calls(result.get("intermediate_steps", [])),
        )
```

Register in the orchestrator:

```python
orch.add_agent(Intent.SEARCH, MyAgent(memory=orch.memory))
```

Or use the plugin system — no core changes needed:

```python
# plugins/my_plugin.py
def register_agents(): return {"my_agent": MyAgent}
def register_tools():  return [MyTool()]
```

---

## Evaluation

```bash
python evaluation/run_evals.py --suite smoke      # quick: one case per agent
python evaluation/run_evals.py --suite financial  # financial agent quality
python evaluation/run_evals.py --suite chat       # conversational quality
python evaluation/run_evals.py --suite all --save
python evaluation/run_evals.py --list
```

Built-in suites: `smoke` · `chat` · `code` · `news` · `search` · `document` · `financial` · `routing`

```python
from evaluation import Contains, NoError, UsedTools, AgentIs, EvalCase, EvalSuite, Evaluator

suite = EvalSuite("my_suite")
suite.add(EvalCase(
    query="Write fibonacci in Python",
    criteria=[AgentIs("code_agent"), Contains("def fibonacci"), NoError()],
    agent_hint="code",   # chat | code | news | search | document | finance
))
Evaluator().run(suite).print_summary()
```

---

## Docker

```bash
docker build -t virtual-assistant .

# CLI
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  virtual-assistant

# API server
docker run -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  virtual-assistant \
  sh -c "uvicorn api.server:app --host 0.0.0.0 --port 8080"

# Full stack
docker compose up -d ollama
docker compose run --rm assistant bash scripts/setup_ollama.sh
docker compose run -it --rm assistant
```

---

## Development

```bash
make install-dev    # venv + all deps + dev extras
make test           # 693 tests
make test-cov       # with HTML coverage → data/coverage/
make test-unit      # unit tests only
make test-api       # API tests only
make lint           # ruff check
make format         # black + ruff --fix
make typecheck      # mypy
make eval-smoke     # quick quality check
make eval-all       # full evaluation + JSON report
make run            # start REPL
make run-api        # FastAPI with auto-reload
```

---

## Project Structure

```
virtual-assistant/
│
├── start.sh                            ← single entry point (setup + all modes)
├── .env                                ← your configuration (git-ignored)
├── .env.example                        ← template with all settings documented
│
├── agents/
│   ├── base_agent.py                   abstract BaseAgent + AgentResponse
│   ├── chat_agent.py                   direct LLM chat, history injection, streaming
│   ├── code_agent.py                   5-task router: write/verify/review/debug/execute
│   ├── news_agent.py                   RSS + DuckDuckGo news aggregation
│   ├── search_agent.py                 4-route: math/URL/encyclopedic/multi-source
│   ├── document_agent.py               7 tools: ingest, bulk ingest, search, manage
│   ├── financial_agent.py              8-task router, company-name resolution, yfinance
│   ├── rag_precheck.py                 KB scan before agents; returns RAG answer or None
│   └── orchestrator.py                 Intent enum · 2-stage routing · fallback loop
│                                       RAG pre-check · quality gates · synthesis
│                                       fallback chains · LangGraph graph
│
├── core/
│   ├── llm_manager.py                  Ollama / vLLM factory + @lru_cache singleton
│   ├── async_runner/                   AsyncAgentRunner · fan-out · streaming
│   ├── cache/                          ToolCache (TTL, LRU, disk) · @cached_tool
│   ├── logging/                        JsonFormatter · AssistantLogger
│   ├── long_term_memory/               EpisodicMemory (ChromaDB, cross-session)
│   ├── memory/                         ConversationMemory · PersistentMemory
│   ├── resilience/                     ResilientLLM · CircuitBreaker · retry
│   ├── scheduler/                      TaskScheduler + 4 built-in tasks
│   ├── summariser/                     ConversationSummariser (rolling compression)
│   ├── tracing/                        Tracer · Span · TraceStore (JSONL)
│   ├── user_prefs/                     UserPreferences (Pydantic, disk-persisted)
│   └── voice/                          Whisper STT · MicrophoneListener · pyttsx3 TTS
│
├── document_processing/
│   ├── __init__.py                     exports all public symbols incl. IngestResult
│   ├── type_detector.py                12 types · 17 extensions · magic-byte sniffing
│   ├── docling_processor.py            Docling → DocumentChunk with breadcrumb refs
│   ├── document_manager.py             IngestResult · KnowledgeBaseStats
│   │                                   ingest/search/list/delete · LangChain retriever
│   ├── mass_uploader.py                MassUploader · FileOutcome · UploadReport
│   │                                   concurrent · dedup · dry-run · progress hooks
│   └── vector_store.py                 ChromaDB · cosine similarity · idempotent ingest
│
├── tools/
│   ├── code_tools.py                   CodeWriter · CodeReviewer · Executor
│   ├── document_tools.py               Ingest · BulkIngestDirectory · BulkIngestFiles
│   │                                   Search · List · Get · Delete
│   ├── financial_tools.py              StockQuote · CompanyInfo · Statements · Ratios
│   │                                   PriceHistory · StockComparison
│   │                                   yfinance primary · LLM knowledge fallback
│   ├── news_tools.py                   Headlines · TopicNews · RSSFeed
│   └── search_tools.py                 DuckDuckGo · Wikipedia · WebFetch · Calculator
│
├── api/
│   ├── server.py                       FastAPI app · all endpoints · bulk-ingest route
│   ├── sse.py                          GET /stream (Server-Sent Events)
│   ├── rate_limiter.py                 RateLimiter · RateLimitMiddleware
│   └── startup_validator.py            pre-flight checks
│
├── ui/
│   └── index.html                      self-contained dark-mode web client
│
├── evaluation/
│   ├── eval_harness.py                 10 criteria · EvalSuite · Evaluator · EvalReport
│   ├── builtin_suites.py               smoke · chat · code · news · search
│   │                                   document · financial · routing
│   └── run_evals.py                    CLI runner
│
├── plugins/
│   ├── plugin_loader.py                discovery · inject_into_orchestrator()
│   └── example_calendar_plugin.py      complete working demo
│
├── admin/
│   └── admin_cli.py                    sessions · kb · kb bulk · traces
│                                       cache · prefs · scheduler · health
│
├── tests/                              693 tests · 16 test files · autouse LLM mock
│   ├── conftest.py
│   ├── test_agents.py                  (19)
│   ├── test_advanced_modules.py        (34)
│   ├── test_api.py                     (21)
│   ├── test_chat_agent.py              (33)
│   ├── test_document_processing.py     (15)
│   ├── test_embedding_backends.py      (47)
│   ├── test_final_modules.py           (25)
│   ├── test_financial_agent.py         (56)
│   ├── test_integration.py             (10)
│   ├── test_mass_uploader.py           (127)
│   ├── test_new_modules.py             (41)
│   ├── test_orchestrator.py            (86)
│   ├── test_rag_precheck.py            (49)
│   ├── test_rate_limiter.py            (18)
│   ├── test_tools.py                   (23)
│   └── test_voice.py                   (4)
│
├── scripts/
│   ├── install.sh                      dev environment setup
│   ├── setup_ollama.sh                 pull Ollama models
│   └── start_vllm.sh                   launch vLLM server
│
├── .github/workflows/ci.yml            lint · test matrix · coverage · Docker · audit
├── Dockerfile                          multi-stage (builder + slim runtime)
├── docker-compose.yml                  Ollama + assistant + optional vLLM GPU profile
├── Makefile
├── pyproject.toml                      pytest · ruff · black · mypy · coverage config
├── requirements.txt                    all pinned dependencies (setuptools<81 for vllm)
└── CHANGELOG.md
```

---

## License

MIT
