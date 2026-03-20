# Virtual Personal Assistant

A fully local, privacy-first AI assistant with deep multi-agent capabilities — no cloud APIs, no data leaving your machine.

```
╔══════════════════════════════════════════════════════════════════╗
║          Virtual Personal Assistant  v1.0                        ║
║  Code · News · Search · Documents · Voice                        ║
╚══════════════════════════════════════════════════════════════════╝

You> Write a binary search in Python and test it
[code_agent] Here's a clean implementation with tests…

You> What are today's AI headlines?
[news_agent] ## Latest Headlines (2025-01-15 09:00 UTC)…

You> Summarise the Q3 report I uploaded
[document_agent] Based on Q3_Report › Page 4 › Revenue Analysis…
  [1] Q3_Report › Page 4 › Revenue Analysis
  [2] Q3_Report › Page 7 › Outlook
```

---

## Quick Start

```bash
# Clone and launch — everything else is automatic
git clone https://github.com/yourname/virtual-assistant.git
cd virtual-assistant
bash start.sh
```

`start.sh` handles the full setup on first run: creates a virtualenv, installs all dependencies, copies `.env`, creates data directories, starts Ollama, and pulls the configured model. Subsequent runs skip completed steps automatically.

### Other launch modes

```bash
bash start.sh voice          # Whisper speech-to-text + offline TTS
bash start.sh api            # FastAPI server on http://localhost:8080
bash start.sh docker         # Full Docker Compose stack
bash start.sh test           # Run the 233-test suite

bash start.sh --query "What is quantum entanglement?"   # single shot
bash start.sh --ingest ./documents/                     # ingest a folder
bash start.sh --model llama3.2:3b                       # lighter model
bash start.sh --backend vllm api                        # GPU server mode
bash start.sh --help                                    # all options
```

---

## Features

| Capability | Detail |
|---|---|
| **4 specialist agents** | Code · News · Search · Document — auto-routed by intent |
| **Document Q&A** | Ingest PDF, DOCX, XLSX, PPTX via Docling; answers cite page + section |
| **Voice I/O** | Whisper STT (offline, 99 languages) + pyttsx3 TTS |
| **Dual LLM backend** | Ollama (CPU) or vLLM (GPU) — swap with one env var |
| **FastAPI server** | REST + WebSocket streaming + Server-Sent Events |
| **Web UI** | Dark-mode chat with streaming, document upload, reference display |
| **Resilience** | Circuit breaker, retry with backoff, automatic failover to secondary LLM |
| **Tool cache** | TTL-based cache with `@cached_tool` decorator, disk persistence |
| **Long-term memory** | Episodic memory store — recalls relevant facts from past sessions |
| **Async fan-out** | Run multiple agents in parallel with `asyncio.gather` |
| **Plugin system** | Drop a `.py` file in `plugins/` to add new agents and tools |
| **Evaluation harness** | 10 criteria types, built-in suites, LLM judge, JSON reports |
| **Admin CLI** | Inspect sessions, traces, KB, cache, tasks from the terminal |
| **Scheduled tasks** | Background cron jobs: cache purge, KB re-index, news digest |
| **Structured logging** | JSON log lines with session tagging and per-request tracing |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Interfaces                                    │
│   CLI (Typer)  ·  Interactive REPL  ·  Voice loop  ·  Web UI            │
│   FastAPI REST  ·  WebSocket stream  ·  SSE /stream                     │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────────┐
│                         Orchestrator                                    │
│            Keyword router → LLM intent classifier                       │
│            Shared PersistentMemory · Tracing · Rate limiter             │
└───────┬──────────┬──────────┬────────────────────┬───────────────────── ┘
        │          │          │                    │
  ┌─────▼──┐  ┌────▼───┐  ┌──▼──────┐   ┌────────▼────────┐
  │  Code  │  │  News  │  │ Search  │   │    Document     │
  │ Agent  │  │ Agent  │  │  Agent  │   │     Agent       │
  └─────┬──┘  └────┬───┘  └──┬──────┘   └────────┬────────┘
        │          │          │                    │
  ┌─────▼──┐  ┌────▼───┐  ┌──▼──────┐   ┌────────▼────────┐
  │ Write  │  │  RSS   │  │DuckDuck │   │    Docling      │
  │ Review │  │ Topics │  │Go · Wiki│   │  PDF/DOCX/XLSX  │
  │ Execute│  │ Fetch  │  │Calc·Web │   │     /PPTX       │
  └────────┘  └────────┘  └─────────┘   └────────┬────────┘
                                                  │
                                         ┌────────▼────────┐
                                         │    ChromaDB     │
                                         │  Vector Store   │
                                         │ chunk + refs    │
                                         └─────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            Core Layer                                   │
│  LLM Manager (Ollama ↔ vLLM)  ·  ResilientLLM (circuit breaker)         │
│  Whisper STT  ·  pyttsx3 TTS  ·  ConversationMemory + Summariser        │
│  EpisodicMemory  ·  ToolCache  ·  Tracer  ·  TaskScheduler              │
│  AsyncAgentRunner  ·  UserPreferences  ·  StructuredLogger              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          Plugin System                                  │
│   plugins/*.py  →  register_agents()  +  register_tools()               │
│   Entry-points group: virtual_assistant.plugins                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## CLI Reference

### `start.sh` — the only command you need

```bash
bash start.sh [MODE] [OPTIONS]

Modes:
  chat      Interactive text REPL (default)
  voice     REPL with Whisper STT + TTS
  api       FastAPI server (REST + WebSocket + SSE)
  docker    Full Docker Compose stack
  test      Run the 233-test suite

Options:
  --query  TEXT   Run a single query and exit
  --ingest PATH   Ingest a document or directory, then start
  --backend NAME  llm backend: ollama | vllm
  --model  TAG    Ollama model override (e.g. llama3.2:3b)
  --port   INT    API server port (default: 8080)
  --fresh         Force re-install of all dependencies
  --help          Show all options
```

### `cli.py` — full command set

```bash
python cli.py chat                       # interactive REPL
python cli.py chat --voice               # with voice I/O
python cli.py ask "Explain SOLID"        # single query
python cli.py ask "Write a sort" --agent code

python cli.py ingest report.pdf          # index one document
python cli.py ingest ./docs/ --directory # index a folder

python cli.py docs list                  # list knowledge base
python cli.py docs search "revenue Q3"  # semantic search
python cli.py docs search "costs" --doc "Q3 Report"
python cli.py docs delete "Q3 Report"

python cli.py transcribe meeting.wav     # Whisper transcription
python cli.py config                     # show configuration
```

### `admin/admin_cli.py` — operational tools

```bash
python admin/admin_cli.py sessions list
python admin/admin_cli.py sessions clear my-session
python admin/admin_cli.py kb stats
python admin/admin_cli.py kb ingest ./reports/
python admin/admin_cli.py traces list --n 20
python admin/admin_cli.py traces stats
python admin/admin_cli.py cache stats
python admin/admin_cli.py cache clear
python admin/admin_cli.py prefs show alice
python admin/admin_cli.py prefs set alice response_style technical
python admin/admin_cli.py scheduler list
python admin/admin_cli.py scheduler run cache_purge
python admin/admin_cli.py health
```

### REPL commands (prefix with `:`)

| Command | Description |
|---|---|
| `:help` | Show all commands |
| `:quit` / `:exit` | Exit |
| `:clear` | Clear conversation memory |
| `:docs` | List ingested documents |
| `:ingest <path>` | Ingest a document |
| `:delete <title>` | Remove a document |
| `:agent <name>` | Force next query to a specific agent |
| `:history` | Show recent conversation |
| `:voice` | Toggle voice input/output |

---

## API Reference

Server starts with `bash start.sh api` or `uvicorn api.server:app --port 8080`.

### REST endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Single query → JSON response |
| `DELETE` | `/chat/{session_id}` | Clear session memory |
| `GET` | `/stream` | SSE streaming (EventSource / curl) |
| `WS` | `/ws/{session_id}` | WebSocket streaming |
| `POST` | `/documents/ingest` | Upload + ingest a document |
| `GET` | `/documents` | List knowledge base |
| `DELETE` | `/documents/{title}` | Remove a document |
| `GET` | `/documents/search?q=...` | Semantic search |
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Cache, rate limiter, tracer stats |
| `GET` | `/traces` | Recent request traces |
| `GET` | `/docs` | Swagger UI |

### WebSocket event types

```json
{"type": "start",     "agent": "code_agent"}
{"type": "token",     "content": "Here "}
{"type": "reference", "ref": "Q3_Report › Page 4 › Revenue"}
{"type": "done",      "latency_ms": 1234, "tool_call_count": 3}
{"type": "error",     "message": "..."}
```

### SSE streaming (curl example)

```bash
curl -N "http://localhost:8080/stream?query=What+is+Python&session_id=s1"
```

---

## Supported Document Types

| Format | Extensions | Parser |
|---|---|---|
| PDF | `.pdf` | Docling (OCR, tables, heading detection) |
| Word | `.docx` | Docling → python-docx fallback |
| Excel | `.xlsx` `.xls` | Docling → openpyxl fallback |
| PowerPoint | `.pptx` `.ppt` | Docling → python-pptx fallback |

Every chunk carries a full reference: **`Title › Page N › Section`**. The document agent always surfaces these in its answers:

```
[1] Q3_Report › Page 4 › Revenue Analysis  (relevance: 0.92)
[2] Q3_Report › Page 7 › Outlook           (relevance: 0.87)
```

---

## Configuration

All settings live in `.env`. The file is created automatically from `.env.example` on first run.

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `vllm` |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `VOICE_ENABLED` | `false` | Enable mic + TTS |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embeddings |
| `CHUNK_SIZE` | `512` | Document chunk size |
| `CHUNK_OVERLAP` | `64` | Chunk overlap |
| `VECTOR_STORE_PATH` | `./data/vector_store` | ChromaDB path |
| `AGENT_MAX_ITERATIONS` | `15` | Max ReAct steps |
| `AGENT_VERBOSE` | `false` | Print tool reasoning |
| `RATE_LIMIT_REQUESTS` | `60` | API requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Recommended models by hardware

| RAM / VRAM | Model |
|---|---|
| 4 GB | `llama3.2:3b` · `phi3:mini` |
| 8 GB | `llama3.1:8b` · `mistral:7b` ← **default** |
| 16 GB | `llama3.1:70b-q4` · `qwen2.5:14b` |
| GPU 16 GB+ | vLLM with `mistralai/Mistral-7B-Instruct-v0.2` |

---

## LLM Backends

### Ollama (default — CPU, easy setup)

```bash
# Install: https://ollama.com/download
# start.sh pulls models automatically. Or manually:
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### vLLM (GPU — higher throughput)

```bash
pip install vllm
bash scripts/start_vllm.sh --model mistralai/Mistral-7B-Instruct-v0.2

# .env:
LLM_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8000/v1
```

vLLM uses continuous batching and PagedAttention for significantly higher throughput under concurrent load.

---

## Voice

```bash
# System requirements
sudo apt-get install ffmpeg libsndfile1 espeak-ng portaudio19-dev  # Linux
brew install ffmpeg portaudio espeak                               # macOS

# Run with voice
bash start.sh voice
# or inside REPL: type :voice to toggle
```

| Component | Library | Notes |
|---|---|---|
| Speech-to-text | [Whisper](https://github.com/openai/whisper) | Fully offline, 99 languages |
| Text-to-speech | pyttsx3 + espeak-ng | Offline, no API key |
| Mic capture | sounddevice | Energy-based VAD |

---

## Adding a New Agent

All agents extend `BaseAgent`. Minimum implementation (~50 lines):

```python
# agents/my_agent.py
from agents.base_agent import AgentResponse, BaseAgent
from tools.my_tools import get_my_tools

class MyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "my_agent"

    @property
    def description(self) -> str:
        return "Does X. Use for: A, B, C."

    def get_tools(self):
        return get_my_tools()

    def run(self, query: str, **kwargs) -> AgentResponse:
        executor = self._build_react_agent()
        result   = executor.invoke({"input": query})
        return AgentResponse(
            output=result["output"],
            agent_name=self.name,
            tool_calls=self._extract_tool_calls(result.get("intermediate_steps", [])),
        )
```

Register in `agents/orchestrator.py`:

```python
self._agents[Intent.MY_INTENT] = MyAgent(memory=self._memory)
```

### Plugin system (no core changes needed)

Drop a file in `plugins/` that exports `register_agents()` and/or `register_tools()`:

```python
# plugins/weather_plugin.py
def register_agents():
    return {"weather_agent": WeatherAgent}

def register_tools():
    return [WeatherTool()]
```

It is discovered and loaded automatically at startup. See `plugins/example_calendar_plugin.py` for a complete working example.

---

## Evaluation

```bash
python evaluation/run_evals.py --suite smoke      # quick smoke test
python evaluation/run_evals.py --suite code       # code agent tests
python evaluation/run_evals.py --suite all --save # all + JSON report

python evaluation/run_evals.py --list             # see all suites
```

Built-in suites: `smoke` · `code` · `news` · `search` · `document` · `routing`

Write custom criteria:

```python
from evaluation import Contains, NoError, UsedTools, EvalCase, EvalSuite, Evaluator

suite = EvalSuite("my_suite")
suite.add(EvalCase(
    query="Write fibonacci in Python",
    criteria=[Contains("def fibonacci"), NoError(), UsedTools("code_writer")],
    agent_hint="code",
))
report = Evaluator().run(suite)
report.print_summary()
```

---

## Docker

```bash
# Build and run CLI
docker build -t virtual-assistant .
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  virtual-assistant

# Full stack (Ollama + assistant)
docker compose up -d ollama
docker compose run --rm assistant bash scripts/setup_ollama.sh
docker compose run -it --rm assistant

# API server in Docker
docker run -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  virtual-assistant \
  sh -c "uvicorn api.server:app --host 0.0.0.0 --port 8080"
```

GPU support via the `gpu` Compose profile — see `docker-compose.yml`.

---

## Development

```bash
make install-dev          # install all deps + dev extras
make test                 # run 233 tests
make test-cov             # with HTML coverage → data/coverage/
make test-unit            # unit tests only
make test-api             # API endpoint tests
make lint                 # ruff check
make format               # black + ruff --fix
make typecheck            # mypy
make eval-smoke           # quick agent quality check
make eval-all             # full eval suite + JSON report
make run-api              # start FastAPI with auto-reload
```

---

## Project Structure

```
virtual-assistant/
├── start.sh                       ← single entry point for everything
├── .env                           ← your configuration (git-ignored)
├── .env.example                   ← template
│
├── agents/
│   ├── base_agent.py              SOLID base class + AgentResponse
│   ├── code_agent.py              write / review / execute code
│   ├── news_agent.py              RSS + DuckDuckGo news
│   ├── search_agent.py            web search, Wikipedia, calculator
│   ├── document_agent.py          Docling + vector search + references
│   └── orchestrator.py            LangGraph router + shared memory
│
├── core/
│   ├── llm_manager.py             Ollama / vLLM factory + embeddings
│   ├── async_runner/              asyncio fan-out + streaming
│   ├── cache/                     TTL tool cache + @cached_tool
│   ├── logging/                   JSON structured logger + tracer tags
│   ├── long_term_memory/          episodic memory (cross-session recall)
│   ├── memory/                    sliding-window + persistent memory
│   ├── resilience/                circuit breaker + retry + failover
│   ├── scheduler/                 cron-style background tasks
│   ├── summariser/                rolling conversation compression
│   ├── tracing/                   per-request span collector
│   ├── user_prefs/                per-user settings + style injection
│   └── voice/                     Whisper STT + pyttsx3 TTS
│
├── document_processing/
│   ├── docling_processor.py       Docling → DocumentChunk with references
│   ├── vector_store.py            ChromaDB store + semantic search
│   └── document_manager.py        high-level façade
│
├── tools/
│   ├── code_tools.py              CodeWriter, CodeReviewer, Executor
│   ├── document_tools.py          ingest, search, list, get, delete
│   ├── news_tools.py              headlines, topic search, RSS
│   └── search_tools.py            DuckDuckGo, Wikipedia, web fetch, calc
│
├── api/
│   ├── server.py                  FastAPI app + WebSocket + tracing
│   ├── sse.py                     Server-Sent Events /stream endpoint
│   ├── rate_limiter.py            sliding-window per-IP rate limiting
│   └── startup_validator.py       pre-flight health checks
│
├── ui/
│   └── index.html                 self-contained dark-mode web client
│
├── evaluation/
│   ├── eval_harness.py            criteria, EvalSuite, Evaluator
│   ├── builtin_suites.py          smoke / code / news / search / doc / routing
│   └── run_evals.py               CLI runner
│
├── plugins/
│   ├── plugin_loader.py           auto-discovery + inject_into_orchestrator
│   └── example_calendar_plugin.py working demo plugin
│
├── admin/
│   └── admin_cli.py               operational CLI (sessions, KB, traces, cache…)
│
├── tests/                         233 tests across 9 files
│
├── scripts/
│   ├── install.sh                 dev environment setup
│   ├── setup_ollama.sh            pull Ollama models
│   └── start_vllm.sh              launch vLLM server
│
├── .github/workflows/ci.yml       GitHub Actions (lint, test matrix, Docker, audit)
├── Dockerfile                     multi-stage build
├── docker-compose.yml             Ollama + assistant + vLLM GPU profile
├── Makefile                       all dev shortcuts
├── pyproject.toml                 pytest, ruff, black, mypy, coverage config
└── requirements.txt               all dependencies
```

---

## License

MIT
