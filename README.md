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
[code_agent] Here's a clean implementation with unit tests…

You> What are today's AI headlines?
[news_agent] ## Latest Headlines (2025-01-15 09:00 UTC)…

You> Summarise the Q3 report I uploaded
[document_agent] Based on Q3_Report.pdf › Page 4 › Revenue…
  [1] Q3_Report › Page 4 › Revenue Analysis
  [2] Q3_Report › Page 7 › Outlook
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
bash start.sh test                                     # run the 384-test suite
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
| **Direct LLM chat** | ChatAgent talks straight to the model with full conversation history |
| **Document Q&A** | Ingest PDF, DOCX, XLSX, PPTX via Docling; answers cite page + section |
| **Voice I/O** | Whisper STT (offline, 99 languages) + pyttsx3 TTS |
| **Dual LLM backend** | Ollama (CPU) or vLLM (GPU) — swap with one env var |
| **FastAPI server** | REST + WebSocket streaming + Server-Sent Events |
| **Dark-mode Web UI** | Streaming chat, document upload, agent selector, reference display |
| **Resilience** | Circuit breaker · retry with exponential backoff · automatic LLM failover |
| **Tool cache** | TTL cache with `@cached_tool` decorator, disk persistence |
| **Long-term memory** | Episodic store — recalls relevant facts from past sessions |
| **Conversation summariser** | Auto-compresses old turns so context never overflows |
| **Async fan-out** | Run multiple agents in parallel with `asyncio.gather` |
| **User preferences** | Per-user style, language, timezone, news topics — persisted |
| **Plugin system** | Drop a `.py` in `plugins/` to add agents and tools without touching core |
| **Evaluation harness** | 10 criteria types, 7 built-in suites, LLM judge, JSON reports |
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
│   CLI (Typer)  ·  Interactive REPL  ·  Voice loop  ·  Web UI           │
│   FastAPI REST ·  WebSocket /ws     ·  SSE /stream  ·  OpenAPI /docs   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                           Orchestrator                                   │
│                                                                          │
│  Intent Router (2-stage)                                                 │
│    Stage 1: keyword regex      < 1ms, no LLM call                       │
│    Stage 2: LLM classification  only for ambiguous queries               │
│                                                                          │
│  ┌──────┬──────┬──────┬──────────┬───────────────┐                      │
│  │ Chat │ Code │ News │  Search  │   Document    │                      │
│  │Agent │Agent │Agent │  Agent   │    Agent      │                      │
│  └──────┴──────┴──────┴──────────┴───────────────┘                      │
│                                                                          │
│  Post-processing pipeline (after every agent call)                      │
│    PersistentMemory.save()  •  EpisodicMemory.extract_and_store()       │
│    ConversationSummariser.maybe_summarise()  •  Tracer.record()         │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                             Core Layer                                   │
│                                                                          │
│  llm_manager       Ollama / vLLM factory · @lru_cache singleton         │
│  resilience        ResilientLLM · CircuitBreaker · retry + failover     │
│  memory            ConversationMemory · PersistentMemory (JSON)          │
│  summariser        ConversationSummariser (rolling LLM compression)     │
│  long_term_memory  EpisodicMemory (ChromaDB, cross-session recall)      │
│  async_runner      AsyncAgentRunner · fan-out · streaming callbacks     │
│  cache             ToolCache (TTL, LRU, disk) · @cached_tool            │
│  tracing           Tracer · Span · TraceStore (JSONL sink)              │
│  user_prefs        UserPreferences (Pydantic, per-user JSON)            │
│  scheduler         TaskScheduler (daemon thread, 4 built-in tasks)      │
│  logging           JsonFormatter · AssistantLogger · agent_call()       │
│  voice             Whisper STT · MicrophoneListener VAD · pyttsx3 TTS   │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          Plugin System                                   │
│   plugins/*.py  →  register_agents()  +  register_tools()              │
│   Entry-point group: virtual_assistant.plugins                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Chat Agent

ChatAgent is the default conversational fallback — it talks directly to the LLM without invoking any external tools.

| Query type | Example |
|---|---|
| Greetings / small talk | `"Hello! How are you?"` |
| Explanations | `"Explain recursion like I'm five"` |
| Creative writing | `"Write a haiku about autumn"` |
| Brainstorming | `"Give me startup ideas for a food tech company"` |
| Opinions and advice | `"What do you think about microservices?"` |
| Reasoning / analysis | `"Compare the pros and cons of Python vs Go"` |
| Translation / rewriting | `"Translate this paragraph to Spanish"` |
| Follow-up questions | Any continuation of a prior answer |
| Ambiguous queries | Anything not clearly code, news, search, or documents |

Every call injects the full sliding-window conversation history so the model maintains context across many turns.

```python
# Streaming
agent = ChatAgent()
for token in agent.stream("Tell me about black holes"):
    print(token, end="", flush=True)

# Force chat mode in CLI
python cli.py ask "Write me a short poem" --agent chat
```

---

## Orchestrator

The `Orchestrator` is the single entry-point for all interactions.

### Intent routing

Two-stage pipeline — Stage 1 is regex (< 1 ms, no LLM call); Stage 2 calls the LLM only when Stage 1 returns ambiguous:

```
"Hello, how are you?"              →  CHAT     (greeting keyword)
"Write a haiku about rain"         →  CHAT     (creative "write a haiku")
"Write a quicksort in Python"      →  CODE     (code noun after "write")
"Debug this TypeError"             →  CODE     (debug keyword)
"What are today's headlines?"      →  NEWS     (headlines keyword)
"Give me a daily news briefing"    →  NEWS     (news + briefing)
"Summarise the uploaded PDF"       →  DOCUMENT (document keyword)
"What is the capital of France?"   →  SEARCH   (no other signals)
"Anything ambiguous..."            →  LLM classifies → defaults to CHAT
```

### OrchestratorState

The LangGraph pipeline threads this TypedDict through every node:

```python
class OrchestratorState(TypedDict, total=False):
    query:    str                    # original user query
    intent:   str                    # resolved Intent enum value
    response: AgentResponse          # agent output
    history:  list[dict[str, str]]   # conversation snapshot (logging)
    metadata: dict[str, Any]         # user_id, doc_title, request_id…
    error:    str                    # set if any node raises
```

### Constructor

```python
orch = Orchestrator(
    session_id        = "alice",
    enable_episodic   = True,    # long-term fact recall across sessions
    enable_summariser = True,    # auto-compress old conversation turns
    enable_plugins    = True,    # load plugins/ on startup (default)
    enable_scheduler  = True,    # start background TaskScheduler
    summarise_after   = 40,      # compress after this many messages
    recent_k          = 10,      # verbatim turns to keep after compression
)
```

### Key methods

```python
# Routing and execution
response = orch.run("Hello!")
response = orch.run("Write a sort", intent="code")
response = orch.run("Summarise the report", doc_title="Q3 Report")
intent   = orch.route_only("Write a haiku")   # → Intent.CHAT, no agent call

# Document management
orch.ingest_document("reports/Q3.pdf")
docs = orch.list_documents()

# Agent registry
orch.add_agent(Intent.CHAT, MyCustomChatAgent())
agent  = orch.get_agent(Intent.CODE)
intents = orch.intents   # [Intent.CHAT, Intent.CODE, …]

# Async
response  = await orch.run_async("Explain TCP/IP")
responses = await orch.fanout("What is Python?", [Intent.CHAT, Intent.SEARCH])

# LangGraph pipeline
graph  = orch.build_graph()
result = graph.invoke({"query": "Hello!"})

# Memory and lifecycle
orch.clear_memory()
orch.start_scheduler()
orch.stop_scheduler()
print(repr(orch))   # <Orchestrator session='alice' agents=[chat, code, …]>
```

---


## Financial Agent

FinancialAgent retrieves real-time market data and financial statements from Yahoo Finance (no API key required) and uses the LLM to synthesise raw numbers into plain-English analysis.

### What it can do

| Query type | Example |
|---|---|
| Stock quote | `"What is Tesla's current price?"` |
| Company profile | `"What does Microsoft do as a business?"` |
| Valuation ratios | `"Is Apple overvalued? Show me the P/E"` |
| Price history | `"How has NVDA performed over 5 years?"` |
| Financial statements | `"Show Apple's income statement"` |
| Multi-stock compare | `"Compare AAPL vs MSFT vs GOOGL"` |
| Deep analysis | `"Full investment analysis of Tesla"` |

### Tools

| Tool | Description |
|---|---|
| `stock_quote` | Current price, change, volume, market cap, 52-week range |
| `company_info` | Business description, sector, employees, executives |
| `financial_statements` | Income statement, balance sheet, or cash flow (annual / quarterly) |
| `financial_ratios` | P/E, P/B, ROE, ROA, margins, D/E — each with a qualitative grade |
| `price_history` | OHLCV history + total return, CAGR, annualised volatility, max drawdown |
| `stock_comparison` | Side-by-side table for up to 5 tickers across 14 metrics |

### Usage

```bash
# CLI
python cli.py ask "What is AAPL trading at?" --agent finance
python cli.py ask "Full analysis of Microsoft" --agent finance

# REPL
:agent finance
> Compare AAPL, MSFT, GOOGL on valuation
```

Every response includes an investment disclaimer. Pass `add_disclaimer=False` to the constructor to disable it in automated pipelines.

### Routing

The orchestrator uses a two-tier keyword matcher to detect financial intent:

```
"What is AAPL trading at?"              →  FINANCE  (stock price keyword)
"Apple stock P/E ratio"                 →  FINANCE  (P/E ratio keyword)
"Compare MSFT vs GOOGL P/E"             →  FINANCE  (compare + P/E)
"Analyse Tesla for investment"          →  FINANCE  (analyse + invest phrase)
"NVDA stock performed over 5 years"     →  FINANCE  (stock performance phrase)
"Write a stock price fetcher"           →  CODE     (code keywords win)
"Latest stock market news"             →  NEWS     (news keywords win)
```

## CLI Reference

### `start.sh`

```bash
bash start.sh [MODE] [OPTIONS]

Modes
  chat      Interactive text REPL (default)
  voice     REPL with Whisper STT + TTS
  api       FastAPI server on :8080
  docker    Full Docker Compose stack
  test      Run the 384-test suite

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
python cli.py ask "Write a sort" --agent code
python cli.py ask "Hello!" --agent chat

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
| `:agent <n>` | Force next query: `chat` `code` `news` `search` `document` |
| `:history` | Show recent conversation |
| `:voice` | Toggle voice I/O |

### `admin/admin_cli.py`

```bash
python admin/admin_cli.py sessions list
python admin/admin_cli.py sessions clear <id>
python admin/admin_cli.py kb stats
python admin/admin_cli.py kb ingest ./reports/
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
python admin/admin_cli.py scheduler enable daily_news_digest
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
{"type": "start",     "agent": "chat_agent"}
{"type": "token",     "content": "Sure, "}
{"type": "reference", "ref": "Q3_Report › Page 4 › Revenue Analysis"}
{"type": "done",      "latency_ms": 823, "tool_call_count": 2}
{"type": "error",     "message": "..."}
```

### SSE streaming

```bash
curl -N "http://localhost:8080/stream?query=Explain+TCP+IP&session_id=s1"
```

Every response includes `X-RateLimit-Limit` and `X-RateLimit-Remaining` headers. `/health` and `/metrics` are exempt from rate limiting.

---

## Document Types

| Format | Extensions | Parser |
|---|---|---|
| PDF | `.pdf` | Docling (OCR, tables, heading detection) |
| Word | `.docx` | Docling → python-docx fallback |
| Excel | `.xlsx` `.xls` | Docling → openpyxl fallback |
| PowerPoint | `.pptx` `.ppt` | Docling → python-pptx fallback |

Every chunk carries a full reference: **`Title › Page N › Section`**

```
[1] Q3_Report › Page 4 › Revenue Analysis  (relevance: 0.92)
[2] Q3_Report › Page 7 › Outlook          (relevance: 0.87)
```

Ingestion is idempotent — re-ingesting the same file skips existing chunks.

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
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embeddings |
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

## LLM Backends

### Ollama (default — CPU, easy)

```bash
# start.sh pulls models automatically. Or manually:
ollama pull llama3.1:8b && ollama pull nomic-embed-text
```

### vLLM (GPU — high throughput)

```bash
bash scripts/start_vllm.sh --model mistralai/Mistral-7B-Instruct-v0.2

# .env:
LLM_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8000/v1
```

---

## Voice

```bash
sudo apt-get install ffmpeg libsndfile1 espeak-ng portaudio19-dev   # Linux
brew install ffmpeg portaudio espeak                                # macOS

bash start.sh voice    # or toggle with :voice inside the REPL
```

| Component | Library | Notes |
|---|---|---|
| Speech-to-text | Whisper | Fully offline, 99 languages |
| Text-to-speech | pyttsx3 + espeak-ng | Offline, no API key |
| Mic capture | sounddevice | Energy-based VAD |

---

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

Register: `orch.add_agent(Intent.SEARCH, MyAgent(memory=orch.memory))`

Or use the plugin system — no core changes needed:

```python
# plugins/my_plugin.py
def register_agents(): return {"my_agent": MyAgent}
def register_tools():  return [MyTool()]
```

See `plugins/example_calendar_plugin.py` for a complete working example.

---

## Evaluation

```bash
python evaluation/run_evals.py --suite smoke    # quick: one case per agent
python evaluation/run_evals.py --suite chat     # conversational quality
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
    agent_hint="code",
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
make test           # 384 tests
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
├── start.sh                       ← single entry point (setup + all modes)
├── .env                           ← your configuration (git-ignored)
├── .env.example                   ← template with all settings documented
│
├── agents/
│   ├── base_agent.py              abstract BaseAgent + AgentResponse
│   ├── chat_agent.py              direct LLM chat, history injection, streaming
│   ├── code_agent.py              write / review / execute code
│   ├── news_agent.py              RSS + DuckDuckGo news aggregation
│   ├── search_agent.py            web search, Wikipedia, calculator
│   ├── document_agent.py          Docling + vector search + chunk references
│   ├── financial_agent.py         8-task routing, yfinance tools, LLM synthesis
│   └── orchestrator.py            Intent · OrchestratorState · Orchestrator
│                                  (2-stage routing · post-processing · LangGraph)
│
├── core/
│   ├── llm_manager.py             Ollama / vLLM factory + @lru_cache singleton
│   ├── async_runner/              AsyncAgentRunner · fan-out · streaming
│   ├── cache/                     ToolCache (TTL, LRU, disk) · @cached_tool
│   ├── logging/                   JsonFormatter · AssistantLogger
│   ├── long_term_memory/          EpisodicMemory (ChromaDB, cross-session)
│   ├── memory/                    ConversationMemory · PersistentMemory
│   ├── resilience/                ResilientLLM · CircuitBreaker · retry
│   ├── scheduler/                 TaskScheduler + 4 built-in tasks
│   ├── summariser/                ConversationSummariser (rolling compression)
│   ├── tracing/                   Tracer · Span · TraceStore (JSONL)
│   ├── user_prefs/                UserPreferences (Pydantic, disk-persisted)
│   └── voice/                     Whisper STT · MicrophoneListener · pyttsx3 TTS
│
├── document_processing/
│   ├── docling_processor.py       Docling → DocumentChunk with breadcrumb refs
│   ├── vector_store.py            ChromaDB · semantic search · LangChain retriever
│   └── document_manager.py        high-level façade
│
├── tools/
│   ├── code_tools.py              CodeWriter · CodeReviewer · Executor
│   ├── document_tools.py          Ingest · Search · List · Get · Delete
│   ├── financial_tools.py         StockQuote · CompanyInfo · Statements · Ratios
│   │                              PriceHistory · StockComparison
│   ├── news_tools.py              Headlines · TopicNews · RSSFeed
│   └── search_tools.py            DuckDuckGo · Wikipedia · WebFetch · Calculator
│
├── api/
│   ├── server.py                  FastAPI app · all endpoints · tracing
│   ├── sse.py                     GET /stream (Server-Sent Events)
│   ├── rate_limiter.py            RateLimiter · RateLimitMiddleware
│   └── startup_validator.py       pre-flight checks
│
├── ui/
│   └── index.html                 self-contained dark-mode web client
│
├── evaluation/
│   ├── eval_harness.py            10 criteria · EvalSuite · Evaluator · EvalReport
│   ├── builtin_suites.py          smoke·chat·code·news·search·document·routing
│   └── run_evals.py               CLI runner
│
├── plugins/
│   ├── plugin_loader.py           discovery · inject_into_orchestrator()
│   └── example_calendar_plugin.py complete working demo
│
├── admin/
│   └── admin_cli.py               sessions·kb·traces·cache·prefs·scheduler·health
│
├── tests/                         384 tests · 11 test files · autouse LLM mock
│   ├── conftest.py
│   ├── test_agents.py             (19 tests)
│   ├── test_chat_agent.py         (33 tests)
│   ├── test_api.py                (21 tests)
│   ├── test_advanced_modules.py   (34 tests)
│   ├── test_document_processing.py (15 tests)
│   ├── test_final_modules.py      (25 tests)
│   ├── test_integration.py        (10 tests)
│   ├── test_new_modules.py        (41 tests)
│   ├── test_rate_limiter.py       (18 tests)
│   ├── test_tools.py              (23 tests)
│   └── test_voice.py              (4 tests)
│
├── scripts/
│   ├── install.sh                 dev environment setup
│   ├── setup_ollama.sh            pull Ollama models
│   └── start_vllm.sh              launch vLLM server
│
├── .github/workflows/ci.yml       lint · test matrix · coverage · Docker · audit
├── Dockerfile                     multi-stage (builder + slim runtime)
├── docker-compose.yml             Ollama + assistant + optional vLLM GPU profile
├── Makefile
├── pyproject.toml                 pytest · ruff · black · mypy · coverage config
├── requirements.txt
└── CHANGELOG.md
```

---

## License

MIT
