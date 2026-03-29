# Virtual Personal Assistant

A fully local, privacy-first AI assistant with a modern web interface.
Runs entirely on your machine — no data leaves your network.

**1,088 tests · 23 test files · 100 Python files**

---

## Quick Start

```bash
# ── Ollama setup (CPU / low-VRAM) ─────────────────────────────────────────────
ollama pull llama3.1:8b           # chat and all general agents
ollama pull nomic-embed-text      # document embeddings
ollama pull deepseek-r1:7b        # deep research reasoning model
playwright install chromium       # JS-rendered page fetching (one-time)
pip install -r requirements.txt

bash start.sh web                 # web app  →  http://localhost:8080/ui/
bash start.sh                     # interactive REPL
bash start.sh api                 # REST + WebSocket API only
bash start.sh --query "What is CRISPR?"

# ── SGLang setup (GPU — higher throughput, lower latency) ─────────────────────
pip install sglang[all]           # requires CUDA 12+, ≥ 8 GB VRAM
huggingface-cli login             # if using a gated model

# Managed stack — starts server + web UI in one command:
bash start.sh sglang                      # SGLang server + web UI  (default)
bash start.sh sglang --app-mode api       # SGLang server + REST API
bash start.sh sglang --app-mode chat      # SGLang server + REPL
bash start.sh sglang --app-mode voice     # SGLang server + voice REPL

# Or run the server standalone (separate terminal):
bash scripts/start_sglang.sh
# Then in another terminal:
bash start.sh web     # with LLM_BACKEND=sglang in .env
```

Default web credentials: **admin / admin123** — change immediately via the Register tab.

---

## Web Interface

Start with `bash start.sh web`, then open **http://localhost:8080/ui/** in your browser.

### Login & Registration

- **Sign In** tab — username + password login, JWT token issued on success
- **Register** tab — create a new account (username, email, password ≥ 6 chars)
- Credentials stored in `data/auth/users.db` (SQLite, bcrypt-hashed passwords)
- JWT tokens are HS256-signed, 24-hour lifetime (configurable via `AUTH_TOKEN_EXPIRE_MINUTES`)

### Chat Interface

Designed like a modern LLM application:

- **Welcome screen** with clickable suggestion prompts to get started quickly
- **Message bubbles** — user messages right (indigo gradient), assistant messages left (bordered)
- **Per-agent label** above each response showing which agent answered
- **Streaming simulation** — responses render token-by-token rather than appearing all at once
- **Markdown rendering** with full syntax highlighting (code blocks, tables, lists, blockquotes)
- **Copy** and **Retry** action buttons appear on hover over any message
- **Source reference pills** appear inline at the bottom of responses that cite documents

### Left Sidebar — Conversation History

Toggle with the **☰** button. Persisted to `localStorage`.

- Conversations grouped into **Today / Yesterday / Earlier**
- Each entry shows an auto-generated title (first message), timestamp, and emoji icon
- Hover a conversation to reveal the **✕** delete button
- **+** button starts a new conversation

### Right Sidebar — Resources

Toggle with the **⊞** button. Three tabs:

| Tab | Content |
|---|---|
| **Sources** | Reference list from the last response, with file-type icons. Offers the 3D graph when ≥ 2 sources are present. |
| **Documents** | All documents currently ingested in the knowledge base |
| **3D Graph** | Interactive three-dimensional relationship graph |

### 3D Relationship Graph

When a response cites multiple sources, a **View 3D Relationship Graph** button appears in the Sources tab. Switching to the Graph tab renders a live Three.js scene:

- **Central node** (purple, pulsing) represents the query
- **Source nodes** orbit the centre, colour-coded and spaced on a Fibonacci sphere for even distribution
- **Catmull-Rom curve edges** connect each source to the centre, with particle points along each edge
- **Rotating torus glow ring** around the central node
- **Mouse drag** — orbit the scene
- **Scroll wheel** — zoom in / out
- **Hover** — raycasting tooltip shows the full reference text
- **↻ button** — toggle auto-rotation
- **⊙ button** — reset camera position
- **Colour legend** below the canvas maps each colour to its source

### Agent Selection

Click any agent pill in the header to force the next message to a specific agent:

`auto` · `chat` · `code` · `news` · `search` · `document` · `finance` · `research` · `data` · `writing`

### Input Controls

- **Enter** — send message · **Shift+Enter** — new line
- **📚 RAG** toggle — enable / disable the vector-store pre-check (highlighted when active)
- **📎 Upload** — upload and ingest a document directly from the chat interface

---

## Architecture

```
Browser  (http://localhost:8080/ui/)
  │
  │  HTTP/WebSocket
  ▼
FastAPI Server  (api/server.py)
  ├── /auth/login · /auth/register · /auth/me   ← JWT auth
  ├── /chat                                      ← single-turn query
  ├── /ws/{session_id}                           ← streaming WebSocket
  ├── /stream                                    ← SSE streaming
  ├── /documents/*                               ← KB management
  ├── /health · /metrics · /traces
  └── /ui/*                                      ← SPA static files
       │
       ▼
  Orchestrator
    ├── Stage 1: keyword regex router  (fast, no LLM)
    ├── Stage 2: LLM classifier        (ambiguous queries only)
    ├── Schedule detector  ── "remind me every hour" → TaskScheduler
    ├── RAG pre-check      ── vector store scan before any agent
    │
    └── Agent registry (9 specialist agents + fallback chains)
          ChatAgent              conversation, creative, opinions
          CodeAgent              write / debug / execute / save code
          NewsAgent              RSS feeds, headlines, topic news
          SearchAgent            web search, Wikipedia, calculator
          DocumentAgent          PDF/DOCX/XLSX/PPTX Q&A
          FinancialAgent         stock quotes, ratios, earnings
          DeepResearchAgent      plan → gather → evaluate → synthesise
          DataAnalysisAgent      CSV/Excel, pandas, matplotlib charts
          WritingAssistantAgent  outline → draft → edit → DOCX/Markdown
```

---

## Agents

### ChatAgent
General-purpose conversation. Injects full history and user style preferences. True token-by-token streaming.

**Triggers:** greetings, opinions, explanations, creative writing, follow-up questions

### CodeAgent
Writes, debugs, reviews, and executes Python. Saves output as files.

**Sandbox:** expanded blocklist (filesystem / subprocess / socket / import tricks), minimal env isolation, 64 KB output cap.

**Tools:** `code_writer` · `code_reviewer` · `python_executor` · `save_code_file` · `save_text_file`

**Triggers:** write / debug / review / execute code

### NewsAgent
Fetches and summarises news from RSS feeds and DuckDuckGo News.

**Tools:** `headlines` · `topic_news` · `rss_feed`

**Triggers:** news, headlines, latest, breaking, current events

### SearchAgent
Factual lookup, web search, Wikipedia, arithmetic.

**Tools:** `web_search` · `ranked_web_search` · `fetch_webpage` · `fetch_webpage_js` · `wikipedia_lookup` · `calculator`

**Triggers:** what is, who is, how does, search for, calculate

### DocumentAgent
Answers questions about uploaded documents. Auto-detects comparison and table queries.

**Tools:** `search_documents` · `search_documents_tables` · `compare_documents` · `ingest_document` · `update_document` · `list_documents` · `get_full_document` · `delete_document` · `bulk_ingest_directory`

**Triggers:** in the document, uploaded file, knowledge base, what does the report say

### FinancialAgent
Stock quotes, ratios, earnings, portfolio analysis.

**Tools:** `stock_quote` · `company_profile` · `financial_ratios` · `price_history` · `income_statement` · `compare_stocks` · `financial_summary`

**Triggers:** stock price, P/E ratio, earnings, revenue, market cap, portfolio

### DeepResearchAgent
Multi-turn research using a dedicated reasoning model.

**Workflow:** Plan (sub-questions + queries) → Gather (web + Wikipedia) → Evaluate (SUFFICIENT or NEED_MORE) → Synthesise (Markdown report with citations)

**Model:** `OLLAMA_REASONING_MODEL` (default: `deepseek-r1:7b`)

**Triggers:** research, deep dive, comprehensive report, literature review

### DataAnalysisAgent *(new)*
Analyses structured data with pandas and matplotlib.

| Tool | What it does |
|---|---|
| `load_data` | Load CSV / TSV / Excel → DataFrame |
| `inspect_data` | Shape, dtypes, head, describe, missing values |
| `run_pandas` | Execute pandas code (blocklisted); captures `result` or `print()` |
| `plot_data` | Generate matplotlib chart → PNG |
| `export_data` | Save transformed DataFrame → CSV or Excel |
| `list_loaded_data` | Show all datasets in memory |

**Triggers:** CSV, Excel, pandas, plot, chart, correlation, group by, analyse the data

### WritingAssistantAgent *(new)*
Long-form writing partner.

| Tool | What it does |
|---|---|
| `create_outline` | LLM → structured outline with headings and notes |
| `draft_section` | Write one section at a time with full prior context |
| `assemble_draft` | Combine all sections into one Markdown document |
| `edit_draft` | Rewrite full draft or one section on instructions |
| `export_markdown` | Save to `.md` file |
| `export_docx` | Save to `.docx` (heading styles, bold/italic, page margins) |

**Fast path:** emails, poems, ≤ 300 words, or "brief/quick/short" — bypasses the outline pipeline.

**Triggers:** write an article / essay / report, draft a report, compose a letter, create a blog post, export to DOCX

---

## Document Preprocessing Pipeline

Every document passes through three stages before vector storage.

### Stage 1 — Parse → Markdown

Docling converts the source to structured Markdown. Fallback renderers:

| Format | Renderer | Output |
|---|---|---|
| PDF | Docling ▸ pypdf | `## Page N` + paragraphs |
| DOCX | Docling ▸ python-docx | `# / ## / ###` headings + body |
| XLSX | Docling ▸ openpyxl | `## Sheet` + GFM pipe-table |
| PPTX | Docling ▸ python-pptx | `## Slide N` + body |

### Stage 2 — UTF-8 normalisation (`to_utf8`)

```
bytes → decode(source_encoding, errors="replace")
     → UTF-8 round-trip  (flush lone surrogates)
     → unicodedata.normalize("NFKC")   ﬁ→fi · １→1 · ™→TM
     → strip control chars (\x00–\x1f, \x7f–\x9f except \t \n \r)
```

### Stage 3 — Heading-aware chunking

Markdown is parsed line-by-line. The active heading stack is tracked and embedded in every chunk's `section_path`.

---

## Memory & Personalisation

### User profile
Learned automatically from conversation.

- **Fast heuristics** (< 1ms, every turn): name extraction via regex (`"My name is X"`)
- **LLM extraction** (background thread, every 10th turn): interests, style preference, preferred agent
- Stored in `data/logs/user_prefs/<user_id>.json`

**REPL:** `:prefs` · `:memory`

### Session recall
When you return after ≥ 4 hours, the assistant greets you with a recap of recent stored facts.

### Long-term memory
Enabled by default. After each response the LLM extracts 0–3 memorable facts stored in ChromaDB (`episodic_memory` collection).

---

## Task Scheduling

```
You: Remind me to check Apple stock every hour.
You: Send me a news briefing every morning.
```

The orchestrator detects scheduling patterns and registers tasks with the background `TaskScheduler`. Tasks persist to `data/logs/user_tasks.json` and reload on restart.

**REPL:** `:tasks`

---

## File Output

Agents can save results as files to `data/uploads/generated/`:

```
You: Write a Python script to analyse the CSV and save it.
You: Write a report and export it to Word.
You: Export this data as JSON.
```

Charts save to `data/uploads/generated/charts/`. Writing exports save to `data/uploads/generated/writing/`.

---

## Web Capabilities

### Re-ranked search (`ranked_web_search`)
Fetches up to 8 DuckDuckGo results, uses the LLM to score each for relevance, returns the top-k.

### JS-rendered pages (`fetch_webpage_js`)
Headless Chromium via Playwright — handles SPAs, dynamic dashboards, cookie banners.

```bash
playwright install chromium   # one-time setup
```


---

## SGLang Inference Backend

SGLang is a high-performance inference server that provides significantly higher
throughput and lower latency than Ollama for GPU workloads.

### Why SGLang

| Feature | Ollama | vLLM | SGLang |
|---|---|---|---|
| CPU support | ✓ | — | — |
| Easy setup | ✓ | — | — |
| RadixAttention (prefix cache) | — | — | ✓ |
| Chunked prefill | — | partial | ✓ |
| EAGLE speculative decoding | — | — | ✓ |
| Ollama-compatible API | native | — | ✓ |
| Multi-user throughput | low | high | highest |

**RadixAttention** reuses the KV-cache of shared prompt prefixes across requests.
For RAG workloads (where every request has the same long system prompt), this
typically reduces latency by 2–5× after the first request.

### Setup

```bash
pip install sglang[all]   # requires CUDA 12+ and PyTorch

# Optional: pre-download the model
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

### Configuration (`.env`)

```env
LLM_BACKEND=sglang
SGLANG_BASE_URL=http://localhost:11435   # port 11435 avoids clash with Ollama on 11434
SGLANG_MODEL=meta-llama/Llama-3.1-8B-Instruct
SGLANG_MEM_FRACTION=0.85                # GPU VRAM fraction (default 0.85)
SGLANG_MAX_PREFILL_TOKENS=16384
SGLANG_NUM_PREDICT=2048
SGLANG_ENABLE_SPECULATIVE=false          # true = EAGLE draft-model speedup
SGLANG_DRAFT_MODEL=                      # lmsys/sglang-EAGLE-Llama3.1-Instruct-8B
```

### Launch modes

```bash
# Managed stack (recommended) — server + app in one command
bash start.sh sglang                     # + web UI  →  http://localhost:8080/ui/
bash start.sh sglang --app-mode api      # + REST API
bash start.sh sglang --app-mode chat     # + interactive REPL
bash start.sh sglang --app-mode voice    # + voice REPL

# Standalone server (then run app separately)
bash scripts/start_sglang.sh             # blocks; Ctrl-C to stop
bash scripts/start_sglang.sh --model Qwen/Qwen2.5-7B-Instruct
bash scripts/start_sglang.sh --quantization fp8    # 4× less VRAM
bash scripts/start_sglang.sh --speculative \
    --draft-model lmsys/sglang-EAGLE-Llama3.1-Instruct-8B
bash scripts/start_sglang.sh --dry-run  # print command without running

# Benchmark — compare Ollama vs SGLang
bash start.sh benchmark   # with BENCHMARK_BACKENDS="ollama sglang"
python -m evaluation.inference_benchmark --backends ollama sglang --runs 10
```

### No silent fallback

When `LLM_BACKEND=sglang` is set, Jarvis will **not** fall back to Ollama.
If the SGLang server is not reachable at startup, the application exits with
a clear error message rather than silently using a different backend.

The `require_backend()` function in `core/llm_manager.py` probes the server
at `SGLANG_BASE_URL/api/tags` before creating the LangChain model instance.
Set `LLM_BACKEND_SKIP_PROBE=true` to disable this check (used in tests).

---

## REPL Commands

```bash
bash start.sh   # or: python cli.py chat
```

| Command | Description |
|---|---|
| `:help` | Show all commands |
| `:quit` / `:exit` | Exit |
| `:clear` | Clear conversation memory |
| `:docs` | List ingested documents |
| `:ingest <path>` | Ingest a document |
| `:delete <title>` | Remove a document |
| `:agent <n>` | Force next query to a specific agent |
| `:history` | Show recent conversation turns |
| `:voice` | Toggle voice I/O |
| `:rag on/off` | Enable or disable the RAG pre-check |
| `:prefs` | Show your user profile |
| `:memory` | Show stored long-term facts |
| `:tasks` | Show and manage scheduled tasks |

---

## CLI Reference

```bash
# Single query
python cli.py ask "Explain SOLID principles"
python cli.py ask "Write a sort function"            --agent code
python cli.py ask "Apple P/E ratio"                  --agent finance
python cli.py ask "Research quantum computing"       --agent research
python cli.py ask "Analyse sales.csv"                --agent data
python cli.py ask "Write a blog post about AI"       --agent writing
python cli.py ask "What is CRISPR?" --no-rag

# Documents
python cli.py ingest report.pdf
python cli.py ingest ./docs/ --directory --workers 4
python cli.py ingest ./docs/ --directory --dry-run
python cli.py docs list
python cli.py docs search "revenue Q3" --threshold 0.6
python cli.py docs delete "Q3 Report"
python cli.py docs stats

# Utilities
python cli.py transcribe meeting.wav
python cli.py config
```

---

## Configuration (`.env`)

### LLM

| Variable | Default | Notes |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `vllm` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `llama3.1:8b` | All agents except DeepResearch |
| `OLLAMA_REASONING_MODEL` | `deepseek-r1:7b` | DeepResearchAgent only |

**Model sizing guide:**

| RAM / VRAM | Recommended |
|---|---|
| ≤ 4 GB | `llama3.2:3b` · `phi3:mini` |
| ≤ 8 GB | `llama3.1:8b` · `mistral:7b` ← default |
| ≤ 16 GB | `llama3.1:70b-q4` · `qwen2.5:14b` |
| GPU 24 GB | `qwen2.5:72b-q4` · `llama3.3:70b` |

### Embeddings

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_BACKEND` | `ollama` | `ollama` (recommended) or `huggingface` |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Pull with `ollama pull nomic-embed-text` |
| `EMBEDDING_BATCH_SIZE` | `32` | Lower for < 6 GB VRAM |

### Authentication

| Variable | Default | Notes |
|---|---|---|
| `AUTH_SECRET_KEY` | auto-generated | Set in `.env` for stable tokens across restarts |
| `AUTH_TOKEN_EXPIRE_MINUTES` | `1440` | JWT lifetime (24 hours) |

### Vector Store

| Variable | Default | Notes |
|---|---|---|
| `VECTOR_STORE_PATH` | `./data/vector_store` | ChromaDB files |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |

### Deep Research

| Variable | Default | Notes |
|---|---|---|
| `RESEARCH_MAX_ITERATIONS` | `5` | Gather→evaluate cycles |
| `RESEARCH_MAX_SOURCES` | `8` | Max evidence items |
| `RESEARCH_CHUNK_BUDGET` | `6000` | Characters passed to synthesis |

### Agent / Voice / API

| Variable | Default | Notes |
|---|---|---|
| `AGENT_MAX_ITERATIONS` | `15` | ReAct loop limit |
| `MEMORY_MAX_TOKENS` | `4096` | Conversation memory budget |
| `VOICE_ENABLED` | `false` | Requires ffmpeg + portaudio |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `RATE_LIMIT_REQUESTS` | `60` | Requests per window per IP |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Features at a Glance

| Feature | Detail |
|---|---|
| **Web UI** | Login · chat · history sidebar · resources sidebar · 3D graph |
| **SGLang backend** | Ollama-compatible API · RadixAttention · chunked prefill · speculative decoding |
| **No-fallback guarantee** | `LLM_BACKEND=sglang` → hard error if server is down, never silently uses Ollama |
| **JWT authentication** | SQLite user DB, bcrypt hashing, HS256 tokens, 24 h expiry |
| **3D relationship graph** | Three.js — orbit/zoom/hover, Fibonacci layout, auto-rotate |
| **9 specialist agents** | Chat · Code · News · Search · Document · Finance · Research · Data · Writing |
| **Intent routing** | 2-stage: keyword regex → LLM classifier |
| **RAG pre-check** | Vector store scan before any agent |
| **Fallback loop** | Quality-gated chain with synthesis on total failure |
| **Streaming responses** | Token-by-token via WebSocket / SSE |
| **Conversation context** | Follow-up detection; history injected into all agents |
| **User profile** | Name, interests, style — learned from conversation |
| **Session recall** | Welcome-back recap after ≥ 4 hours away |
| **Long-term memory** | Episodic fact extraction, cross-session recall (ChromaDB) |
| **Task scheduling** | Natural-language recurring tasks |
| **File output** | Agents save .md / .py / .json / .csv / .docx |
| **Code sandbox** | Expanded blocklist, env isolation, 64 KB output cap |
| **PDF → Markdown** | UTF-8 normalisation + NFKC + heading-aware chunking |
| **Mass document upload** | Split-lane parallel ingestion with dedup + dry-run |
| **Document update** | SHA-256 hash comparison — replaces only changed documents |
| **Multi-doc comparison** | Side-by-side retrieval across multiple documents |
| **Table-aware retrieval** | Separates GFM pipe-table chunks from prose |
| **Re-ranked search** | LLM scores DuckDuckGo results before returning top-k |
| **JS-rendered pages** | Playwright headless Chromium |
| **Data analysis** | pandas + matplotlib — load CSV, run queries, plot charts |
| **Writing assistant** | Outline → draft → edit → export DOCX/Markdown |
| **Deep research** | Multi-turn plan→gather→evaluate→synthesise |
| **Voice I/O** | Whisper STT + pyttsx3 TTS (offline) |
| **FastAPI server** | REST + WebSocket + SSE streaming |
| **Plugin system** | Drop Python files in `plugins/` to add custom agents |
| **Evaluation harness** | Built-in test suites + custom eval runner |

---

## Project Structure

```
virtual-assistant/
├── agents/
│   ├── base_agent.py              AgentResponse · BaseAgent · _ExecutorCompat
│   ├── chat_agent.py              streaming · memory injection
│   ├── code_agent.py              write / review / debug / execute / save
│   ├── data_analysis_agent.py     pandas · matplotlib · auto-load       ← new
│   ├── deep_research_agent.py     plan → gather → evaluate → synthesise
│   ├── document_agent.py          7 tools · compare/table fast-paths
│   ├── financial_agent.py         8-task router · yfinance
│   ├── news_agent.py              RSS + DuckDuckGo
│   ├── orchestrator.py            Intent enum · routing · fallback loop
│   ├── rag_precheck.py            KB scan · _RAG_SKIP_INTENTS
│   ├── search_agent.py            4-route: math/URL/encyclopedic/research
│   └── writing_agent.py           outline → draft → edit → export       ← new
│
├── tools/
│   ├── code_tools.py              CodeWriter · CodeReviewer · CodeExecutor
│   ├── data_analysis_tools.py     LoadData · RunPandas · PlotData · Export  ← new
│   ├── document_tools.py          Ingest · Search · TableSearch · Compare
│   ├── file_output_tools.py       SaveText · SaveCode · SaveJson
│   ├── financial_tools.py         StockQuote · Ratios · History
│   ├── news_tools.py              Headlines · TopicNews · RSSFeed
│   ├── search_tools.py            DuckDuckGo · RankedSearch · Playwright
│   └── writing_tools.py           Outline · Draft · Assemble · ExportDocx  ← new
│
├── api/
│   ├── auth.py                    SQLite users · bcrypt · JWT             ← new
│   ├── server.py                  FastAPI · /auth/* · /chat · /ws · /ui
│   ├── rate_limiter.py            sliding-window per-IP
│   └── sse.py                     SSE streaming helpers
│
├── document_processing/
│   ├── docling_processor.py       parse → markdown → utf8 → chunk
│   ├── document_manager.py        ingest_or_update · compare · tables
│   ├── mass_uploader.py           split-lane parallel ingestion
│   └── vector_store.py            ChromaDB · batch_size · dedup
│
├── core/
│   ├── conversation_context.py    follow-up detection · context injection
│   ├── llm_manager.py             get_llm · get_embeddings
│   ├── long_term_memory/          EpisodicMemory (ChromaDB)
│   ├── memory/                    PersistentMemory · AssistantMemory
│   ├── profile_extractor.py       extract_and_update_profile · session_recall
│   ├── scheduler/                 TaskScheduler (daemon thread)
│   ├── user_prefs/                UserPreferences (Pydantic · disk-persisted)
│   ├── user_task_scheduler.py     UserTaskManager · parse_schedule()
│   └── voice/                     Whisper STT · TTS
│
├── ui/
│   └── index.html                 SPA — login · chat · sidebars · 3D graph  ← new
│
├── tests/                         1,088 tests · 23 files
│   ├── conftest.py                no_real_llm autouse fixture
│   ├── test_advanced_modules.py   (34)
│   ├── test_agents.py             (19)
│   ├── test_api.py                (21)
│   ├── test_chat_agent.py         (33)
│   ├── test_conversation_and_docs.py (43)
│   ├── test_deep_research_agent.py   (64)
│   ├── test_document_processing.py   (15)
│   ├── test_embedding_backends.py    (48)
│   ├── test_final_modules.py      (25)
│   ├── test_financial_agent.py    (56)
│   ├── test_integration.py        (10)
│   ├── test_mass_uploader.py      (143)
│   ├── test_memory_personalisation.py (49)
│   ├── test_new_agents.py         (68)
│   ├── test_new_modules.py        (41)
│   ├── test_orchestrator.py       (70)
│   ├── test_preprocessing_pipeline.py (62)
│   ├── test_rag_precheck.py       (45)
│   ├── test_rate_limiter.py       (18)
│   ├── test_task_execution.py     (41)
│   ├── test_tools.py              (23)
│   ├── test_voice.py              (4)
│   └── test_web_capabilities.py   (28)
│
├── cli.py                         Typer CLI (ask · ingest · docs · config)
├── main.py                        REPL · streaming · :commands · session recall
├── start.sh                       launcher (chat/voice/api/web/docker/test)
├── .env                           all configuration
├── requirements.txt               Python dependencies
└── README.md                      this file
```

---

## Running Tests

```bash
python -m pytest tests/ -q           # full suite (1,088 tests)
python -m pytest tests/test_api.py   # API tests only
python -m pytest tests/ -k "auth"    # auth-related tests
```

---

## System Requirements

| Component | Requirement |
|---|---|
| Python | 3.10+ |
| Ollama | Latest (auto-started by `start.sh`) |
| RAM (minimum) | 8 GB |
| RAM (recommended) | 16 GB |
| Disk | 10–20 GB (models + vector store) |
| OS | Linux · macOS · Windows (WSL2) |

Optional system packages:

```bash
# Voice I/O
sudo apt-get install ffmpeg portaudio19-dev   # Linux
brew install ffmpeg portaudio                  # macOS

# JS page fetching
playwright install chromium
```
