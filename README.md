# Virtual Personal Assistant

A fully local, privacy-first AI assistant built on LangChain, Ollama, and ChromaDB.
Runs entirely on your machine — no data leaves your network.

**1,088 tests · 23 test files · 99 Python files · 32,000+ lines**

---

## Quick Start

```bash
# 1. Pull required models
ollama pull llama3.1:8b          # chat and all general agents
ollama pull nomic-embed-text     # document embeddings
ollama pull deepseek-r1:7b       # deep research (reasoning model)

# 2. Install Playwright browser (for JS-rendered page fetching)
playwright install chromium

# 3. Launch
bash start.sh                    # interactive REPL
bash start.sh api                # REST + WebSocket + SSE server on :8080
bash start.sh --query "What is CRISPR?"   # single-shot query
bash start.sh --ingest ./docs/            # ingest a directory
```

---

## Architecture

```
User  (REPL · API · Voice · CLI)
  │
  ▼
Orchestrator
  ├── Stage 1: keyword regex router (fast, no LLM)
  ├── Stage 2: LLM classifier (ambiguous queries only)
  ├── Schedule detector  ─── "remind me every hour" → TaskScheduler
  ├── RAG pre-check      ─── vector store scan before any agent
  │
  └── Agent registry (9 specialist agents + fallback chains)
        ChatAgent          ← conversation, creative, opinions
        CodeAgent          ← write / debug / execute / save code
        NewsAgent          ← RSS feeds, headlines, topic news
        SearchAgent        ← web search, Wikipedia, calculator
        DocumentAgent      ← PDF/DOCX/XLSX/PPTX Q&A
        FinancialAgent     ← stock quotes, ratios, earnings, analysis
        DeepResearchAgent  ← multi-turn plan→gather→evaluate→synthesise
        DataAnalysisAgent  ← CSV/Excel, pandas, matplotlib charts
        WritingAssistantAgent ← outline→draft→edit→DOCX/Markdown export
```

### Fallback chains

| Primary | Fallbacks |
|---|---|
| DOCUMENT | SEARCH → CHAT |
| SEARCH | DOCUMENT → CHAT |
| NEWS | SEARCH → CHAT |
| FINANCE | SEARCH → CHAT |
| CODE | SEARCH → CHAT |
| RESEARCH | SEARCH → CHAT |
| DATA | CODE → CHAT |
| WRITING | CHAT |
| CHAT | SEARCH |

---

## Agents

### ChatAgent
General-purpose conversational agent. Injects full conversation history and
user style preferences. True token-by-token streaming in the REPL.

**Triggers:** greetings, opinions, explanations, creative writing, brainstorming, follow-up questions

### CodeAgent
Writes, debugs, reviews, and executes Python. Saves output as files.

**Tools:** `code_writer` · `code_reviewer` · `python_executor` (sandboxed) ·
`save_code_file` · `save_text_file`

**Sandbox:** expanded blocklist (os.remove / shutil / subprocess / socket / import tricks),
minimal env isolation, 64 KB output cap.

**Triggers:** write / debug / review / execute code

### NewsAgent
Fetches and summarises news from RSS feeds and DuckDuckGo News.

**Tools:** `headlines` · `topic_news` · `rss_feed`

**Triggers:** news, headlines, latest, breaking, current events

### SearchAgent
Factual lookup, web search, Wikipedia, and arithmetic.

**Tools:** `web_search` · `ranked_web_search` · `fetch_webpage` ·
`fetch_webpage_js` · `wikipedia_lookup` · `calculator`

**Triggers:** what is, who is, how does, search for, calculate

### DocumentAgent
Answers questions about uploaded documents. Auto-detects comparison and
table queries for specialised handling.

**Tools:** `search_documents` · `search_documents_tables` · `compare_documents` ·
`ingest_document` · `update_document` · `list_documents` · `get_full_document` ·
`delete_document` · `bulk_ingest_directory` · `bulk_ingest_files`

**Triggers:** in the document, uploaded file, knowledge base, what does the report say

### FinancialAgent
Stock quotes, financial ratios, earnings, portfolio analysis.

**Tools:** `stock_quote` · `company_profile` · `financial_ratios` ·
`price_history` · `income_statement` · `compare_stocks` · `financial_summary`

**Triggers:** stock price, P/E ratio, earnings, revenue, market cap, portfolio

### DeepResearchAgent
Multi-turn research pipeline using a dedicated reasoning model.

**Workflow:** `Plan` (sub-questions + search queries) →
`Gather` (web + Wikipedia) → `Evaluate` (SUFFICIENT or NEED_MORE) →
`Synthesise` (Markdown report with `[n]` citations)

**Model:** `OLLAMA_REASONING_MODEL` (default: `deepseek-r1:7b`). Strips
`<think>…</think>` blocks from deepseek-r1 output.

**Triggers:** research, deep dive, comprehensive report, literature review, investigate

### DataAnalysisAgent *(new)*
Analyses structured data (CSV, TSV, Excel) with pandas and matplotlib.
A shared `_FRAMES` registry keeps datasets in memory across tool calls within
one session.

**Tools:**

| Tool | What it does |
|---|---|
| `load_data` | Load CSV / TSV / Excel → pandas DataFrame |
| `inspect_data` | Shape, dtypes, head, describe, missing values |
| `run_pandas` | Execute arbitrary pandas code (blocklisted); captures `result` or `print()` |
| `plot_data` | Generate matplotlib chart → PNG in `data/uploads/generated/charts/` |
| `export_data` | Save transformed DataFrame → CSV or Excel |
| `list_loaded_data` | Show all datasets currently in memory |

**Auto-load:** if `file_path` is in the query kwargs or the query mentions a
file path, the dataset is loaded before the ReAct loop starts.

**Triggers:** CSV, Excel, spreadsheet, pandas, plot, chart, correlation,
distribution, group by, aggregate, analyse the data

### WritingAssistantAgent *(new)*
Long-form writing partner: outline → draft → edit → export.
A shared `_DRAFTS` registry stores all drafts in memory within one session.

**Tools:**

| Tool | What it does |
|---|---|
| `create_outline` | LLM → structured JSON outline with section headings and notes |
| `draft_section` | Write one section at a time with full prior-section context |
| `assemble_draft` | Combine all sections into a single Markdown document |
| `edit_draft` | Rewrite full draft or one section on natural-language instructions |
| `export_markdown` | Save to `.md` file |
| `export_docx` | Save to `.docx` (Heading styles, bold/italic, page margins) |
| `list_drafts` | Show active drafts with completion status |

**Fast path:** emails, poems, ≤ 300 words, or "brief/short/concise" keywords
bypass the outline pipeline and call the LLM directly.

**Doc-type inference:** automatically detects article / essay / report /
blog_post / email / letter / technical_doc / poem from the query.

**Triggers:** write an article / essay / report, draft a report, compose a letter,
create a blog post, help me write, outline for, export to DOCX

---

## Document Preprocessing Pipeline

Every document passes through a three-stage pipeline before vector storage.

### Stage 1 — Parse → Markdown

Docling converts the source to structured Markdown. Fallback renderers:

| Format | Renderer | Output |
|---|---|---|
| PDF | Docling ▸ pypdf | `## Page N` headings + paragraphs |
| DOCX | Docling ▸ python-docx | `# / ## / ###` headings + body |
| XLSX | Docling ▸ openpyxl | `## Sheet` + GFM pipe-table |
| PPTX | Docling ▸ python-pptx | `## Slide N` headings + body |

### Stage 2 — UTF-8 normalisation (`to_utf8`)

```
bytes → decode(source_encoding, errors="replace")
     → UTF-8 round-trip  (flush lone surrogate code points)
     → unicodedata.normalize("NFKC")   ﬁ→fi · １→1 · ™→TM
     → strip control chars  (\x00–\x1f, \x7f–\x9f except \t \n \r)
```

### Stage 3 — Heading-aware chunking

Markdown is parsed line-by-line. The active heading stack is tracked and
embedded in every chunk's `section_path`. `## Page N` / `## Slide N` markers
set `page_number` without polluting the breadcrumb.

### Chunk metadata

```python
DocumentChunk(
    chunk_id     = "a3f9c2d1b8e47f6a",   # SHA-256 of path:page:offset
    text         = "## Revenue\n\n| Q | Revenue |\n…",  # UTF-8 Markdown
    doc_path     = "/data/uploads/Q3_Report.pdf",
    doc_title    = "Q3 Report",
    page_number  = 4,
    section_path = ["Financial Summary", "Revenue Analysis"],
    metadata     = {"doctype": "pdf", "markdown_source": True},
)
```

---

## Mass Document Upload

```bash
python cli.py ingest ./docs/ --directory --workers 4
python cli.py ingest ./docs/ --directory --dry-run   # plan only
python cli.py ingest report.pdf                       # single file
```

**Split-lane processing** — pdfium safety:
- DOCLING / FALLBACK (PDF, DOCX) → sequential lane, fresh processor per file
- TEXT / STRUCTURED (TXT, MD, CSV, JSON) → parallel `ThreadPoolExecutor`

**Document update:**
```bash
python cli.py ingest updated_report.pdf   # replaces old version if content hash changed
```
`ingest_or_update()` computes SHA-256 of the file content. If the hash matches
the stored chunks it skips; if changed it deletes all old chunks and re-ingests.

---

## Memory & Personalisation

### User profile
Learned automatically from conversation — no configuration required.

```
You: My name is Alice and I prefer technical answers.
Assistant: Got it, Alice. I'll keep my responses technical.
```

- **Fast heuristics** (< 1ms, every turn): name extraction via regex
- **LLM extraction** (background thread, every 10th turn): interests, style, preferred agent
- Stored in `data/logs/user_prefs/<user_id>.json` (Pydantic model)

REPL: `:prefs` (view profile) · `:memory` (long-term facts)

### Session recall
When you return after ≥ 4 hours, the assistant greets you with a recap of
the last 3 stored facts from episodic memory.

### Long-term memory (episodic)
Enabled by default. After each response the LLM extracts 0–3 memorable facts
and stores them in a dedicated ChromaDB collection (`episodic_memory`).
Facts are recalled automatically on semantically relevant future queries.

---

## Task Scheduling

Schedule recurring tasks using natural language:

```
You: Remind me to check Apple stock every hour.
You: Send me a news briefing every morning.
You: Schedule a portfolio analysis every Friday.
```

The orchestrator detects scheduling patterns and registers tasks with the
background `TaskScheduler`. Tasks persist to `data/logs/user_tasks.json`
and reload on startup.

REPL: `:tasks` — view and manage scheduled tasks.

---

## File Output

Agents can save results as files:

```
You: Write a Python script to analyse the CSV and save it.
You: Generate a Markdown report on this topic and save it.
You: Export this data as JSON.
You: Write a report and export it to Word.
```

Files saved to `data/uploads/generated/` (text / code / JSON) or
`data/uploads/generated/charts/` (PNG charts) or
`data/uploads/generated/writing/` (DOCX / Markdown documents).

---

## Web Capabilities

### Re-ranked search (`ranked_web_search`)
Fetches up to 8 raw DuckDuckGo results, uses the LLM to score each by
relevance, returns the top-k. Falls back to raw results on any error.

### JS-rendered pages (`fetch_webpage_js`)
Headless Chromium (Playwright) fetches pages requiring JavaScript: SPAs,
dynamic dashboards, cookie-gated content. Supports an optional CSS selector
click before extraction (e.g. dismiss cookie banners).

```bash
playwright install chromium   # one-time setup
```

---

## Conversation Context

All specialist agents receive recent history when a follow-up is detected:

| Signal | Examples |
|---|---|
| Short query (≤ 4 words) | `"why?"` `"how?"` `"and then?"` |
| Referential pronoun | `"it"` `"this"` `"the code above"` `"what you just wrote"` |
| Continuation word | `"but"` `"so"` `"also"` `"additionally"` |

Context is prepended automatically — no user action required.

---

## REPL Commands

Start an interactive session: `python cli.py chat` or `bash start.sh`

| Command | Description |
|---|---|
| `:help` | Show all commands |
| `:quit` / `:exit` | Exit |
| `:clear` | Clear conversation memory |
| `:docs` | List ingested documents |
| `:ingest <path>` | Ingest a document into the knowledge base |
| `:delete <title>` | Remove a document from the knowledge base |
| `:agent <n>` | Force the next query to a specific agent |
| `:history` | Show recent conversation turns |
| `:voice` | Toggle voice I/O |
| `:rag on/off` | Enable or disable the RAG pre-check |
| `:prefs` | Show your user profile and learned preferences |
| `:memory` | Show stored long-term facts from past sessions |
| `:tasks` | Show and manage your scheduled recurring tasks |

Valid agents for `:agent`: `chat` · `code` · `news` · `search` · `document` · `finance` · `research` · `data` · `writing`

---

## CLI Reference

```bash
# Single query
python cli.py ask "Explain SOLID principles"
python cli.py ask "Write a sort function"          --agent code
python cli.py ask "Apple P/E ratio"                --agent finance
python cli.py ask "Research quantum computing"     --agent research
python cli.py ask "Analyse sales.csv"              --agent data
python cli.py ask "Write a blog post about AI"     --agent writing
python cli.py ask "What is CRISPR?" --no-rag       # skip RAG pre-check

# Document management
python cli.py ingest report.pdf
python cli.py ingest ./docs/ --directory --workers 4
python cli.py ingest ./docs/ --directory --dry-run

python cli.py docs list
python cli.py docs search "revenue Q3" --threshold 0.6
python cli.py docs search "costs" --doc "Q3 Report"
python cli.py docs delete "Q3 Report"
python cli.py docs stats

# Utilities
python cli.py transcribe meeting.wav
python cli.py config                               # show all settings
```

---

## Configuration

All settings live in `.env` (copy `.env.example` to get started).

### LLM

| Variable | Default | Notes |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `vllm` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `llama3.1:8b` | All agents except DeepResearch |
| `OLLAMA_REASONING_MODEL` | `deepseek-r1:7b` | DeepResearchAgent only |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Only when `LLM_BACKEND=vllm` |
| `VLLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | |

**Model sizing guide:**

| RAM / VRAM | Recommended model |
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
| `EMBEDDING_DEVICE` | `cpu` | HuggingFace only: `cpu` / `cuda` / `mps` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace only |
| `EMBEDDING_BATCH_SIZE` | `32` | Lower for < 6 GB VRAM |

`EMBEDDING_BACKEND=ollama` is the recommended default — embeddings run inside the
Ollama server process, leaving application-side VRAM free for Docling and the LLM.

### Vector Store

| Variable | Default | Notes |
|---|---|---|
| `VECTOR_STORE_PATH` | `./data/vector_store` | ChromaDB files |
| `VECTOR_STORE_COLLECTION` | `assistant_docs` | |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `UPLOADS_PATH` | `./data/uploads` | Ingested files + generated output |

### Deep Research

| Variable | Default | Notes |
|---|---|---|
| `RESEARCH_MAX_ITERATIONS` | `5` | Gather→evaluate cycles (~15–60s each) |
| `RESEARCH_MAX_SOURCES` | `8` | Max evidence items |
| `RESEARCH_CHUNK_BUDGET` | `6000` | Chars passed to synthesis prompt |

### Agent Behaviour

| Variable | Default | Notes |
|---|---|---|
| `AGENT_MAX_ITERATIONS` | `15` | ReAct loop limit |
| `AGENT_VERBOSE` | `false` | Print tool calls to stdout |
| `MEMORY_MAX_TOKENS` | `4096` | Conversation memory budget |

### Voice

| Variable | Default | Notes |
|---|---|---|
| `VOICE_ENABLED` | `false` | Microphone input + TTS output |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `VOICE_LANGUAGE` | `en` | ISO 639-1 language code |

### API Server

| Variable | Default | Notes |
|---|---|---|
| `RATE_LIMIT_REQUESTS` | `60` | Requests per window per IP |
| `RATE_LIMIT_WINDOW` | `60` | Window size in seconds |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Features at a Glance

| Feature | Detail |
|---|---|
| **9 specialist agents** | Chat · Code · News · Search · Document · Finance · Research · Data · Writing |
| **Intent routing** | 2-stage: keyword regex (fast) → LLM classifier (ambiguous) |
| **RAG pre-check** | Vector store scan before any agent; skipped for code/chat/research/data/writing |
| **Fallback loop** | Quality-gated fallback chain with synthesis on total failure |
| **Streaming responses** | Token-by-token REPL streaming via `Orchestrator.stream_response()` |
| **Conversation context** | Follow-up detection; history injected into all agents automatically |
| **User profile** | Name, interests, style — learned from conversation, persisted |
| **Session recall** | Welcome-back recap after ≥ 4 hours away |
| **Long-term memory** | Episodic fact extraction and cross-session recall (ChromaDB) |
| **Task scheduling** | Natural-language recurring tasks ("remind me every hour") |
| **File output** | Agents save results as .md / .py / .json / .csv / .docx |
| **Code sandbox** | Expanded blocklist, env isolation, 64 KB output cap |
| **PDF → Markdown** | UTF-8 normalisation + NFKC + heading-aware chunking |
| **Mass document upload** | Split-lane parallel ingestion with dedup + dry-run |
| **Document update** | SHA-256 hash comparison — replaces only changed documents |
| **Multi-doc comparison** | Side-by-side retrieval from multiple documents |
| **Table-aware retrieval** | Separates GFM pipe-table chunks from prose |
| **Re-ranked search** | LLM scores DuckDuckGo results before returning top-k |
| **JS-rendered pages** | Playwright headless Chromium — handles any modern web page |
| **Data analysis** | pandas + matplotlib — load CSV, run queries, plot charts |
| **Writing assistant** | Outline → draft → edit → export DOCX/Markdown pipeline |
| **Deep research** | Multi-turn plan→gather→evaluate→synthesise with reasoning model |
| **Voice I/O** | Whisper STT + pyttsx3 TTS (offline) |
| **FastAPI server** | REST + WebSocket + SSE streaming |
| **Plugin system** | Drop Python files in `plugins/` to add custom agents |
| **Evaluation harness** | Built-in test suites + custom eval runner |
| **Admin CLI** | Manage sessions, episodic memory, plugins, scheduled tasks |

---

## Project Structure

```
virtual-assistant/
├── agents/
│   ├── base_agent.py              AgentResponse · BaseAgent ABC · _ExecutorCompat
│   ├── chat_agent.py              Direct LLM · streaming · memory injection
│   ├── code_agent.py              5-task router: write/review/debug/execute/save
│   ├── data_analysis_agent.py     pandas · matplotlib · auto-load · file refs  ← new
│   ├── deep_research_agent.py     plan/gather/evaluate/synthesise · reasoning model
│   ├── document_agent.py          7 tools · compare/table fast-paths
│   ├── financial_agent.py         8-task router · 70+ company name map · yfinance
│   ├── news_agent.py              RSS + DuckDuckGo news
│   ├── orchestrator.py            Intent enum · 2-stage routing · fallback loop
│   │                              schedule detection · profile extraction
│   ├── rag_precheck.py            KB scan before agents · _RAG_SKIP_INTENTS
│   ├── search_agent.py            4-route: math/URL/encyclopedic/research
│   └── writing_agent.py           outline→draft→edit→export · fast path       ← new
│
├── tools/
│   ├── code_tools.py              CodeWriter · CodeReviewer · CodeExecutor (sandboxed)
│   ├── data_analysis_tools.py     LoadData · InspectData · RunPandas · PlotData  ← new
│   │                              ExportData · ListLoadedData
│   ├── document_tools.py          IngestDocument · UpdateDocument · Search
│   │                              TableSearch · CompareDocuments · BulkIngest
│   ├── file_output_tools.py       SaveText · SaveCode · SaveJson · ListSaved
│   ├── financial_tools.py         StockQuote · CompanyProfile · Ratios · History
│   ├── news_tools.py              Headlines · TopicNews · RSSFeed
│   ├── search_tools.py            DuckDuckGo · RankedSearch · Wikipedia
│   │                              WebFetch · PlaywrightFetch · Calculator
│   └── writing_tools.py           Outline · DraftSection · Assemble · Edit     ← new
│                                  ExportMarkdown · ExportDocx · ListDrafts
│
├── document_processing/
│   ├── docling_processor.py       3-stage pipeline: parse→markdown→utf8→chunk
│   ├── document_manager.py        ingest_or_update · search_multi_doc
│   │                              compare_documents · search_with_tables
│   ├── mass_uploader.py           split-lane parallel ingestion
│   ├── type_detector.py           12 DocumentTypes · ExtractionStrategy
│   └── vector_store.py            ChromaDB · batch_size · dedup
│
├── core/
│   ├── async_runner/              asyncio bridge
│   ├── cache/                     ToolCache (TTL)
│   ├── conversation_context.py    is_followup_query · build_conversation_context
│   ├── llm_manager.py             get_llm · get_reasoning_llm · get_embeddings
│   ├── logging/                   structured logging
│   ├── long_term_memory/          EpisodicMemory (ChromaDB)
│   ├── memory/                    PersistentMemory · AssistantMemory
│   ├── profile_extractor.py       extract_and_update_profile · build_session_recall
│   ├── resilience/                retry · circuit breaker
│   ├── scheduler/                 TaskScheduler (daemon thread)
│   ├── summariser/                ConversationSummariser
│   ├── tracing/                   Tracer (spans)
│   ├── user_prefs/                UserPreferences (Pydantic · disk-persisted)
│   ├── user_task_scheduler.py     UserTaskManager · parse_schedule()
│   └── voice/                     Whisper STT · MicrophoneListener · TTS
│
├── api/
│   ├── server.py                  FastAPI · /chat · /stream (SSE) · /ws · /ui
│   ├── rate_limiter.py            sliding-window per-IP
│   └── streaming.py               SSE helpers
│
├── config/
│   ├── settings.py                Pydantic BaseSettings · all env vars
│   └── __init__.py
│
├── evaluation/
│   ├── harness.py                 EvalHarness · EvalCase · EvalResult
│   ├── built_in_suites.py         routing · quality · tool · doc suites
│   └── runner.py                  CLI eval runner
│
├── plugins/                       drop .py files here to add custom agents
│
├── tests/                         1,088 tests · 23 files · autouse LLM mock
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
├── cli.py                         Typer CLI (ask · ingest · docs · transcribe · config)
├── main.py                        REPL · streaming · :commands · session recall
├── start.sh                       one-shot launcher (chat/voice/api/docker/test)
├── .env                           all configuration
├── requirements.txt               Python dependencies
└── README.md                      this file
```

---

## Running Tests

```bash
python -m pytest tests/ -v              # full suite (1,088 tests)
python -m pytest tests/ -q              # quiet summary
python -m pytest tests/test_new_agents.py -v   # new agents only
python -m pytest tests/ -k "data"       # data analysis tests
python -m pytest tests/ -k "writing"    # writing assistant tests
```

All LLM and external API calls are mocked via the `no_real_llm` autouse
fixture in `conftest.py`. Tests run offline in seconds.

---

## System Requirements

| Component | Requirement |
|---|---|
| Python | 3.10+ |
| Ollama | Latest (auto-started by `start.sh`) |
| RAM (minimum) | 8 GB (for `llama3.1:8b` + `nomic-embed-text`) |
| RAM (recommended) | 16 GB (for parallel Docling + LLM inference) |
| Disk | 10–20 GB (models + vector store) |
| OS | Linux · macOS · Windows (WSL2) |

Optional:
- `ffmpeg` + `portaudio` — voice I/O
- `playwright install chromium` — JS-rendered page fetching
- Docker + Docker Compose — containerised deployment
