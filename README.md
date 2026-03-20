# Virtual Personal Assistant

A fully local, privacy-first virtual assistant with deep multi-agent capabilities — no cloud APIs required.

```
╔══════════════════════════════════════════════════════════════╗
║          Virtual Personal Assistant  v1.0                    ║
║  Agents: Code · News · Search · Documents                    ║
║  Backend: ollama        Voice: OFF                           ║
║  Type  :quit  to exit  |  :help  for commands               ║
╚══════════════════════════════════════════════════════════════╝

You: Write a binary search in Python
[code_agent] Here's an implementation...

You: What are today's AI headlines?
[news_agent] ## Latest Headlines ...

You: Summarise the attached report.pdf
[document_agent] Based on the document...
  [1] Q3_Report › Page 4 › Revenue Analysis
  [2] Q3_Report › Page 7 › Outlook
```

---

## Features

| Capability | Detail |
|---|---|
| **Code agent** | Write, review, debug, and execute code in any language |
| **News agent** | Aggregate RSS feeds + DuckDuckGo news search with summaries |
| **Search agent** | Web search, Wikipedia, page fetching, arithmetic |
| **Document agent** | Ingest PDF/DOCX/XLSX/PPTX via Docling; chat with full source references |
| **Voice I/O** | Whisper speech-to-text + pyttsx3 text-to-speech (fully offline) |
| **Dual LLM backend** | Ollama (ease) or vLLM (throughput) — swap with one env var |
| **Persistent memory** | JSON-backed per-session conversation history |
| **Vector store** | ChromaDB with chunk-level references (page + section breadcrumb) |
| **SOLID architecture** | Extensible agent base class; add new agents in ~50 lines |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│              CLI (Typer)  ·  Interactive REPL  ·  Voice loop     │
└─────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────┐
│                        Orchestrator                              │
│          Keyword router → LLM intent classifier                 │
│          Shared persistent memory across all agents             │
└──────┬──────────┬──────────┬──────────────────┬─────────────────┘
       │          │          │                  │
  ┌────▼───┐ ┌───▼────┐ ┌───▼─────┐  ┌────────▼────────┐
  │  Code  │ │  News  │ │ Search  │  │    Document     │
  │ Agent  │ │ Agent  │ │  Agent  │  │     Agent       │
  └────┬───┘ └───┬────┘ └───┬─────┘  └────────┬────────┘
       │         │          │                  │
  ┌────▼───┐ ┌───▼────┐ ┌───▼──────┐  ┌───────▼─────────┐
  │ Write  │ │  RSS   │ │ DuckDuck │  │    Docling      │
  │ Review │ │ News   │ │  Go Web  │  │    Processor    │
  │ Execute│ │ Search │ │ Wikipedia│  │  (PDF/DOCX/     │
  └────────┘ └────────┘ │ Fetch   │  │   XLSX/PPTX)    │
                        │ Calc    │  └───────┬─────────┘
                        └─────────┘          │
                                     ┌───────▼─────────┐
                                     │    ChromaDB     │
                                     │  Vector Store   │
                                     │ (chunk + refs)  │
                                     └─────────────────┘

       ┌─────────────────────────────────────────┐
       │              Core Layer                 │
       │  LLM Manager (Ollama ↔ vLLM factory)   │
       │  Whisper STT · pyttsx3 TTS             │
       │  Persistent Conversation Memory        │
       └─────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourname/virtual-assistant.git
cd virtual-assistant

# Automated setup (creates venv, installs deps, copies .env)
bash scripts/install.sh --dev

# Or manually:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure

Edit `.env`:

```bash
# Choose your inference backend
LLM_BACKEND=ollama          # or "vllm"

# Ollama (default — easiest to start)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# vLLM (higher throughput, requires GPU)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Voice (optional)
VOICE_ENABLED=true
WHISPER_MODEL=base           # tiny | base | small | medium | large
```

### 3. Pull models

```bash
# Ollama (CPU-friendly)
bash scripts/setup_ollama.sh

# vLLM (GPU required)
bash scripts/start_vllm.sh --model mistralai/Mistral-7B-Instruct-v0.2
```

### 4. Run

```bash
python cli.py chat                 # interactive REPL
python cli.py chat --voice         # with voice I/O
python cli.py ask "What is SOLID?" # single query
```

---

## CLI Reference

```
python cli.py --help

Commands:
  chat        Start interactive REPL
  ask         Single query, print response and exit
  ingest      Index a document into the knowledge base
  docs list   List all ingested documents
  docs search Semantic search the knowledge base
  docs delete Remove a document
  transcribe  Transcribe an audio file with Whisper
  config      Show current configuration
```

### Examples

```bash
# Interactive chat
python cli.py chat
python cli.py chat --voice --session my-session

# One-shot queries
python cli.py ask "Write a quicksort algorithm in Rust"
python cli.py ask "What's happening in AI today?" --agent news
python cli.py ask "Explain the attached contract" --agent document

# Document management
python cli.py ingest reports/Q3_2024.pdf
python cli.py ingest ./documents/ --directory        # batch ingest
python cli.py docs list
python cli.py docs search "revenue projections" --k 10
python cli.py docs search "benefits" --doc "Employee_Handbook"
python cli.py docs delete "Q3_2024"

# Audio transcription
python cli.py transcribe meeting_recording.wav
```

---

## REPL Commands

Inside the interactive REPL, prefix commands with `:`:

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

## Supported Document Types

| Format | Extension | Parser |
|---|---|---|
| PDF | `.pdf` | Docling (OCR, tables, headings) |
| Word | `.docx` | Docling → python-docx fallback |
| Excel | `.xlsx` / `.xls` | Docling → openpyxl fallback |
| PowerPoint | `.pptx` / `.ppt` | Docling → python-pptx fallback |

All documents are split into overlapping chunks and stored in ChromaDB. Every chunk carries:
- **Document title** — human-readable name
- **Page / slide number** — for pinpoint navigation
- **Section breadcrumb** — e.g. `Chapter 2 › Revenue Analysis`
- **Chunk ID** — stable SHA-256 for deduplication

When the document agent answers a question, it appends full references:

```
[1] Q3_Report › Page 4 › Revenue Analysis (relevance: 0.92)
[2] Q3_Report › Page 7 › Outlook (relevance: 0.87)
```

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
        return "Does something specific. Use for X, Y, Z."

    def get_tools(self):
        return get_my_tools()

    def run(self, query: str, **kwargs) -> AgentResponse:
        self._executor = self._build_react_agent()   # or override entirely
        result = self._executor.invoke({"input": query})
        return AgentResponse(
            output=result["output"],
            agent_name=self.name,
            tool_calls=self._extract_tool_calls(result.get("intermediate_steps", [])),
        )
```

Then register it in `agents/orchestrator.py`:

```python
# In Orchestrator.__init__
self._agents[Intent.MY_INTENT] = MyAgent(memory=self._memory)
```

And add keyword patterns to `_CODE_KEYWORDS` / create a new regex for routing.

---

## LLM Backends

### Ollama (default)

Best for getting started — runs on CPU, easy setup.

```bash
# Install: https://ollama.com/download
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

Recommended models by hardware:

| VRAM / RAM | Model |
|---|---|
| 4 GB | `llama3.2:3b`, `phi3:mini` |
| 8 GB | `llama3.1:8b`, `mistral:7b` |
| 16 GB | `llama3.1:70b-q4`, `qwen2.5:14b` |

### vLLM (production / higher throughput)

Requires NVIDIA GPU with ≥16 GB VRAM (or use `--quantization awq` for 8 GB).

```bash
pip install vllm
bash scripts/start_vllm.sh --model mistralai/Mistral-7B-Instruct-v0.2

# Then in .env:
LLM_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8000/v1
```

vLLM uses continuous batching and PagedAttention for significantly higher throughput than Ollama under concurrent load.

---

## Voice

Voice requires additional system packages:

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg libsndfile1 espeak-ng portaudio19-dev

# macOS
brew install ffmpeg portaudio espeak
```

| Component | Library | Notes |
|---|---|---|
| Speech-to-text | [Whisper](https://github.com/openai/whisper) | Fully offline, supports 99 languages |
| Text-to-speech | pyttsx3 + espeak-ng | Offline, no API key |
| Microphone capture | sounddevice | Energy-based VAD, configurable silence threshold |

Voice is automatically disabled in Docker (no `/dev/snd` by default).

---

## Docker

```bash
# Build
docker build -t virtual-assistant .

# Run (Ollama must be accessible from the container)
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  virtual-assistant

# Full stack with Docker Compose
docker compose up -d ollama
docker compose run --rm assistant bash scripts/setup_ollama.sh
docker compose run -it --rm assistant chat
```

---

## Development

```bash
# Run tests
make test

# With coverage
make test-cov

# Lint + format
make lint
make format

# Type check
make typecheck
```

### Project Structure

```
virtual-assistant/
├── agents/
│   ├── base_agent.py          # Abstract base (SOLID OCP/LSP/ISP)
│   ├── code_agent.py          # Write / review / execute code
│   ├── news_agent.py          # RSS + topic news aggregation
│   ├── search_agent.py        # Web search, Wikipedia, calculator
│   ├── document_agent.py      # Docling + vector store Q&A
│   └── orchestrator.py        # LangGraph multi-agent router
├── config/
│   └── settings.py            # Pydantic-settings with validation
├── core/
│   ├── llm_manager.py         # Ollama / vLLM factory
│   ├── memory/
│   │   └── conversation_memory.py   # Window + persistent memory
│   └── voice/
│       ├── speech_to_text.py  # Whisper STT + mic listener
│       └── text_to_speech.py  # pyttsx3 TTS + streaming synth
├── document_processing/
│   ├── docling_processor.py   # Docling parser → DocumentChunk
│   ├── vector_store.py        # ChromaDB store + search + references
│   └── document_manager.py    # High-level façade
├── tools/
│   ├── code_tools.py          # CodeWriter, CodeReviewer, Executor
│   ├── news_tools.py          # Headlines, TopicNews, RSSFeed
│   ├── search_tools.py        # DuckDuckGo, Wikipedia, WebFetch, Calc
│   └── document_tools.py      # Ingest, Search, List, Get, Delete
├── tests/
│   ├── conftest.py            # Shared fixtures (LLM mock autouse)
│   ├── test_agents.py         # Agent unit tests
│   ├── test_document_processing.py
│   ├── test_tools.py
│   ├── test_voice.py
│   └── test_integration.py    # End-to-end pipeline tests
├── scripts/
│   ├── install.sh             # Dev environment setup
│   ├── setup_ollama.sh        # Pull Ollama models
│   └── start_vllm.sh          # Launch vLLM server
├── main.py                    # REPL + voice loop
├── cli.py                     # Typer CLI entry point
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml             # pytest, ruff, black, mypy config
├── Makefile
└── requirements.txt
```

---

## Configuration Reference

All settings can be set via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `vllm` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Chat model for Ollama |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compat URL |
| `VLLM_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | vLLM model name |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `VOICE_ENABLED` | `true` | Enable voice I/O |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embeddings |
| `CHUNK_SIZE` | `512` | Document chunk size (chars) |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `VECTOR_STORE_PATH` | `./data/vector_store` | ChromaDB persistence path |
| `UPLOADS_PATH` | `./data/uploads` | Document upload directory |
| `AGENT_MAX_ITERATIONS` | `15` | Max ReAct loop steps |
| `NEWS_MAX_ARTICLES` | `10` | Headlines per request |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## License

MIT
