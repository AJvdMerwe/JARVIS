#!/usr/bin/env bash
# =============================================================================
#  start.sh  —  Virtual Personal Assistant  •  One-shot launcher
# =============================================================================
#
#  Usage:
#    bash start.sh                  # auto-detect best mode, interactive REPL
#    bash start.sh --mode chat      # force interactive text REPL
#    bash start.sh --mode voice     # REPL with voice I/O (Whisper + TTS)
#    bash start.sh --mode api       # FastAPI server on port 8080
#    bash start.sh --mode web       # Web app with auth UI on port 8080
#    bash start.sh --mode docker    # full Docker Compose stack
#    bash start.sh --mode test      # run the full test suite
#
#    bash start.sh --query "What is quantum computing?"   # single-shot query
#    bash start.sh --ingest ./documents/                  # ingest a directory
#    bash start.sh --backend vllm                        # override LLM backend
#    bash start.sh --model llama3.2:3b                   # override Ollama model
#    bash start.sh --fresh                               # re-run full setup first
#
#  Environment (can also be set in .env):
#    LLM_BACKEND   ollama | vllm   (default: ollama)
#    OLLAMA_MODEL  any Ollama tag  (default: llama3.1:8b)
#    API_PORT      integer          (default: 8080)
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[0;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="chat"
BACKEND=""
MODEL=""
QUERY=""
INGEST_PATH=""
FRESH=false
API_PORT="${API_PORT:-8080}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
#  Parse arguments
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)       MODE="$2";        shift 2 ;;
    --backend)    BACKEND="$2";     shift 2 ;;
    --model)      MODEL="$2";       shift 2 ;;
    --query|-q)   QUERY="$2";       shift 2 ;;
    --ingest)     INGEST_PATH="$2"; shift 2 ;;
    --port)       API_PORT="$2";    shift 2 ;;
    --fresh)      FRESH=true;       shift   ;;
    --help|-h)
      echo "Virtual Personal Assistant — start.sh"
      echo ""
      echo "Usage:  bash start.sh [MODE] [OPTIONS]"
      echo ""
      echo "Modes:"
      echo "  chat            Interactive text REPL (default)"
      echo "  voice           REPL with Whisper STT + TTS"
      echo "  api             FastAPI server (REST + WebSocket + SSE)"
  echo "  benchmark       Run inference benchmark (compare backends)"
  echo "  web             Full web app with login UI — open browser to http://localhost:8080/ui/"
      echo "  docker          Full Docker Compose stack"
      echo "  test            Run the test suite (844 tests)"
      echo ""
      echo "Options:"
      echo "  --query  TEXT   Run a single query and exit"
      echo "  --ingest PATH   Ingest a document or directory"
      echo "  --backend NAME  LLM backend: ollama | vllm | sglang"
      echo "  --model  TAG    Ollama model tag (e.g. llama3.2:3b)"
      echo "  --port   INT    API server port (default: 8080)"
      echo "  --fresh         Force re-install of dependencies"
      echo ""
      echo "Examples:"
      echo "  bash start.sh"
      echo "  bash start.sh api --port 9000"
      echo "  bash start.sh --query 'What is SOLID?'"
      echo "  bash start.sh --ingest ./reports/ --mode chat"
      echo "  bash start.sh --backend vllm voice"
      exit 0 ;;
    chat|voice|api|web|sglang|benchmark|docker|test)
                  MODE="$1";        shift   ;;
    *)
      echo -e "${RED}Unknown argument: $1${RESET}"
      echo "Usage: bash start.sh [--mode chat|voice|api|docker|test] [options]"
      exit 1 ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────────────────────────────
print_banner() {
  echo -e "${CYAN}${BOLD}"
  echo "  ╔══════════════════════════════════════════════════════════╗"
  echo "  ║         Virtual Personal Assistant  v1.0                ║"
  echo "  ║  Chat · Code · News · Search · Docs · Finance · Research ║"
  echo "  ╚══════════════════════════════════════════════════════════╝"
  echo -e "${RESET}"
}

log()  { echo -e "${GREEN}▶${RESET}  $*"; }
info() { echo -e "${CYAN}ℹ${RESET}  $*"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
fail() { echo -e "${RED}✗${RESET}  $*"; exit 1; }
ok()   { echo -e "${GREEN}✓${RESET}  $*"; }

# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — Python version check
# ─────────────────────────────────────────────────────────────────────────────
check_python() {
  local python_cmd
  for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
      local ver
      ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
      local major minor
      IFS='.' read -r major minor <<< "$ver"
      if (( major > 3 || (major == 3 && minor >= 10) )); then
        PYTHON="$cmd"
        ok "Python ${ver} found ($cmd)"
        return
      fi
    fi
  done
  fail "Python 3.10+ is required. Please install it from https://python.org"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — Virtual environment
# ─────────────────────────────────────────────────────────────────────────────
setup_venv() {
  local venv_dir="${SCRIPT_DIR}/.venv"

  if [[ "$FRESH" == true && -d "$venv_dir" ]]; then
    log "Removing existing venv for fresh install…"
    rm -rf "$venv_dir"
  fi

  if [[ ! -d "$venv_dir" ]]; then
    log "Creating virtual environment…"
    "$PYTHON" -m venv "$venv_dir"
    ok "Virtual environment created at .venv"
  fi

  # Activate
  # shellcheck source=/dev/null
  source "${venv_dir}/bin/activate"
  PYTHON="python"
  ok "Virtual environment activated"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
install_deps() {
  local req="${SCRIPT_DIR}/requirements.txt"
  local stamp="${SCRIPT_DIR}/.venv/.install_stamp"

  # Skip if already installed and requirements haven't changed (unless --fresh)
  if [[ "$FRESH" == false && -f "$stamp" ]]; then
    local req_hash stamp_hash
    req_hash=$(md5sum "$req" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$req" | cut -d' ' -f1)
    stamp_hash=$(cat "$stamp" 2>/dev/null || echo "")
    if [[ "$req_hash" == "$stamp_hash" ]]; then
      ok "Dependencies already installed (requirements.txt unchanged)"
      return
    fi
  fi

  log "Installing Python dependencies…"
  # Pin setuptools<81 — vllm requires setuptools<81.0.0,>=77.0.3 on Python>3.11
  python -m pip install --upgrade pip wheel "setuptools>=77.0.3,<81.0.0" -q
  python -m pip install -r "$req" -q

  # Write stamp hash
  md5sum "$req" 2>/dev/null | cut -d' ' -f1 > "$stamp" \
    || shasum -a 256 "$req" | cut -d' ' -f1 > "$stamp"

  ok "Dependencies installed"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 — Environment file
# ─────────────────────────────────────────────────────────────────────────────
setup_env() {
  local env_file="${SCRIPT_DIR}/.env"
  local env_example="${SCRIPT_DIR}/.env.example"

  if [[ ! -f "$env_file" ]]; then
    if [[ -f "$env_example" ]]; then
      cp "$env_example" "$env_file"
      ok "Created .env from .env.example"
      info "Edit ${SCRIPT_DIR}/.env to customise your settings"
    else
      warn ".env.example not found — creating minimal .env"
      cat > "$env_file" << 'ENVEOF'
# LLM_BACKEND defaults to ollama only if not already set in environment or .env
LLM_BACKEND="${LLM_BACKEND:-ollama}"
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BACKEND=ollama
OLLAMA_REASONING_MODEL=deepseek-r1:7b
VOICE_ENABLED=false
WHISPER_MODEL=base
VECTOR_STORE_PATH=./data/vector_store
UPLOADS_PATH=./data/uploads
LOG_LEVEL=INFO
LOG_PATH=./data/logs/assistant.log
ENVEOF
    fi
  else
    ok ".env found"
  fi

  # Apply CLI overrides to .env for this session
  if [[ -n "$BACKEND" ]]; then
    export LLM_BACKEND="$BACKEND"
    info "LLM backend overridden → ${BACKEND}"
  fi
  if [[ -n "$MODEL" ]]; then
    export OLLAMA_MODEL="$MODEL"
    info "Ollama model overridden → ${MODEL}"
  fi
  if [[ "$MODE" == "voice" ]]; then
    export VOICE_ENABLED=true
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 5 — Data directories
# ─────────────────────────────────────────────────────────────────────────────
setup_dirs() {
  local dirs=(
    "${SCRIPT_DIR}/data/uploads"
    "${SCRIPT_DIR}/data/vector_store"
    "${SCRIPT_DIR}/data/logs/sessions"
    "${SCRIPT_DIR}/data/eval_reports"
  )
  local created=0
  for d in "${dirs[@]}"; do
    if [[ ! -d "$d" ]]; then
      mkdir -p "$d"
      (( created++ ))
    fi
  done
  if (( created > 0 )); then
    ok "Created ${created} data directory/ies"
  else
    ok "Data directories ready"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 6 — Ollama server check (if ollama backend)
# ─────────────────────────────────────────────────────────────────────────────
ensure_ollama() {
  # Determine backend from .env or override
  local backend="${LLM_BACKEND:-$(grep -m1 '^LLM_BACKEND=' "${SCRIPT_DIR}/.env" 2>/dev/null | cut -d= -f2 | tr -d ' "' || echo 'ollama')}"
  [[ "$backend" != "ollama" ]] && return

  local ollama_url="${OLLAMA_BASE_URL:-http://localhost:11434}"
  local ollama_model="${OLLAMA_MODEL:-llama3.1:8b}"

  # Check if Ollama CLI is installed
  if ! command -v ollama &>/dev/null; then
    warn "Ollama is not installed."
    echo ""
    echo -e "  ${BOLD}Install Ollama:${RESET}"
    echo -e "  ${DIM}curl -fsSL https://ollama.com/install.sh | sh${RESET}"
    echo ""
    echo -e "  Or visit: ${CYAN}https://ollama.com/download${RESET}"
    echo ""
    if [[ "$MODE" != "test" ]]; then
      warn "Continuing without Ollama — LLM calls will fail until it is running."
    fi
    return
  fi

  # Check if server is already running
  if curl -sf "${ollama_url}/api/tags" &>/dev/null; then
    ok "Ollama server running at ${ollama_url}"
  else
    log "Starting Ollama server…"
    ollama serve &>/dev/null &
    OLLAMA_PID=$!

    local waited=0
    while ! curl -sf "${ollama_url}/api/tags" &>/dev/null; do
      sleep 1
      (( waited++ ))
      if (( waited >= 20 )); then
        warn "Ollama server did not start within 20s (PID ${OLLAMA_PID})"
        return
      fi
    done
    ok "Ollama server started (PID ${OLLAMA_PID})"
  fi

  # Check if the configured model is already pulled
  local available_models
  available_models=$(curl -sf "${ollama_url}/api/tags" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print('\n'.join(m['name'] for m in d.get('models',[])))" 2>/dev/null || echo "")

  if echo "$available_models" | grep -qF "${ollama_model%%:*}"; then
    ok "Model '${ollama_model}' is available"
  else
    log "Pulling model '${ollama_model}' (this may take a few minutes)…"
    echo -e "${DIM}  Progress is shown below. Ctrl-C to skip and use a pre-pulled model.${RESET}"
    if ollama pull "$ollama_model"; then
      ok "Model '${ollama_model}' pulled"
    else
      warn "Could not pull '${ollama_model}'. Using whatever is available."
    fi
  fi

  # Pull embedding model if not present
  local embed_model="${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}"
  if echo "$available_models" | grep -qF "${embed_model%%:*}"; then
    ok "Embedding model '${embed_model}' is available"
  else
    log "Pulling embedding model '${embed_model}'…"
    ollama pull "$embed_model" || warn "Could not pull embedding model '${embed_model}'"
  fi

  # Pull reasoning model for DeepResearchAgent if not present
  local reason_model="${OLLAMA_REASONING_MODEL:-deepseek-r1:7b}"
  if echo "$available_models" | grep -qF "${reason_model%%:*}"; then
    ok "Reasoning model '${reason_model}' is available"
  else
    info "Reasoning model '${reason_model}' not found."
    info "DeepResearchAgent ('research' intent) requires it."
    info "Pull manually: ollama pull ${reason_model}"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 7 — System dependency hints
# ─────────────────────────────────────────────────────────────────────────────
check_system_deps() {
  local missing=()

  command -v ffmpeg &>/dev/null || missing+=("ffmpeg")

  if [[ "$MODE" == "voice" ]]; then
    # portaudio is required for sounddevice mic capture
    if ! python -c "import sounddevice" &>/dev/null 2>&1; then
      missing+=("portaudio (for voice: sudo apt-get install portaudio19-dev  OR  brew install portaudio)")
    fi
  fi

  if (( ${#missing[@]} > 0 )); then
    warn "Optional system packages not found: ${missing[*]}"
    echo -e "${DIM}  Install on Ubuntu: sudo apt-get install ${missing[*]%% *}"
    echo -e "  Install on macOS:  brew install ${missing[*]%% *}${RESET}"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Step 8 — Quick import sanity check
# ─────────────────────────────────────────────────────────────────────────────
sanity_check() {
  local result
  result=$(python - << 'PYEOF' 2>&1
import sys
sys.path.insert(0, '.')
errors = []
checks = [
    ("config",       "settings"),
    ("agents",       "Orchestrator"),
    ("document_processing", "DocumentManager"),
    ("tools",        "get_code_tools"),
    ("api.server",   "app"),
]
for mod, attr in checks:
    try:
        m = __import__(mod, fromlist=[attr])
        getattr(m, attr)
    except Exception as e:
        errors.append(f"{mod}.{attr}: {e}")
if errors:
    print("FAIL:" + "|".join(errors))
else:
    print("OK")
PYEOF
)

  if [[ "$result" == "OK" ]]; then
    ok "Import sanity check passed"
  else
    local errs="${result#FAIL:}"
    warn "Some imports failed (non-fatal):"
    IFS='|' read -ra ERR_LIST <<< "$errs"
    for e in "${ERR_LIST[@]}"; do
      echo -e "  ${DIM}• $e${RESET}"
    done
    echo ""
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Launch modes
# ─────────────────────────────────────────────────────────────────────────────

launch_web() {
  echo ""
  echo -e "${BOLD}${CYAN}Starting web application…${RESET}"
  echo -e "${DIM}  Open your browser at: http://localhost:${API_PORT}/ui/"
  echo -e "  Default credentials:  admin / admin123  (change on first login)${RESET}"
  echo ""
  exec uvicorn api.server:app \
    --host 0.0.0.0 \
    --port "${API_PORT}" \
    --reload \
    --log-level info
}


launch_benchmark() {
  echo ""
  echo -e "${BOLD}${CYAN}Running inference benchmark…${RESET}"
  echo -e "${DIM}  Backends: ${BENCHMARK_BACKENDS:-ollama}"
  echo -e "  Results will be saved to: data/benchmarks/${RESET}"
  echo ""
  python -m evaluation.inference_benchmark \
    --backends ${BENCHMARK_BACKENDS:-ollama} \
    --runs "${BENCHMARK_RUNS:-5}" \
    --warmup "${BENCHMARK_WARMUP:-1}"
}

launch_sglang() {
  # ── Mode: "bash start.sh sglang" — launches the SGLang server only ───────
  echo ""
  echo -e "${BOLD}${CYAN}Starting SGLang inference server…${RESET}"
  if ! python -c "import sglang" 2>/dev/null; then
    echo ""
    echo -e "${RED}ERROR: sglang is not installed.${RESET}"
    echo -e "  Run: ${BOLD}pip install sglang[all]${RESET}"
    echo -e "  (Requires CUDA 12+ and PyTorch)"
    exit 1
  fi
  # Derive port from SGLANG_BASE_URL (e.g. http://localhost:11435 → 11435)
  local sglang_port
  sglang_port=$(python -c "
import os; u = os.environ.get('SGLANG_BASE_URL','http://localhost:11435')
try:
    from urllib.parse import urlparse; print(urlparse(u).port or 11435)
except Exception: print(11435)
")
  local sglang_model="${SGLANG_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

  echo -e "${DIM}  Model  : ${sglang_model}"
  echo -e "  Port   : ${sglang_port}"
  echo -e "  URL    : http://localhost:${sglang_port}  (Ollama-compat API)${RESET}"
  echo ""

  # Build arg list
  local sglang_args=(
    --model-path   "${sglang_model}"
    --host         "0.0.0.0"
    --port         "${sglang_port}"
    --trust-remote-code
  )
  [[ -n "${SGLANG_MEM_FRACTION:-}"        ]] && sglang_args+=(--mem-fraction-static "${SGLANG_MEM_FRACTION}")
  [[ -n "${SGLANG_MAX_PREFILL_TOKENS:-}"  ]] && sglang_args+=(--max-prefill-tokens  "${SGLANG_MAX_PREFILL_TOKENS}")
  [[ "${SGLANG_ENABLE_SPECULATIVE:-false}" == "true" ]] && {
    sglang_args+=(--speculative-algo EAGLE)
    [[ -n "${SGLANG_DRAFT_MODEL:-}" ]] && sglang_args+=(--speculative-draft-model-path "${SGLANG_DRAFT_MODEL}")
  }

  exec python -m sglang.launch_server "${sglang_args[@]}"
}

ensure_sglang() {
  # ── Called before launching the app when LLM_BACKEND=sglang ──────────────
  # Derives the URL from the environment and probes /api/tags.
  # Fails hard if SGLang is not reachable — never falls back to Ollama.
  local backend="${LLM_BACKEND:-$(grep -m1 '^LLM_BACKEND=' "${SCRIPT_DIR}/.env" 2>/dev/null | cut -d= -f2 | tr -d ' "' || echo 'ollama')}"
  [[ "$backend" != "sglang" ]] && return 0

  local sglang_url
  sglang_url="${SGLANG_BASE_URL:-$(grep -m1 '^SGLANG_BASE_URL=' "${SCRIPT_DIR}/.env" 2>/dev/null | cut -d= -f2 | tr -d ' "' || echo 'http://localhost:11435')}"

  local probe_url="${sglang_url%/}/api/tags"

  echo -e "${DIM}  Checking SGLang server at ${sglang_url}…${RESET}"

  local waited=0
  local max_wait=10   # seconds — SGLang should already be running
  until curl -sf "${probe_url}" &>/dev/null; do
    sleep 1; (( waited++ ))
    if (( waited >= max_wait )); then
      echo ""
      echo -e "${RED}${BOLD}ERROR: SGLang server not reachable at ${sglang_url}${RESET}"
      echo ""
      echo -e "  LLM_BACKEND=sglang is set but the SGLang server is not running."
      echo -e "  Start it in a separate terminal first:"
      echo ""
      echo -e "    ${BOLD}bash start.sh sglang${RESET}"
      echo ""
      echo -e "  Or start it manually:"
      echo -e "    ${DIM}python -m sglang.launch_server \\"
      echo -e "        --model-path \${SGLANG_MODEL:-meta-llama/Llama-3.1-8B-Instruct} \\"
      echo -e "        --port 11435 --host 0.0.0.0 --trust-remote-code${RESET}"
      echo ""
      echo -e "  ${BOLD}Jarvis will NOT fall back to Ollama when --backend sglang is set.${RESET}"
      echo ""
      exit 1
    fi
    echo -e "  ${DIM}Waiting for SGLang… (${waited}s)${RESET}"
  done

  ok "SGLang server ready at ${sglang_url}"
}

launch_benchmark() {
  echo ""
  echo -e "Running inference benchmark..."
  BACKENDS="${BENCHMARK_BACKENDS:-ollama}"
  SUITES="${BENCHMARK_SUITES:-short,medium}"
  python -c "
import sys, os
sys.path.insert(0, '.')
from core.benchmark import run_full_benchmark, format_report, save_report
backends = os.environ.get('BENCHMARK_BACKENDS','ollama').split(',')
suites   = os.environ.get('BENCHMARK_SUITES','short,medium').split(',')
print('Benchmarking:', backends)
results  = run_full_benchmark(backends=backends, suites=suites, warmup_runs=1)
report   = format_report(results)
print(report)
md, js   = save_report(results)
print('Report:', md)
print('JSON:  ', js)
"
}

launch_chat() {
  echo ""
  echo -e "${BOLD}${CYAN}Starting interactive REPL…${RESET}"
  echo -e "${DIM}  Type :help for commands, :quit to exit${RESET}"
  echo ""
  exec python cli.py chat
}

launch_voice() {
  echo ""
  echo -e "${BOLD}${CYAN}Starting voice-enabled REPL…${RESET}"
  echo -e "${DIM}  Whisper STT + pyttsx3 TTS active${RESET}"
  echo ""
  exec python cli.py chat --voice
}

launch_api() {
  echo ""
  echo -e "${BOLD}${CYAN}Starting FastAPI server on port ${API_PORT}…${RESET}"
  echo -e "${DIM}  REST:      http://localhost:${API_PORT}/chat"
  echo -e "  WebSocket: ws://localhost:${API_PORT}/ws/{session_id}"
  echo -e "  SSE:       http://localhost:${API_PORT}/stream"
  echo -e "  Web UI:    http://localhost:${API_PORT}/ui"
  echo -e "  API docs:  http://localhost:${API_PORT}/docs${RESET}"
  echo ""
  exec uvicorn api.server:app \
    --host 0.0.0.0 \
    --port "${API_PORT}" \
    --reload \
    --log-level info
}

launch_docker() {
  echo ""
  echo -e "${BOLD}${CYAN}Starting Docker Compose stack…${RESET}"

  if ! command -v docker &>/dev/null; then
    fail "Docker is not installed. Visit https://docs.docker.com/get-docker/"
  fi
  if ! docker compose version &>/dev/null 2>&1; then
    fail "Docker Compose is not available. Update Docker Desktop or install the plugin."
  fi

  log "Building image (if needed)…"
  docker compose build --quiet

  log "Starting Ollama service…"
  docker compose up -d ollama

  echo -e "${DIM}  Waiting for Ollama health check…${RESET}"
  local waited=0
  until docker compose exec ollama curl -sf http://localhost:11434/api/tags &>/dev/null; do
    sleep 2; (( waited += 2 ))
    if (( waited >= 60 )); then
      warn "Ollama health check timed out after 60s"
      break
    fi
  done

  log "Pulling models inside container…"
  docker compose run --rm assistant bash scripts/setup_ollama.sh || true

  echo ""
  echo -e "${BOLD}${GREEN}Stack is ready.${RESET}"
  echo -e "${DIM}  Interactive REPL:${RESET}"
  echo -e "    docker compose run -it --rm assistant"
  echo -e "${DIM}  API server:${RESET}"
  echo -e "    docker run -p ${API_PORT}:${API_PORT} virtual-assistant sh -c 'uvicorn api.server:app --host 0.0.0.0 --port ${API_PORT}'"
  echo ""
}

launch_test() {
  echo ""
  echo -e "${BOLD}${CYAN}Running test suite…${RESET}"
  echo ""
  exec python -m pytest tests/ -v --tb=short
}

launch_query() {
  echo ""
  exec python cli.py ask "$QUERY"
}

launch_ingest() {
  echo ""
  local is_dir=false
  [[ -d "$INGEST_PATH" ]] && is_dir=true

  if [[ "$is_dir" == true ]]; then
    exec python cli.py ingest "$INGEST_PATH" --directory
  else
    exec python cli.py ingest "$INGEST_PATH"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
  cd "$SCRIPT_DIR"

  print_banner

  echo -e "${BOLD}Mode: ${CYAN}${MODE}${RESET}$([ -n "$QUERY" ] && echo "  Query: \"${QUERY:0:50}…\"")$([ -n "$INGEST_PATH" ] && echo "  Ingest: ${INGEST_PATH}")"
  echo ""

  # ── Setup steps (always run, fast if already done) ──────────────────────
  check_python
  ensure_sglang

  # Docker mode skips Python setup entirely
  if [[ "$MODE" == "docker" ]]; then
    launch_docker
    exit 0
  fi

  setup_venv
  install_deps
  setup_env
  setup_dirs

  # ── Ollama check (skip for test, docker modes) ───────────────────────────
  if [[ "$MODE" != "test" ]]; then
    ensure_ollama
    check_system_deps
  fi

  # ── Sanity check ─────────────────────────────────────────────────────────
  sanity_check

  # ── Separator ────────────────────────────────────────────────────────────
  echo ""
  echo -e "${DIM}────────────────────────────────────────────────────────${RESET}"
  echo ""

  # ── Single-shot modes ────────────────────────────────────────────────────
  if [[ -n "$QUERY" ]]; then
    launch_query
  fi
  if [[ -n "$INGEST_PATH" ]]; then
    launch_ingest
  fi

  # ── Long-running modes ───────────────────────────────────────────────────
  case "$MODE" in
    chat)   launch_chat   ;;
    voice)  launch_voice  ;;
    api)    launch_api    ;;
    web)       launch_web       ;;
    benchmark) export LLM_BACKEND_SKIP_PROBE=true; launch_benchmark ;;
    test)      export LLM_BACKEND_SKIP_PROBE=true; launch_test      ;;
    *)      fail "Unknown mode: ${MODE}" ;;
  esac
}

main "$@"
