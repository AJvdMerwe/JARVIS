#!/usr/bin/env bash
# scripts/start_vllm.sh
# ─────────────────────────────────────────────────────────────────────────────
# Launch the vLLM inference server (standalone, foreground process).
#
# vLLM uses the OpenAI-compatible API (/v1/chat/completions, /v1/models).
# The Jarvis app connects to it via langchain-openai's ChatOpenAI.
#
# vLLM advantages
# ────────────────
#   • PagedAttention       — efficient KV-cache paging → higher throughput
#   • Continuous batching  — no idle GPU time between requests
#   • Tensor parallelism   — multi-GPU inference (--tensor-parallel-size N)
#   • Quantisation         — AWQ / GPTQ / FP8 for lower VRAM usage
#   • OpenAI-compatible    — drop-in for any OpenAI API client
#   • Speculative decoding — draft model for faster generation (optional)
#
# Requirements
# ─────────────
#   • CUDA 12+, driver 525+
#   • GPU with ≥ 8 GB VRAM for 7B models (fp16)  |  ≥ 4 GB with AWQ/GPTQ
#   • pip install vllm
#   • HuggingFace token if model is gated: huggingface-cli login
#
# Usage
# ─────
#   bash scripts/start_vllm.sh
#   bash scripts/start_vllm.sh --model Qwen/Qwen2.5-7B-Instruct
#   bash scripts/start_vllm.sh --port 8001 --quantization awq
#   bash scripts/start_vllm.sh --tensor-parallel 2     # 2-GPU inference
#   bash scripts/start_vllm.sh --speculative --draft-model <hf-path>
#   bash scripts/start_vllm.sh --dry-run    # print command without running
#
# After starting, configure Jarvis:
#   LLM_BACKEND=vllm
#   VLLM_BASE_URL=http://localhost:8000/v1
#   VLLM_MODEL=<same model you passed to --model>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

# ── Defaults (all overridable via .env or CLI flags) ─────────────────────────
MODEL="${VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
DTYPE="${VLLM_DTYPE:-auto}"                   # auto | float16 | bfloat16
QUANTIZATION="${VLLM_QUANTIZATION:-}"         # awq | gptq | fp8 | ""
TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"  # number of GPUs
ENABLE_SPECULATIVE="${VLLM_ENABLE_SPECULATIVE:-false}"
DRAFT_MODEL="${VLLM_DRAFT_MODEL:-}"
NUM_SPECULATIVE="${VLLM_NUM_SPECULATIVE_TOKENS:-5}"
DRY_RUN=false
LOG_LEVEL="${VLLM_LOG_LEVEL:-info}"

# ── Source .env if present ────────────────────────────────────────────────────
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
  MODEL="${VLLM_MODEL:-${MODEL}}"
  PORT="${VLLM_PORT:-${PORT}}"
  GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-${GPU_UTIL}}"
  MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${MAX_MODEL_LEN}}"
  DTYPE="${VLLM_DTYPE:-${DTYPE}}"
  QUANTIZATION="${VLLM_QUANTIZATION:-${QUANTIZATION}}"
  TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-${TENSOR_PARALLEL}}"
  ENABLE_SPECULATIVE="${VLLM_ENABLE_SPECULATIVE:-${ENABLE_SPECULATIVE}}"
  DRAFT_MODEL="${VLLM_DRAFT_MODEL:-${DRAFT_MODEL}}"
fi

# ── Parse CLI flags (override .env) ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)            MODEL="$2";            shift 2 ;;
    --port)             PORT="$2";             shift 2 ;;
    --host)             HOST="$2";             shift 2 ;;
    --gpu-util)         GPU_UTIL="$2";         shift 2 ;;
    --max-model-len)    MAX_MODEL_LEN="$2";    shift 2 ;;
    --dtype)            DTYPE="$2";            shift 2 ;;
    --quantization)     QUANTIZATION="$2";     shift 2 ;;
    --tensor-parallel)  TENSOR_PARALLEL="$2";  shift 2 ;;
    --speculative)      ENABLE_SPECULATIVE=true; shift ;;
    --draft-model)      DRAFT_MODEL="$2";      shift 2 ;;
    --num-speculative)  NUM_SPECULATIVE="$2";  shift 2 ;;
    --log-level)        LOG_LEVEL="$2";        shift 2 ;;
    --dry-run)          DRY_RUN=true;          shift ;;
    --help|-h)
      sed -n '3,60p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0 ;;
    *)
      echo -e "${RED}Unknown flag: $1${RESET}"
      echo "Run with --help for usage."
      exit 1 ;;
  esac
done

# ── Banner ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║           Virtual Assistant  ×  vLLM Server             ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Model            ${RESET}: ${MODEL}"
echo -e "  ${BOLD}Host:Port        ${RESET}: ${HOST}:${PORT}  ${DIM}(OpenAI-compatible API)${RESET}"
echo -e "  ${BOLD}GPU memory util  ${RESET}: ${GPU_UTIL}"
echo -e "  ${BOLD}Max context len  ${RESET}: ${MAX_MODEL_LEN} tokens"
echo -e "  ${BOLD}dtype            ${RESET}: ${DTYPE}"
echo -e "  ${BOLD}Tensor parallel  ${RESET}: ${TENSOR_PARALLEL} GPU(s)"
[[ -n "$QUANTIZATION"                  ]] && echo -e "  ${BOLD}Quantisation     ${RESET}: ${QUANTIZATION}"
[[ "$ENABLE_SPECULATIVE" == "true"     ]] && echo -e "  ${BOLD}Speculative      ${RESET}: ${NUM_SPECULATIVE} tokens${DRAFT_MODEL:+  (draft: ${DRAFT_MODEL})}"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if ! python3 -c "import vllm" 2>/dev/null; then
  echo -e "${RED}${BOLD}ERROR: vLLM is not installed.${RESET}"
  echo ""
  echo -e "  Install with:"
  echo -e "    ${BOLD}pip install vllm${RESET}"
  echo ""
  echo -e "  Requirements:"
  echo -e "    • CUDA 12+  (check: nvidia-smi)"
  echo -e "    • PyTorch with CUDA support"
  echo -e "    • GPU with ≥ 8 GB VRAM (fp16)  |  ≥ 4 GB with --quantization awq"
  echo ""
  exit 1
fi
echo -e "  ${GREEN}✓${RESET} vLLM installed"

# Check CUDA
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  CUDA_INFO=$(python3 -c "
import torch
n    = torch.cuda.device_count()
gb   = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1024**3
name = torch.cuda.get_device_name(0)
print(f'{n}× {name}  ({gb:.1f} GB total VRAM)')
" 2>/dev/null || echo "unknown")
  echo -e "  ${GREEN}✓${RESET} CUDA available — ${CUDA_INFO}"
  # Warn if tensor-parallel > device count
  N_GPU=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
  if (( TENSOR_PARALLEL > N_GPU )); then
    echo -e "  ${RED}WARNING${RESET}: --tensor-parallel ${TENSOR_PARALLEL} but only ${N_GPU} GPU(s) detected."
    TENSOR_PARALLEL="${N_GPU}"
    echo -e "  Reducing to ${TENSOR_PARALLEL}."
  fi
else
  echo -e "  ${RED}✗${RESET} CUDA not available — vLLM requires a CUDA GPU."
  echo ""
  echo -e "  ${DIM}For CPU-only inference use Ollama instead: LLM_BACKEND=ollama${RESET}"
  echo ""
  exit 1
fi

# Check port availability
if command -v ss &>/dev/null && ss -tlnp | grep -q ":${PORT} "; then
  # Something is listening — is it already vLLM?
  if curl -sf "http://localhost:${PORT}/v1/models" &>/dev/null; then
    echo -e "  ${GREEN}✓${RESET} vLLM already running on port ${PORT}"
    echo ""
    echo -e "  ${BOLD}.env settings:${RESET}"
    echo -e "    ${DIM}LLM_BACKEND=vllm"
    echo -e "    VLLM_BASE_URL=http://localhost:${PORT}/v1"
    echo -e "    VLLM_MODEL=${MODEL}${RESET}"
    exit 0
  else
    echo -e "  ${RED}✗${RESET} Port ${PORT} is in use by another process."
    echo -e "    Change VLLM_BASE_URL in .env or use: ${BOLD}--port <other_port>${RESET}"
    exit 1
  fi
fi
echo -e "  ${GREEN}✓${RESET} Port ${PORT} is available"
echo ""

# ── Build argument list ───────────────────────────────────────────────────────
CMD=(
  python3 -m vllm.entrypoints.openai.api_server
  --model                   "${MODEL}"
  --host                    "${HOST}"
  --port                    "${PORT}"
  --gpu-memory-utilization  "${GPU_UTIL}"
  --max-model-len           "${MAX_MODEL_LEN}"
  --dtype                   "${DTYPE}"
  --tensor-parallel-size    "${TENSOR_PARALLEL}"
  --trust-remote-code
)

[[ -n "$QUANTIZATION"            ]] && CMD+=(--quantization "${QUANTIZATION}")
[[ "$LOG_LEVEL" == "debug"       ]] && CMD+=(--debug)

if [[ "$ENABLE_SPECULATIVE" == "true" ]]; then
  if [[ -z "$DRAFT_MODEL" ]]; then
    echo -e "  ${RED}WARNING${RESET}: Speculative decoding requires --draft-model <hf-path>."
    echo -e "  Continuing without speculative decoding."
  else
    CMD+=(
      --speculative-model  "${DRAFT_MODEL}"
      --num-speculative-tokens "${NUM_SPECULATIVE}"
    )
  fi
fi

# ── Dry run ───────────────────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "true" ]]; then
  echo -e "${DIM}[DRY RUN] Command that would be executed:${RESET}"
  echo ""
  printf '  %s \\\n' "${CMD[@]}"
  echo ""
  exit 0
fi

# ── Info before blocking ──────────────────────────────────────────────────────
echo -e "${DIM}  Polling endpoint : http://localhost:${PORT}/v1/models"
echo -e "  Model load       : 30–120 s depending on model size"
echo -e "  API format       : OpenAI-compatible  (/v1/chat/completions)${RESET}"
echo ""
echo -e "${DIM}  Configure Jarvis (.env):"
echo -e "    LLM_BACKEND=vllm"
echo -e "    VLLM_BASE_URL=http://localhost:${PORT}/v1"
echo -e "    VLLM_MODEL=${MODEL}${RESET}"
echo ""
echo -e "${BOLD}  Starting vLLM server…  (Ctrl-C to stop)${RESET}"
echo ""
echo -e "${DIM}  Command:"
echo -e "  ${CMD[*]}${RESET}"
echo ""

# ── Launch (blocking) ─────────────────────────────────────────────────────────
exec "${CMD[@]}"
