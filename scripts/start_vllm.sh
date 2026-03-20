#!/usr/bin/env bash
# scripts/start_vllm.sh
# ────────────────────────────────────────────────────────────────────────────
# Launch a vLLM OpenAI-compatible inference server.
# vLLM provides significantly higher throughput than Ollama for multi-user
# or production workloads, using continuous batching and PagedAttention.
#
# Requirements: GPU with ≥16GB VRAM for 7B models (fp16).
#               Use --quantization awq for 4-bit (≥8GB VRAM).
#
# Usage:
#   bash scripts/start_vllm.sh
#   bash scripts/start_vllm.sh --model Qwen/Qwen2.5-7B-Instruct --port 8001
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"       # e.g. "awq" or "gptq"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
DTYPE="${VLLM_DTYPE:-auto}"                # auto | float16 | bfloat16

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)           MODEL="$2"; shift 2 ;;
    --port)            PORT="$2"; shift 2 ;;
    --quantization)    QUANTIZATION="$2"; shift 2 ;;
    --max-model-len)   MAX_MODEL_LEN="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "═══════════════════════════════════════════════"
echo " Virtual Assistant – vLLM server"
echo " Model        : $MODEL"
echo " Port         : $PORT"
echo " Max ctx len  : $MAX_MODEL_LEN"
echo " GPU util     : $GPU_UTIL"
if [[ -n "$QUANTIZATION" ]]; then
  echo " Quantisation : $QUANTIZATION"
fi
echo "═══════════════════════════════════════════════"

# ── Verify vllm installed ────────────────────────────────────────────────────
if ! python -c "import vllm" 2>/dev/null; then
  echo "❌  vLLM is not installed."
  echo "    Install with: pip install vllm"
  exit 1
fi

# ── Build command ────────────────────────────────────────────────────────────
CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --dtype "$DTYPE"
  --trust-remote-code
)

if [[ -n "$QUANTIZATION" ]]; then
  CMD+=(--quantization "$QUANTIZATION")
fi

echo ""
echo "⟳  Starting vLLM server…"
echo "   Command: ${CMD[*]}"
echo ""
echo "   Once running, set these in .env:"
echo "     LLM_BACKEND=vllm"
echo "     VLLM_BASE_URL=http://localhost:${PORT}/v1"
echo "     VLLM_MODEL=${MODEL}"
echo ""

exec "${CMD[@]}"
