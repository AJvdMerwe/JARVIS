##!/usr/bin/env bash
## scripts/start_vllm.sh
## ─────────────────────────────────────────────────────────────────────────────
## Launch vLLM in a Docker container following NVIDIA's recommended approach.
## Reference: https://build.nvidia.com/spark/vllm/instructions
##
## GPU detection
## ─────────────
## This script automatically detects the installed GPU architecture and selects
## the appropriate Docker image:
##
##   GX10 / GB10 (Blackwell consumer & DGX Spark)
##     CUDA compute 10.x  →  nvcr.io/nvidia/cuda-based vLLM image with
##                            CUDA 12.8+ and Blackwell-optimised kernels.
##     The DGX Spark GB10 superchip (Project DIGITS) uses this path.
##     GeForce RTX 5000-series (GX10) also uses this path.
##
##   GB20 / GH200 / Grace Blackwell / Grace Hopper
##     Compute 9.0–10.x  →  NGC enterprise image with NVLink/NVMe optimisation.
##
##   Standard (Ada Lovelace, Hopper, Ampere — RTX 3000/4000, A100, H100)
##     Compute 8.x  →  vllm/vllm-openai:latest  (official vLLM Docker Hub image)
##
## Container behaviour
## ───────────────────
##   • Mounts ~/.cache/huggingface so models are cached on the host
##   • Uses --ipc=host for multi-worker shared memory
##   • Passes HF_TOKEN / HUGGING_FACE_HUB_TOKEN for gated models
##   • Sets VLLM_WORKER_MULTIPROC_METHOD=spawn on Blackwell
##   • Maps container port 8000 → host PORT (default 8000)
##
## Jarvis integration
## ──────────────────
##   LLM_BACKEND=vllm
##   VLLM_BASE_URL=http://localhost:<PORT>/v1
##   VLLM_MODEL=<model-id>
##
## Usage
## ─────
##   bash scripts/start_vllm.sh
##   bash scripts/start_vllm.sh --model Qwen/Qwen2.5-7B-Instruct
##   bash scripts/start_vllm.sh --port 8001
##   bash scripts/start_vllm.sh --quantization awq
##   bash scripts/start_vllm.sh --tensor-parallel 2
##   bash scripts/start_vllm.sh --speculative --draft-model <hf-path>
##   bash scripts/start_vllm.sh --image custom/vllm:tag   # override image
##   bash scripts/start_vllm.sh --no-gpu                  # CPU-only (slow)
##   bash scripts/start_vllm.sh --dry-run                 # print command only
##   bash scripts/start_vllm.sh --stop                    # stop running container
## ─────────────────────────────────────────────────────────────────────────────
#set -euo pipefail
#
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
#
## ── Colours ───────────────────────────────────────────────────────────────────
#RED='\033[0;31m'; YELLOW='\033[0;33m'; GREEN='\033[0;32m'
#CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
#
#log()  { echo -e "  ${DIM}${*}${RESET}"; }
#ok()   { echo -e "  ${GREEN}✓${RESET} ${*}"; }
#warn() { echo -e "  ${YELLOW}⚠${RESET}  ${*}"; }
#fail() { echo -e "\n  ${RED}${BOLD}ERROR: ${*}${RESET}\n"; exit 1; }
#
## ── Docker image registry ─────────────────────────────────────────────────────
## Standard vLLM image (Ampere / Ada / Hopper — compute 7.x–8.x)
#VLLM_IMAGE_STANDARD="vllm/vllm-openai:latest"
#
## NVIDIA NGC image for Blackwell (GB10 / GX10 / GB20 — compute 10.x)
## This is the image recommended at build.nvidia.com/spark/vllm for DGX Spark
## and GeForce RTX 5000 series (Blackwell consumer).
##VLLM_IMAGE_BLACKWELL="nvcr.io/nvidia/ai-enterprise/vllm:25.03"
##VLLM_IMAGE_BLACKWELL="nvcr.io/nvidia/vllm:25.12.post1-py3"
#VLLM_IMAGE_BLACKWELL="nvcr.io/nvidia/vllm:26.03-py3"
#
## Container name for lifecycle management
#CONTAINER_NAME="jarvis-vllm"
#
## ── Defaults (all overridable via .env or CLI flags) ──────────────────────────
#MODEL="${VLLM_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
#PORT="${VLLM_PORT:-8000}"
#GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
#MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
#DTYPE="${VLLM_DTYPE:-auto}"                   # auto | float16 | bfloat16
#QUANTIZATION="${VLLM_QUANTIZATION:-}"         # awq | gptq | fp8 | ""
#TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-1}"  # number of GPUs
#ENABLE_SPECULATIVE="${VLLM_ENABLE_SPECULATIVE:-false}"
#DRAFT_MODEL="${VLLM_DRAFT_MODEL:-}"
#NUM_SPECULATIVE="${VLLM_NUM_SPECULATIVE_TOKENS:-5}"
#HF_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}"
#HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
#CUSTOM_IMAGE=""
#NO_GPU=false
#DRY_RUN=false
#STOP_MODE=false
#DETACH=false           # run container in background (for managed-stack use)
#
## ── Source .env if present ────────────────────────────────────────────────────
#ENV_FILE="${PROJECT_DIR}/.env"
#if [[ -f "$ENV_FILE" ]]; then
#  # shellcheck disable=SC1090
#  set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
#  MODEL="${VLLM_MODEL:-${MODEL}}"
#  PORT="${VLLM_PORT:-${PORT}}"
#  GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-${GPU_UTIL}}"
#  MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-${MAX_MODEL_LEN}}"
#  DTYPE="${VLLM_DTYPE:-${DTYPE}}"
#  QUANTIZATION="${VLLM_QUANTIZATION:-${QUANTIZATION}}"
#  TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL:-${TENSOR_PARALLEL}}"
#  ENABLE_SPECULATIVE="${VLLM_ENABLE_SPECULATIVE:-${ENABLE_SPECULATIVE}}"
#  DRAFT_MODEL="${VLLM_DRAFT_MODEL:-${DRAFT_MODEL}}"
#  HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
#fi
#
## ── Parse CLI flags ────────────────────────────────────────────────────────────
#while [[ $# -gt 0 ]]; do
#  case "$1" in
#    --model)            MODEL="$2";            shift 2 ;;
#    --port)             PORT="$2";             shift 2 ;;
#    --gpu-util)         GPU_UTIL="$2";         shift 2 ;;
#    --max-model-len)    MAX_MODEL_LEN="$2";    shift 2 ;;
#    --dtype)            DTYPE="$2";            shift 2 ;;
#    --quantization)     QUANTIZATION="$2";     shift 2 ;;
#    --tensor-parallel)  TENSOR_PARALLEL="$2";  shift 2 ;;
#    --speculative)      ENABLE_SPECULATIVE=true; shift ;;
#    --draft-model)      DRAFT_MODEL="$2";      shift 2 ;;
#    --num-speculative)  NUM_SPECULATIVE="$2";  shift 2 ;;
#    --hf-cache)         HF_CACHE="$2";         shift 2 ;;
#    --hf-token)         HF_TOKEN="$2";         shift 2 ;;
#    --image)            CUSTOM_IMAGE="$2";     shift 2 ;;
#    --container-name)   CONTAINER_NAME="$2";   shift 2 ;;
#    --no-gpu)           NO_GPU=true;           shift ;;
#    --detach|-d)        DETACH=true;           shift ;;
#    --dry-run)          DRY_RUN=true;          shift ;;
#    --stop)             STOP_MODE=true;        shift ;;
#    --help|-h)
#      sed -n '3,75p' "$0" | grep '^#' | sed 's/^# \?//'
#      exit 0 ;;
#    *)
#      echo -e "${RED}Unknown flag: $1${RESET}"
#      echo "Run with --help for usage."
#      exit 1 ;;
#  esac
#done
#
## ── Stop mode ─────────────────────────────────────────────────────────────────
#if [[ "$STOP_MODE" == "true" ]]; then
#  echo ""
#  echo -e "${BOLD}Stopping vLLM container '${CONTAINER_NAME}'…${RESET}"
#  if docker ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
#    docker stop "${CONTAINER_NAME}" && docker rm "${CONTAINER_NAME}" 2>/dev/null || true
#    ok "Container stopped and removed."
#  else
#    warn "No running container named '${CONTAINER_NAME}' found."
#    docker ps -a --filter "name=${CONTAINER_NAME}" --format "  Status: {{.Status}}" || true
#  fi
#  echo ""
#  exit 0
#fi
#
## ─────────────────────────────────────────────────────────────────────────────
##  GPU Detection — determines Docker image and runtime flags
## ─────────────────────────────────────────────────────────────────────────────
#
#GPU_ARCH="unknown"
#GPU_NAME="unknown"
#GPU_COMPUTE=""
#IS_BLACKWELL=false
#IS_GX10=false
#IS_GB10=false
#
#detect_gpu() {
#  # Requires: nvidia-smi + Python with torch (both available inside Docker too)
#  if ! command -v nvidia-smi &>/dev/null; then
#    warn "nvidia-smi not found — cannot detect GPU architecture."
#    return 0
#  fi
#
#  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs || echo "unknown")
#
#  # Get compute capability via Python (most reliable method)
#  if python3 -c "import torch" 2>/dev/null; then
#    GPU_COMPUTE=$(python3 -c "
#import torch
#if torch.cuda.is_available():
#    major, minor = torch.cuda.get_device_capability(0)
#    print(f'{major}.{minor}')
#else:
#    print('0.0')
#" 2>/dev/null || echo "0.0")
#  else
#    # Fallback: parse compute capability from nvidia-smi
#    GPU_COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | xargs || echo "0.0")
#  fi
#
#  local major
#  major=$(echo "${GPU_COMPUTE}" | cut -d. -f1)
#
#  # ── Architecture classification ────────────────────────────────────────────
#  # Compute capability 10.x = Blackwell (GB10, GB20, GX10, GX20, GH200+)
#  if (( major >= 10 )); then
#    IS_BLACKWELL=true
#
#    # GB10 — Grace Blackwell (DGX Spark / Project DIGITS)
#    # Identified by: "GB10" in name, or "DGX Spark", or "Grace Blackwell"
#    if echo "${GPU_NAME}" | grep -qiE "GB10|DGX Spark|Grace Blackwell|DIGITS"; then
#      IS_GB10=true
#      GPU_ARCH="gb10-blackwell"
#
#    # GX10 — GeForce RTX 5000-series (consumer Blackwell)
#    # RTX 5090 / 5080 / 5070 / 5060 etc.
#    elif echo "${GPU_NAME}" | grep -qiE "RTX 50[0-9][0-9]|GeForce.*50[0-9][0-9]|GX10"; then
#      IS_GX10=true
#      GPU_ARCH="gx10-blackwell"
#
#    # Other Blackwell (GB20, GB100, GB200, H200 Blackwell, etc.)
#    else
#      GPU_ARCH="blackwell"
#    fi
#
#  # Compute 9.x = Hopper (H100, H200) or Grace-Hopper
#  elif (( major >= 9 )); then
#    GPU_ARCH="hopper"
#
#  # Compute 8.9 = Ada Lovelace (RTX 4000 series)
#  # Compute 8.0–8.6 = Ampere (A100, RTX 3000)
#  elif (( major >= 8 )); then
#    GPU_ARCH="ampere-ada"
#
#  # Compute 7.x = Turing/Volta (RTX 2000, V100)
#  elif (( major >= 7 )); then
#    GPU_ARCH="turing-volta"
#
#  else
#    GPU_ARCH="legacy"
#    warn "GPU compute capability ${GPU_COMPUTE} may not be supported by vLLM."
#  fi
#}
#
## Run detection unless --no-gpu
#if [[ "$NO_GPU" != "true" ]]; then
#  detect_gpu
#fi
#
## ── Select Docker image ───────────────────────────────────────────────────────
#select_image() {
#  if [[ -n "$CUSTOM_IMAGE" ]]; then
#    echo "$CUSTOM_IMAGE"
#    return
#  fi
#
#  # GX10 and GB10 both require the Blackwell-optimised NGC image
#  if [[ "$IS_BLACKWELL" == "true" ]]; then
#    echo "${VLLM_IMAGE_BLACKWELL}"
#  else
#    echo "${VLLM_IMAGE_STANDARD}"
#  fi
#}
#
#DOCKER_IMAGE="$(select_image)"
#
## ─────────────────────────────────────────────────────────────────────────────
##  Banner
## ─────────────────────────────────────────────────────────────────────────────
#echo ""
#echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════╗${RESET}"
#echo -e "${BOLD}${CYAN}║       Virtual Assistant  ×  vLLM  (Docker)               ║${RESET}"
#echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════╝${RESET}"
#echo ""
#
#if [[ "$NO_GPU" != "true" ]]; then
#  echo -e "  ${BOLD}GPU detection${RESET}"
#  echo -e "  ${DIM}Name         : ${GPU_NAME}"
#  echo -e "  Architecture : ${GPU_ARCH}${IS_GB10:+  (DGX Spark / Grace Blackwell)}${IS_GX10:+  (GeForce RTX 5000-series)}"
#  echo -e "  Compute cap  : ${GPU_COMPUTE:-not detected}${RESET}"
#  if [[ "$IS_BLACKWELL" == "true" ]]; then
#    echo -e "  ${YELLOW}↳ Blackwell GPU detected — using NGC enterprise image${RESET}"
#  fi
#  echo ""
#fi
#
#echo -e "  ${BOLD}Container configuration${RESET}"
#echo -e "  ${DIM}Image        : ${DOCKER_IMAGE}"
#echo -e "  Container    : ${CONTAINER_NAME}"
#echo -e "  Host port    : ${PORT}  →  container 8000"
#echo -e "  HF cache     : ${HF_CACHE}"
#echo -e "  Model        : ${MODEL}"
#echo -e "  GPU util     : ${GPU_UTIL}"
#echo -e "  Max ctx len  : ${MAX_MODEL_LEN} tokens"
#echo -e "  dtype        : ${DTYPE}"
#echo -e "  Tensor //    : ${TENSOR_PARALLEL} GPU(s)"
#[[ -n "${QUANTIZATION}"             ]] && echo -e "  Quantised    : ${QUANTIZATION}"
#[[ "$ENABLE_SPECULATIVE" == "true"  ]] && echo -e "  Speculative  : ${DRAFT_MODEL:-<no draft model>}"
#[[ -n "${HF_TOKEN}"                 ]] && echo -e "  HF token     : ****** (set)"
#[[ "$DETACH" == "true"              ]] && echo -e "  Mode         : detached (background)"
#echo -e "${RESET}"
#
## ─────────────────────────────────────────────────────────────────────────────
##  Pre-flight checks
## ─────────────────────────────────────────────────────────────────────────────
#
## Docker installed?
#if ! command -v docker &>/dev/null; then
#  fail "Docker is not installed.\n  Install: https://docs.docker.com/engine/install/"
#fi
#ok "Docker installed  ($(docker --version | head -1))"
#
## Docker daemon running?
#if ! docker info &>/dev/null 2>&1; then
#  fail "Docker daemon is not running.\n  Start it: sudo systemctl start docker"
#fi
#ok "Docker daemon running"
#
## NVIDIA Container Toolkit (nvidia-docker)?
#if [[ "$NO_GPU" != "true" ]]; then
#  if ! docker run --rm --gpus all --ipc=host \
#       nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
#    warn "NVIDIA Container Toolkit may not be installed or configured."
#    echo -e "  ${DIM}Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
#    echo -e "  If you need CPU-only mode: add --no-gpu flag${RESET}"
#    echo ""
#    fail "NVIDIA GPU access in Docker is required. See warning above."
#  fi
#  ok "NVIDIA Container Toolkit working"
#fi
#
## HuggingFace cache directory
#mkdir -p "${HF_CACHE}"
#ok "HuggingFace cache: ${HF_CACHE}"
#
## Check if container already running
#if docker ps -q --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
#  echo ""
#  ok "Container '${CONTAINER_NAME}' is already running."
#  echo -e "  ${DIM}Endpoint: http://localhost:${PORT}/v1"
#  echo -e "  Stop:     bash scripts/start_vllm.sh --stop${RESET}"
#  echo ""
#  echo -e "  ${BOLD}.env settings:${RESET}"
#  echo -e "    ${DIM}LLM_BACKEND=vllm"
#  echo -e "    VLLM_BASE_URL=http://localhost:${PORT}/v1"
#  echo -e "    VLLM_MODEL=${MODEL}${RESET}"
#  exit 0
#fi
#
## Pull image if not present (with friendly progress)
#if ! docker image inspect "${DOCKER_IMAGE}" &>/dev/null 2>&1; then
#  echo ""
#  log "Image '${DOCKER_IMAGE}' not found locally — pulling…"
#  if [[ "$IS_BLACKWELL" == "true" ]]; then
#    echo -e "  ${DIM}Note: NGC images require NVIDIA GPU Cloud login."
#    echo -e "  If pull fails: docker login nvcr.io${RESET}"
#    echo -e "  ${DIM}  Username: \$oauthtoken"
#    echo -e "  Password: <your NGC API key from https://ngc.nvidia.com>${RESET}"
#    echo ""
#  fi
#  docker pull "${DOCKER_IMAGE}" || {
#    echo ""
#    if [[ "$IS_BLACKWELL" == "true" ]]; then
#      echo -e "${RED}NGC image pull failed.${RESET}"
#      echo -e "  Login to NGC registry first:"
#      echo -e "    ${BOLD}docker login nvcr.io${RESET}"
#      echo -e "  Username: \$oauthtoken"
#      echo -e "  Password: your NGC API key from https://ngc.nvidia.com/setup/api-key"
#      echo ""
#      echo -e "  Alternatively, use the standard image (may lack Blackwell kernels):"
#      echo -e "    ${BOLD}bash scripts/start_vllm.sh --image ${VLLM_IMAGE_STANDARD}${RESET}"
#    else
#      echo -e "${RED}Image pull failed. Check your internet connection.${RESET}"
#    fi
#    exit 1
#  }
#  ok "Image pulled: ${DOCKER_IMAGE}"
#fi
#
## ─────────────────────────────────────────────────────────────────────────────
##  Build docker run command
## ─────────────────────────────────────────────────────────────────────────────
#
#DOCKER_RUN=(
#  docker run
#  --name           "${CONTAINER_NAME}"
#  --ipc=host
#  --ulimit         memlock=-1
#  --ulimit         stack=67108864
#  -v               "${HF_CACHE}:/root/.cache/huggingface"
#  -p               "${PORT}:8000"
##  --restart        unless-stopped
#)
#
## GPU flags
#if [[ "$NO_GPU" == "true" ]]; then
#  warn "Running in CPU-only mode — performance will be very slow."
#else
#  DOCKER_RUN+=(--gpus all)
#fi
#
## HuggingFace token (required for gated models like Llama 3)
#if [[ -n "$HF_TOKEN" ]]; then
#  DOCKER_RUN+=(-e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
#fi
#
## Blackwell-specific environment variables
#if [[ "$IS_BLACKWELL" == "true" ]]; then
#  # Required on Blackwell to avoid multiprocessing fork issues
#  DOCKER_RUN+=(-e "VLLM_WORKER_MULTIPROC_METHOD=spawn")
#  # Enable Blackwell-specific optimisations
#  DOCKER_RUN+=(-e "VLLM_USE_V1=1")
#  if [[ "$IS_GB10" == "true" ]]; then
#    # DGX Spark / Grace Blackwell — uses unified memory architecture
#    # Adjust shared memory limit for GB10 NVLink fabric
#    DOCKER_RUN+=(--shm-size=16g)
#  fi
#fi
#
## Detach flag (used by managed stack in start.sh)
#if [[ "$DETACH" == "true" ]]; then
#  DOCKER_RUN+=(-d)
#else
#  DOCKER_RUN+=(--rm)   # auto-remove when stopped (foreground mode only)
#fi
#
## ── vLLM server arguments (passed after the image name) ──────────────────────
#VLLM_ARGS=(
#  "${DOCKER_IMAGE}"
#  --model                   "${MODEL}"
#  --host                    "0.0.0.0"
#  --port                    "8000"
#  --gpu-memory-utilization  "${GPU_UTIL}"
#  --max-model-len           "${MAX_MODEL_LEN}"
#  --dtype                   "${DTYPE}"
#  --tensor-parallel-size    "${TENSOR_PARALLEL}"
#  --trust-remote-code
#  --served-model-name       "${MODEL}"
#)
#
#[[ -n "$QUANTIZATION" ]] && VLLM_ARGS+=(--quantization "${QUANTIZATION}")
#
#if [[ "$ENABLE_SPECULATIVE" == "true" ]]; then
#  if [[ -z "$DRAFT_MODEL" ]]; then
#    warn "Speculative decoding requires --draft-model. Skipping."
#  else
#    VLLM_ARGS+=(
#      --speculative-model          "${DRAFT_MODEL}"
#      --num-speculative-tokens     "${NUM_SPECULATIVE}"
#    )
#  fi
#fi
#
## ── Final command ─────────────────────────────────────────────────────────────
#FULL_CMD=("${DOCKER_RUN[@]}" "${VLLM_ARGS[@]}")
#
## ── Dry run ───────────────────────────────────────────────────────────────────
#if [[ "$DRY_RUN" == "true" ]]; then
#  echo -e "${DIM}[DRY RUN] Full docker run command:${RESET}"
#  echo ""
#  printf '  %s \\\n' "${FULL_CMD[@]}"
#  echo ""
#  exit 0
#fi
#
## ─────────────────────────────────────────────────────────────────────────────
##  Launch
## ─────────────────────────────────────────────────────────────────────────────
#echo -e "${DIM}  Model load: 30–120 s depending on model size and storage"
#echo -e "  Logs: docker logs -f ${CONTAINER_NAME}"
#echo -e "  Stop: bash scripts/start_vllm.sh --stop${RESET}"
#echo ""
#echo -e "${DIM}  Configure Jarvis (.env):"
#echo -e "    LLM_BACKEND=vllm"
#echo -e "    VLLM_BASE_URL=http://localhost:${PORT}/v1"
#echo -e "    VLLM_MODEL=${MODEL}${RESET}"
#echo ""
#
#if [[ "$IS_BLACKWELL" == "true" ]]; then
#  echo -e "  ${YELLOW}Blackwell GPU:${RESET} Using NGC enterprise image with compute-10 kernels."
#  if [[ "$IS_GB10" == "true" ]]; then
#    echo -e "  ${CYAN}DGX Spark / Grace Blackwell detected:${RESET} Unified memory optimisations enabled."
#  elif [[ "$IS_GX10" == "true" ]]; then
#    echo -e "  ${CYAN}GeForce RTX 5000-series detected:${RESET} Consumer Blackwell path selected."
#  fi
#  echo ""
#fi
#
#echo -e "${BOLD}  Starting vLLM container…  (Ctrl-C to stop in foreground mode)${RESET}"
#echo ""
#echo -e "${DIM}  Command:"
#printf '    %s \\\n' "${FULL_CMD[@]}"
#echo -e "${RESET}"
#
#exec "${FULL_CMD[@]}"
docker run -it --gpus all -p 8000:8000 nvcr.io/nvidia/vllm:26.03-py3 vllm serve mistralai/Mistral-7B-Instruct-v0.2