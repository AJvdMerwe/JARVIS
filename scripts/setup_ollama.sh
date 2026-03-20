#!/usr/bin/env bash
# scripts/setup_ollama.sh
# ────────────────────────────────────────────────────────────────────────────
# Pull required Ollama models and verify the server is running.
# Usage:  bash scripts/setup_ollama.sh [--chat-model llama3.1:8b] [--embed-model nomic-embed-text]
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CHAT_MODEL="${CHAT_MODEL:-llama3.1:8b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chat-model)  CHAT_MODEL="$2"; shift 2 ;;
    --embed-model) EMBED_MODEL="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "═══════════════════════════════════════════════"
echo " Virtual Assistant – Ollama setup"
echo " Chat  model : $CHAT_MODEL"
echo " Embed model : $EMBED_MODEL"
echo " Ollama URL  : $OLLAMA_BASE_URL"
echo "═══════════════════════════════════════════════"

# ── Check Ollama is installed ────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "❌  Ollama is not installed."
  echo "    Install it from: https://ollama.com/download"
  exit 1
fi

# ── Start server if not running ──────────────────────────────────────────────
if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" &>/dev/null; then
  echo "⟳  Ollama server not running – starting in background…"
  ollama serve &
  OLLAMA_PID=$!
  echo "   PID: $OLLAMA_PID"
  # Wait for it to come up
  for i in {1..15}; do
    sleep 1
    curl -sf "${OLLAMA_BASE_URL}/api/tags" &>/dev/null && break
    echo "   Waiting for Ollama ($i/15)…"
  done
fi

if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" &>/dev/null; then
  echo "❌  Ollama server is not responding. Check logs."
  exit 1
fi

echo "✓  Ollama server is running."

# ── Pull models ──────────────────────────────────────────────────────────────
echo ""
echo "⟳  Pulling chat model: $CHAT_MODEL"
ollama pull "$CHAT_MODEL"
echo "✓  Chat model ready."

echo ""
echo "⟳  Pulling embedding model: $EMBED_MODEL"
ollama pull "$EMBED_MODEL"
echo "✓  Embedding model ready."

echo ""
echo "═══════════════════════════════════════════════"
echo "✅  Setup complete!"
echo ""
echo "   To start the assistant:"
echo "     python cli.py chat"
echo ""
echo "   To use a different model:"
echo "     LLM_BACKEND=ollama OLLAMA_MODEL=llama3.2:3b python cli.py chat"
echo "═══════════════════════════════════════════════"
