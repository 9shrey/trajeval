#!/usr/bin/env bash
# Serve a model with vLLM's OpenAI-compatible API and Hermes tool-call parsing, sized for a 4 GB GPU.
#   scripts/serve_vllm.sh <hf-model> <served-name> [port]
# VLLM_PY / VLLM_BIN point at a venv with vllm installed (vLLM pins its own torch).
set -euo pipefail
MODEL=$1; NAME=$2; PORT=${3:-8100}
VLLM_BIN=${VLLM_BIN:-$HOME/.cache/nanoserve-vllm-venv/bin/vllm}
exec "$VLLM_BIN" serve "$MODEL" --port "$PORT" --served-model-name "$NAME" --dtype half \
  --max-model-len "${MAX_LEN:-6144}" --gpu-memory-utilization "${GPU_UTIL:-0.89}" \
  --max-num-seqs "${MAX_SEQS:-16}" --max-num-batched-tokens "${MAX_BATCHED:-2048}" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --enable-auto-tool-choice --tool-call-parser hermes ${EXTRA:-}
