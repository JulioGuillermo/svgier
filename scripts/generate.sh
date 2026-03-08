#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/qwen35_0_8b_lora_bootstrap}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/generated.svg}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINT}")
fi

uv run python -m src.inference.generate_svg \
  --model-name "${MODEL_NAME}" \
  "${CHECKPOINT_ARGS[@]}" \
  --checkpoints-dir "${CHECKPOINTS_DIR}" \
  --output-file "${OUTPUT_FILE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  "$@"
