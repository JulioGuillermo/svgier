#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-0.8B}"
TRAIN_JSONL="${TRAIN_JSONL:-data/processed/bootstrap_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/processed/bootstrap_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen35_0_8b_lora_bootstrap}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-2}"

uv run python -m src.training.train_sft \
  --model-name "${MODEL_NAME}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --val-jsonl "${VAL_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-length "${MAX_LENGTH}" \
  --learning-rate "${LEARNING_RATE}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --epochs "${EPOCHS}" \
  --bf16
