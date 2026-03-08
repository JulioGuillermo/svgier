#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL="${1:-data/raw/bootstrap_input.jsonl}"
OUTPUT_DIR="${2:-data/processed}"
REPORT_PATH="${3:-data/metadata/bootstrap_data_report.md}"
REJECTIONS_PATH="${4:-data/metadata/bootstrap_rejections.jsonl}"
SEED="${5:-42}"

uv run python src/data/build_bootstrap.py \
  --input-jsonl "${INPUT_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --report-path "${REPORT_PATH}" \
  --rejections-path "${REJECTIONS_PATH}" \
  --seed "${SEED}"
