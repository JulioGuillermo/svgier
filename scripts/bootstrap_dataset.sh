#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Usage: ./scripts/bootstrap_dataset.sh [INPUT_JSONL] [OUTPUT_DIR] [REPORT_PATH] [REJECTIONS_PATH] [SEED]

If INPUT_JSONL does not exist, the script auto-downloads a raw bootstrap dataset.

Environment overrides:
  TEXT2SVG_LIMIT      default: 5000
  INSTRUCT_SVG_LIMIT  default: 3500
  SVG_EMOJI_LIMIT     default: 2500
USAGE
  exit 0
fi

INPUT_JSONL="${1:-data/raw/bootstrap_input.jsonl}"
OUTPUT_DIR="${2:-data/processed}"
REPORT_PATH="${3:-data/metadata/bootstrap_data_report.md}"
REJECTIONS_PATH="${4:-data/metadata/bootstrap_rejections.jsonl}"
SEED="${5:-42}"

TEXT2SVG_LIMIT="${TEXT2SVG_LIMIT:-5000}"
INSTRUCT_SVG_LIMIT="${INSTRUCT_SVG_LIMIT:-3500}"
SVG_EMOJI_LIMIT="${SVG_EMOJI_LIMIT:-2500}"

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "[bootstrap] raw input not found: ${INPUT_JSONL}"
  echo "[bootstrap] downloading raw bootstrap dataset..."
  uv run python -m src.data.download_bootstrap_raw \
    --output-jsonl "${INPUT_JSONL}" \
    --text2svg-limit "${TEXT2SVG_LIMIT}" \
    --instruct-svg-limit "${INSTRUCT_SVG_LIMIT}" \
    --svg-emoji-limit "${SVG_EMOJI_LIMIT}"
fi

uv run python src/data/build_bootstrap.py \
  --input-jsonl "${INPUT_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --report-path "${REPORT_PATH}" \
  --rejections-path "${REJECTIONS_PATH}" \
  --seed "${SEED}"
