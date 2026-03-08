# SVGier

Small-LLM project to generate coherent and realistic SVG from user prompts.

## Current Status (2026-03-08)
- Project docs and standards are defined.
- `uv` is the required tooling for environment and execution.
- Bootstrap data pipeline is implemented and working.
- First mini dataset bootstrap has been generated.

### Bootstrap Snapshot
- Raw samples: `8117`
- Kept samples: `8053`
- Rejected samples: `64`
- Splits: `7247 train / 402 val / 404 test`

Rejection reasons:
- `xml_parse_failure`: `39`
- `disallowed_tag:foreignObject`: `21`
- `disallowed_tag:script`: `4`

## What Is Already Implemented
- `src/data/build_bootstrap.py`
  - Input normalization.
  - Required-field checks.
  - Basic SVG safety checks.
  - Deduplication.
  - Train/val/test split generation.
  - Report and rejection export.
- `scripts/bootstrap_dataset.sh`
  - Runner script using `uv run python`.

## Data Sources Used in Initial Bootstrap
- `starvector/text2svg-stack`
- `uwunion/instruct_svg`
- `ServiceNow/svg-emoji`

Note:
- `OmniSVG/MMSVG-Icon` is gated and needs `HF_TOKEN` authentication.

## Quick Start
1. Create virtual environment:
```bash
uv venv
```

2. Install dependencies:
```bash
uv pip install datasets huggingface_hub pillow
```

3. Run bootstrap builder (expects `data/raw/bootstrap_input.jsonl`):
```bash
./scripts/bootstrap_dataset.sh
```

## Important Paths
- Docs:
  - `AGENTS.md`
  - `PROPOSAL.md`
  - `MODELS_DATASETS.md`
  - `DATA_BOOTSTRAP_PLAN.md`
  - `PROGRESS.md`
- Code:
  - `src/data/build_bootstrap.py`
  - `scripts/bootstrap_dataset.sh`

## Git and Artifacts
Large datasets and generated artifacts are ignored by default in `.gitignore`:
- `data/raw/`
- `data/processed/`
- `data/metadata/`
- model checkpoints and logs

## Next Milestones
- Add optional `HF_TOKEN` support for gated datasets.
- Add configurable per-source sampling caps.
- Add stricter SVG safety validation.
- Add unit tests for data normalization and SVG validation.
- Prepare first LoRA training script for `Qwen3.5-0.8B`.
