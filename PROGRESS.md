# Progress Tracker

## Rules
- Update this file after each meaningful task.
- Use checkboxes to track status.
- Keep entries short and action-oriented.

Legend:
- `[x]` done
- `[ ]` pending

## Completed
- [x] Create `AGENTS.md` with engineering rules (English-only code/docs/comments, simplicity, low nesting).
- [x] Add `uv` as mandatory environment and execution tool in `AGENTS.md`.
- [x] Add project structure quick map to `AGENTS.md` (data paths, checkpoints, outputs, logs).
- [x] Create default virtual environment with `uv venv` (`.venv`).
- [x] Create `PROPOSAL.md` with phased strategy (small dataset first, then medium, then mega datasets).
- [x] Create `MODELS_DATASETS.md` with selected models and dataset strategy.
- [x] Create `DATA_BOOTSTRAP_PLAN.md` with measurable data quality gates.
- [x] Implement bootstrap builder script at `src/data/build_bootstrap.py`.
- [x] Implement runner script at `scripts/bootstrap_dataset.sh`.
- [x] Update runner script to use `uv run python`.
- [x] Install dependencies with `uv` (`datasets`, `huggingface_hub`, `pillow`).
- [x] Build first mini raw dataset at `data/raw/bootstrap_input.jsonl`.
- [x] Generate processed splits and metadata reports.
- [x] Review and strengthen root `.gitignore` for ML workflow and large dataset exclusion.
- [x] Create and update `README.md` with current project status and reproducible commands.

## Current Outputs
- [x] `data/processed/bootstrap_train.jsonl`
- [x] `data/processed/bootstrap_val.jsonl`
- [x] `data/processed/bootstrap_test.jsonl`
- [x] `data/metadata/bootstrap_data_report.md`
- [x] `data/metadata/bootstrap_rejections.jsonl`

## In Progress
- [ ] Integrate `OmniSVG/MMSVG-Icon` (requires HF authentication token).

## Next Tasks
- [ ] Add optional `HF_TOKEN` support in data bootstrap flow.
- [ ] Add per-source sampling caps and config file for dataset mixing.
- [ ] Add stricter SVG safety validation (`on*` handlers, external refs, style sanitization).
- [ ] Add basic unit tests for dataset normalization and SVG validation.
- [ ] Create first LoRA training script for `Qwen3.5-0.8B`.
- [ ] Create evaluation script for SVG validity + render success + prompt fidelity.
