# AGENTS.md

## Purpose
This project aims to train and fine-tune a small language model (maximum 1B parameters) to generate coherent, realistic SVG output that matches user requests.

## Environment and Tooling
- Use `uv` as the default tool for Python environment management.
- Use `uv` for dependency installation and project command execution.
- Create and use the default virtual environment at `.venv`.
- Avoid mixing package managers unless explicitly required.

## Project Structure Quick Map
- `src/`: all Python source code.
  - `src/data/`: dataset ingestion, normalization, validation, split building.
  - `src/training/`: training entry points and configs.
  - `src/evaluation/`: evaluation metrics and benchmark scripts.
  - `src/inference/`: generation and repair logic.
- `scripts/`: executable shell scripts for reproducible workflows.
- `data/`: local data workspace (ignored in git).
  - `data/raw/`: downloaded or generated raw datasets.
  - `data/processed/`: cleaned and split datasets for training.
  - `data/metadata/`: reports, rejection logs, data diagnostics.
- `checkpoints/`: training checkpoints (ignored in git).
- `outputs/`: generated artifacts and experiment outputs (ignored in git).
- `logs/` and `wandb/`: run logs and experiment tracking artifacts (ignored in git).
- `docs` in root:
  - `PROPOSAL.md`, `MODELS_DATASETS.md`, `DATA_BOOTSTRAP_PLAN.md`, `PROGRESS.md`.

## Typing Rules
- Any Python code must include explicit type hints for function signatures.
- For dynamically typed languages, add clear type/shape indicators when the language supports it.
- Prefer typed dataclasses or typed config objects for runtime settings.
- Use descriptive variable names when type inference could be ambiguous.

## Core Engineering Rules
- Write all code, comments, docs, commit messages, and examples in English.
- Keep solutions simple and maintainable.
- Prefer small, single-purpose functions.
- If a function becomes complex, split it into smaller functions.
- Avoid excessive nesting.
- Favor explicit logic over clever shortcuts.
- Make behavior deterministic when possible.

## Code Style and Architecture
- Use clear naming for functions, variables, and files.
- Keep modules focused on one responsibility.
- Isolate I/O, data processing, and model logic.
- Add concise docstrings for public functions and classes.
- Validate inputs early and fail with clear error messages.
- Keep configuration in dedicated config files.

## Testing and Quality
- Add unit tests for all critical logic.
- Add data validation tests for dataset parsers and SVG checks.
- Add regression tests for prompt-to-SVG generation behavior.
- Run linting and formatting in CI.

## SVG-Specific Rules
- Enforce valid SVG syntax and XML safety.
- Prefer structured generation constraints over free-form raw text.
- Validate geometry, viewBox, element hierarchy, and style consistency.
- Reject outputs with broken tags, invalid attributes, or unsafe content.

## Model Training Constraints
- Target models <= 1B parameters.
- Optimize for limited context windows.
- Use prompt templates that are short and precise.
- Train with instruction data that maps text requests to clean SVG.
- Track output validity, fidelity to prompt, and rendering quality.

## Documentation Rules
- Keep docs short, task-oriented, and practical.
- Document assumptions, tradeoffs, and known limitations.
- Include reproducible commands for data prep, training, and evaluation.

## Collaboration Rules
- Propose changes with a short rationale.
- Avoid introducing unnecessary dependencies.
- Prefer incremental pull requests with focused scope.
