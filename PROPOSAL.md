# Project Proposal: Small LLM for Coherent SVG Generation

## 1. Goal
Build a practical training pipeline for a small LLM (<=1B parameters) that converts user requests into coherent, realistic, and valid SVG.

## 2. Model Direction
Primary model:
- Qwen3.5-0.8B

Support models:
- SmolLM2-360M-Instruct (cheap baseline and fast iteration)
- Qwen3-0.6B (ablation baseline)

Why this direction:
- Better recent small-model quality than older 2023 baselines.
- Good ecosystem support for LoRA/QLoRA workflows.
- Fits the project size constraint.

## 3. Delivery Strategy: Small First, Then Scale
### 3.1 Phase A: Feasibility with Small Dataset
Objective:
- Validate that the project is technically viable before investing in large-scale training.

Scope:
- 10k-30k high-quality prompt-SVG pairs.
- Fast training cycles, strict validation, quick error analysis.

Exit criteria:
- High SVG parse success.
- Clear prompt fidelity improvement over base model.
- Stable generation without frequent broken tags.

### 3.2 Phase B: Medium Scale
- Expand to 50k-200k samples.
- Add broader style and composition coverage.
- Improve alignment with rejection sampling and rule-based quality filters.

### 3.3 Phase C: Mega Dataset Training
- Use large sets such as OmniSVG collections.
- Focus on robustness, diversity, and edge-case coverage.
- Keep license metadata and provenance for every sample.

## 4. Dataset Plan
### 4.1 Priority Order
1. Small curated starter set (internal or filtered public subset).
2. Medium merged set (curated + synthetic + permissive public data).
3. Large-scale sets (including OmniSVG datasets where license and terms fit the use case).

### 4.2 Data Mix
Use a mix of:
- Human or template-authored prompt-SVG pairs.
- Curated public SVG sources with clear licenses.
- Synthetic pairs generated and filtered by validators.
- Negative samples (invalid SVG) for quality scoring.

### 4.3 Required Metadata
Every sample must include:
- `id`
- `prompt`
- `svg`
- `source`
- `license`
- `split`
- `quality_flags`

## 5. Training Approach
### 5.1 SFT First
- Fine-tune on prompt -> SVG.
- Keep prompt templates short and consistent.
- Normalize SVG before tokenization.

### 5.2 Quality Alignment
- Add rule-based score for:
  - XML validity
  - SVG sanity checks
  - prompt fidelity
  - render success
- Apply rejection sampling and optionally DPO.

### 5.3 Inference Guardrails
- Validate generated SVG.
- Run repair pass for small structural issues.
- Reject unsafe or malformed output.

## 6. Evaluation Plan
Automatic metrics:
- XML parse success rate.
- SVG validity rate.
- Render success rate.
- Prompt-SVG semantic similarity.

Human review:
- Fidelity to user request.
- Visual coherence.
- Structural editability.

## 7. Tech Stack
- Python: data pipeline, training, evaluation, inference.
- PyTorch + Transformers + PEFT + Accelerate.
- `lxml` for validation checks.
- `resvg` or CairoSVG for render testing.

## 8. Project Structure
```text
project/
  AGENTS.md
  PROPOSAL.md
  MODELS_DATASETS.md
  src/
    data/
    training/
    evaluation/
    inference/
  scripts/
  tests/
  data/
```

## 9. Initial Milestones
1. Build validator + schema checks and create a 10k-30k starter dataset.
2. Train first LoRA run on Qwen3.5-0.8B.
3. Evaluate quality and failure cases.
4. Scale to medium dataset only if feasibility metrics pass.
