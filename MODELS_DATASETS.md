# Models and Datasets Selection

## 1. Selected Models
Primary:
- Qwen3.5-0.8B

Baselines:
- SmolLM2-360M-Instruct
- Qwen3-0.6B

Teacher candidate (optional, larger than target):
- Qwen3-4B Instruct (for synthetic data generation and filtering)

## 2. Why These Models
- They are recent small-model families with stronger instruction behavior.
- They are practical for iterative fine-tuning with LoRA/QLoRA.
- They provide clean A/B baselines across similar architecture families.

## 3. Dataset Strategy
### 3.1 Start Small First
Start with a small curated dataset (10k-30k samples) to test viability:
- Faster training loops.
- Lower compute risk.
- Faster error analysis and data cleaning.

Recommended small-start composition:
- 40% curated icon/shape prompts.
- 30% simple illustration prompts.
- 20% chart/diagram prompts.
- 10% hard edge cases (long prompts, multi-object scenes).

### 3.2 Medium Expansion
Move to 50k-200k after feasibility passes:
- Add style diversity.
- Add compositional prompts.
- Add synthetic data filtered by strict validators.

### 3.3 Large-Scale Expansion
Use mega datasets after medium-stage validation:
- OmniSVG / MMSVG collections.
- Additional large open SVG corpora with compatible licenses.

## 4. OmniSVG and MMSVG Notes
Pros:
- Large volume and diversity.
- Useful fields for instruction tuning (`description`, `keywords`, `detail`, `svg`).

Risks:
- License and usage constraints must be reviewed per dataset.
- Preprocessing choices (for example simplification/resizing) may bias output style.

Recommendation:
- Use OmniSVG data after proving pipeline viability with the smaller starter dataset.
- Keep source and license metadata in each sample.

## 5. Candidate Datasets to Evaluate
- OmniSVG/MMSVG-Illustration
- OmniSVG/MMSVG-Icon
- starvector/text2svg-stack
- Curated permissive SVG icon repositories

## 6. Dataset Acceptance Checklist
A dataset enters training only if all checks pass:
- Clear license and allowed use.
- Valid SVG parse rate above threshold.
- No unsafe SVG content.
- Prompt-SVG pair quality is acceptable.
- Metadata fields are complete.

## 7. Recommended Execution Order
1. Build and validate small starter dataset (10k-30k).
2. Train Qwen3.5-0.8B LoRA baseline.
3. Compare against SmolLM2-360M and Qwen3-0.6B.
4. Expand to medium dataset (50k-200k).
5. Add OmniSVG-scale data if metrics improve and licensing is confirmed.
