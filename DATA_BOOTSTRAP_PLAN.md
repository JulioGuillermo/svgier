# Data Bootstrap Plan (10k-30k Samples)

## 1. Objective
Create a high-quality starter dataset to validate project feasibility before training on medium or mega datasets.

## 2. Target Outcome
Deliver a dataset of 10k-30k prompt-SVG pairs with:
- High XML/SVG validity.
- Clear prompt-to-image alignment.
- Clean license/provenance metadata.
- Balanced task coverage (icons, illustrations, diagrams).

## 3. Starter Data Sources
## 3.1 Primary Candidate Sources
- `OmniSVG/MMSVG-Icon` (small/structured SVG focus).
- `starvector/text2svg-stack` (text-SVG aligned pairs).
- Small filtered subset from `OmniSVG/MMSVG-Illustration`.

## 3.2 Optional Supplemental Sources
- Curated permissive icon repositories (license verified).
- Internal template-generated SVG tasks for edge cases.

## 4. Proposed Mix (Starter Phase)
- 40% icons and simple symbols.
- 30% illustrations (single or few objects).
- 20% charts/diagrams and geometric compositions.
- 10% hard prompts (multi-object, style constraints, long instructions).

Target sample count:
- Minimum: 10,000
- Preferred: 20,000
- Maximum for phase A: 30,000

## 5. Data Schema (JSONL)
Each record must contain:
- `id`
- `prompt`
- `svg`
- `source`
- `license`
- `split` (`train`, `val`, `test`)
- `quality_flags`
- `category`
- `complexity_level`

## 6. Ingestion Pipeline
1. Load records from selected sources.
2. Convert each sample to common schema.
3. Normalize SVG formatting.
4. Validate XML and SVG structure.
5. Remove unsafe elements and attributes.
6. Deduplicate by SVG hash + normalized prompt hash.
7. Score quality and filter low-quality samples.
8. Build train/val/test split.
9. Export JSONL and metadata report.

## 7. Validation Rules
## 7.1 Hard Reject Conditions
- XML parse failure.
- Missing `<svg>` root element.
- Disallowed tags (for example `script`, `foreignObject`).
- External references that break portability.
- Empty or near-empty visible content.

## 7.2 Quality Thresholds
- Prompt length within defined range.
- SVG node count inside configured limits.
- Render test success.
- No duplicated records above threshold.

## 8. Suggested Thresholds (Initial)
- XML parse success: >= 98%
- SVG validation pass: >= 95%
- Render success: >= 97%
- Near-duplicate rate: <= 5%
- Manual spot-check prompt fidelity: >= 80% acceptable

## 9. Split Strategy
- Train: 90%
- Validation: 5%
- Test: 5%

Apply stratification by:
- `category`
- `complexity_level`
- `source`

## 10. Human Review Pass
Sample 300 records before final freeze:
- 100 random train
- 100 random validation
- 100 random test

Review criteria:
- Prompt fidelity
- Visual coherence
- SVG editability
- Label/metadata quality

## 11. Deliverables
- `data/processed/bootstrap_train.jsonl`
- `data/processed/bootstrap_val.jsonl`
- `data/processed/bootstrap_test.jsonl`
- `data/metadata/bootstrap_data_report.md`
- `data/metadata/bootstrap_rejections.jsonl`

## 12. Exit Criteria for Phase A
Proceed to model training only if all conditions pass:
- Thresholds from Section 8 met.
- Human review pass completed.
- License/provenance metadata complete for all records.

## 13. Next Step After Approval
Run first LoRA fine-tune on `Qwen3.5-0.8B` using this starter dataset, then compare against:
- `SmolLM2-360M-Instruct`
- `Qwen3-0.6B`
