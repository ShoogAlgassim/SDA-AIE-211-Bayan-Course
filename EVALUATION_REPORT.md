# Bayan Evaluation Report

## Management Headline

Overall validation performance is strong for English examples, but Arabic performance is lower at 75%, showing a clear language-level gap.

The largest model weakness is the `parks` class, which has 0% accuracy while all other classes reach 100%; short inputs also perform slightly worse than medium-length inputs.

## Bootstrap Confidence Interval

| Metric | Value |
| --- | --- |
| Validation accuracy | 0.8750 |
| 95% bootstrap CI lower | 0.8617 |
| 95% bootstrap CI upper | 0.8888 |
| Validation rows | 2400 |
| Validation errors | 300 |

The bootstrap interval reports uncertainty around the observed validation accuracy rather than treating the point estimate as exact.

## Sliced Evaluation

| Slice type | Slice | N | Accuracy | Note |
| --- | --- | --- | --- | --- |
| language | ar | 1200 | 0.7500 |  |
| language | en | 1200 | 1.0000 |  |
| dialect | MSA | 1200 | 0.7500 |  |
| dialect | unknown | 1200 | 1.0000 |  |
| class | billing | 300 | 1.0000 |  |
| class | digital_services | 300 | 1.0000 |  |
| class | licensing | 300 | 1.0000 |  |
| class | lighting | 300 | 1.0000 |  |
| class | parks | 300 | 0.0000 |  |
| class | roads | 300 | 1.0000 |  |
| class | waste | 300 | 1.0000 |  |
| class | water | 300 | 1.0000 |  |
| length | medium | 1646 | 0.8894 |  |
| length | short | 754 | 0.8435 |  |

Slices with small sample sizes are explicitly flagged rather than treated as equally precise.

## Behavioural Evaluation

| Test type | Evidence |
| --- | --- |
| Invariance | 200 supplied templates; runner implemented |
| Directional | 200 supplied templates; requires sentiment predictor |
| MFT | Runner supports explicit MFT cases; no supplied MFT rows found |

The supplied behavioural file contains 200 invariance and 200 directional templates. No sentiment model or explicit MFT rows are supplied in the repository, so unsupported behavioural pass rates are not fabricated.

## Manual Error Review

120 sampled validation errors were manually reviewed.

All 120 reviewed errors showed the same primary confusion:

`parks → roads`

This led to the addition of:

`Class confusion / under-learned class`

as an explicit error-taxonomy category.

### Error-category Histogram

| Category | Count |
| --- | ---: |
| Class confusion / under-learned class | 120 |

### Top 3 Prioritised Fixes

1. Strengthen the `parks` class through label-mapping verification, balanced training, and additional representative examples.
   - Maximum observed validation accuracy improvement if all 300 current errors are corrected: approximately +12.5 percentage points.

2. Add hard contrastive examples for `parks` versus `roads`.
   - Exact metric improvement must be measured after retraining.

3. Add Arabic orthographic and noisy-text augmentation.
   - Expected to improve robustness, but the effect is likely smaller than fixing the main class-confusion problem.

## Full Error Taxonomy

# Error Taxonomy Starter

1. Label ambiguity
2. Arabic orthographic variation
3. Dialect or code-switching
4. Entity boundary or clitic alignment
5. Long-context truncation
6. Retrieval relevance mismatch
7. Preprocessing or serving skew
8. Annotation defect


9. Class confusion / under-learned class

## Lab 6 — Manual Error Review

A random sample of 120 validation errors was read manually.

### Error-category histogram

- Class confusion / under-learned class: 120/120

All reviewed errors had the same primary confusion:

`parks → roads`

The errors occurred even in clear Arabic examples containing strong park-related signals such as `حديقة`, `ألعاب الأطفال`, `الري`, and park accessibility complaints.

Some examples also contained secondary Arabic orthographic variation, elongation, emoji, or PII placeholders, but the same error was observed on clean examples. Therefore, these were not considered the primary cause.

### Top 3 prioritised fixes

1. Strengthen the `parks` class through label-mapping verification, class-balanced training, and additional representative examples.
   - Maximum observed validation accuracy improvement if all 300 current `parks → roads` errors are corrected: approximately +12.5 percentage points.

2. Add hard contrastive examples for `parks` versus `roads`, especially maintenance and accessibility complaints with overlapping location vocabulary.
   - Expected improvement should be measured after retraining; no unsupported exact delta is claimed.

3. Add Arabic orthographic and noisy-text augmentation for variants such as misspellings, elongation, emoji, and masked PII.
   - Expected to provide a smaller robustness improvement than fixing the main class-confusion problem.


## Model Cards

Three model cards are generated in:

`docs/model_cards/`

- Topic classifier
- NER model
- Bilingual semantic-search model

Known limitations must be reviewed and written by hand before final submission.
