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
