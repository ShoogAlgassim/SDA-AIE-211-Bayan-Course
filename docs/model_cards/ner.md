# Model Card — Bayan Named Entity Recognition Model

## Intended use
Extract Bayan entities such as locations, dates, references, and services.

## Artefact / data versions
- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-mix
- Preprocessing version: 1.2.0
- Data version/snapshot: data/models/bayan_ner.conll

## Metrics
| Metric | Value |
| --- | --- |
| Validation entity F1 | 1.0000 |
| Frozen test entity F1 | 1.0000 |
| Frozen test accuracy | 1.0000 |
| Segmented LOCATION recall | 1.0000 |

## Slice metrics
NER-specific slice table not available in the supplied validation prediction file.

## Behavioural tests
Behavioural templates target classification/sentiment behaviour rather than token-level NER.

## Known limitations
- The reported NER scores are measured on the supplied course split and may not reflect unseen domains or noisier real-world inputs.
- The current evaluation does not include detailed slice metrics by entity type, dialect, or text length.
- Behavioural templates in Lab 6 are designed for classification and sentiment behaviour, not token-level NER.

## Contact / owner
Bayan course project team