# Model Card — Bayan Topic Classifier

## Intended use
Classify bilingual citizen feedback into Bayan service topics.

## Artefact / data versions
- Model/checkpoint: CAMeL-Lab/bert-base-arabic-camelbert-mix
- Preprocessing version: 1.2.0
- Data version/snapshot: Bayan feedback course dataset

## Metrics
| Metric | Value |
| --- | --- |
| Validation accuracy | 0.8750 |
| 95% CI | [0.8617, 0.8888] |
| English accuracy | 1.0000 |
| Arabic accuracy | 0.7500 |
| Parks accuracy | 0.0000 |

## Slice metrics
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

## Behavioural tests
| Test type | Evidence |
| --- | --- |
| Invariance | 200 supplied templates; runner implemented |
| Directional | 200 supplied templates; requires sentiment predictor |
| MFT | Runner supports explicit MFT cases; no supplied MFT rows found |

## Known limitations
- Arabic performance is lower than English performance in the supplied validation set.
- The `parks` class shows a severe confusion with `roads` and requires targeted retraining and data review.
- Behavioural directional testing is incomplete because no sentiment predictor is provided in the current repository.

## Contact / owner
Bayan course project team