# Model Card — Bayan Bilingual Semantic Search

## Intended use
Retrieve similar historical Bayan cases in Arabic and English.

## Artefact / data versions
- Model/checkpoint: intfloat/multilingual-e5-base
- Preprocessing version: 1.2.0
- Data version/snapshot: 20,000 Bayan historical cases

## Metrics
| Metric | Value |
| --- | --- |
| Recall@10 without reranking | 0.0385 |
| MRR@10 without reranking | 0.0059 |
| Recall@10 with reranking | 0.0077 |
| MRR@10 with reranking | 0.0013 |
| No-answer correctness | 20/20 |

## Slice metrics
| Slice | Recall@10 |
| --- | --- |
| Arabic | 0.0167 |
| English | 0.0000 |
| Cross-lingual gap | 0.0167 |

## Behavioural tests
No-answer threshold evaluation: 20/20 correct empty results.

## Known limitations
- Retrieval Recall@10 and MRR@10 are below the course targets on the supplied labelled query set.
- The corpus contains many exact duplicate cases that are not included in the provided relevance labels, which affects measured retrieval metrics.
- Reranking did not improve the measured retrieval metrics in the current evaluation and requires further tuning.

## Contact / owner
Bayan course project team