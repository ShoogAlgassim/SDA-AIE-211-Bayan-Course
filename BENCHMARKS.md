# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
## Lab 1 — Tokenizer Audit

| Tokenizer  | Arabic Fertility | English Fertility | Arabic p95 | English p95 |
| ---------- | ---------------: | ----------------: | ---------: | ----------: |
| mBERT      |            2.153 |             1.510 |         27 |          25 |
| XLM-R      |            1.672 |             1.434 |         21 |          23 |
| CAMeLBERT  |            1.405 |             2.705 |         20 |          38 |
| DistilBERT |            4.527 |             1.298 |         47 |          21 |

XLM-R provided the most balanced tokenization performance across Arabic and English. CAMeLBERT achieved the best Arabic fertility, but its English fertility and sequence lengths were much higher. DistilBERT performed well on English but poorly on Arabic.

- Golden preprocessing: 25 / 25 passed
- PII masking recall: **60 / 60 = 100%**


## Lab 2 — Attention Diagnostics

- Numerical equivalence with PyTorch: Passed
- Multi-Head Attention output shape: [1, 4, 16]
- Multi-Head Attention weights shape: [1, 2, 4, 4]
- Future attention mass with causal mask: 0.0
- Pad mass without mask: 1.383461833000183
- Pad mass with mask: 0.0


## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:

## Lab 3A — TF-IDF Baseline
- Macro-F1: 1.0000


## Lab 3A — Topic Classification

### TF-IDF + LinearSVC Baseline
- Test Macro-F1: 1.0000

### Grouped Split Integrity
- Train / Validation citizen overlap: 0
- Train / Test citizen overlap: 0
- Validation / Test citizen overlap: 0

### Transformer Classifier
Checkpoint:
`CAMeL-Lab/bert-base-arabic-camelbert-mix`

Validation:
- Macro-F1: 1.0000
- Accuracy: 1.0000

Frozen Test:
- Macro-F1: 1.0000
- Accuracy: 1.0000

### Improvement over baseline
- Baseline Macro-F1: 1.0000
- Transformer Macro-F1: 1.0000
- Delta: +0.0000

The +0.08 improvement target could not be exceeded because the baseline already achieved the maximum possible Macro-F1 score of 1.0000 on the supplied split.


## Lab 3B — NER + Extractive QA

### NER
- Alignment contract: 8/8 tests passed
- Validation entity-level F1: 1.0000
- Validation accuracy: 1.0000
- Frozen test entity-level F1: 1.0000
- Frozen test accuracy: 1.0000
- Target F1 >= 0.80: Passed

### QA
- QA post-processing contract tests: Passed
- Smoke-set answerable questions: 12/12 correct
- Smoke-set unanswerable questions: 0/0 present in the supplied file

Note: The supplied `qa_smoke_set.json` contains 12 answerable questions and no unanswerable questions, while the README specifies 9 answerable and 3 unanswerable questions. Honest null handling is covered by the QA contract tests.


## Lab 4 — Clitic Segmentation for NER

### LOCATION Recall
- Before segmentation: 1.0000
- After segmentation: 1.0000
- Delta: +0.0000

The target improvement of about +0.04 recall points was not observed because the original Lab 3B NER model already achieved perfect LOCATION recall on the supplied test split, leaving no room for measurable improvement.


## Lab 4 — Arabic Model Bake-off

### Arabic grouped split
- Train: 5026
- Validation: 1077
- Test: 1097
- Train / Validation citizen overlap: 0
- Train / Test citizen overlap: 0
- Validation / Test citizen overlap: 0

### Test dialect distribution
- Gulf: 756
- MSA: 341

### CAMeLBERT-mix
- All Macro-F1: 1.0000
- Gulf Macro-F1: 1.0000
- MSA Macro-F1: 1.0000

### CAMeLBERT-DA
- All Macro-F1: 1.0000
- Gulf Macro-F1: 1.0000
- MSA Macro-F1: 1.0000

### Gulf Slice Delta
- CAMeLBERT-mix: 1.0000
- CAMeLBERT-DA: 1.0000
- Delta: +0.0000

The +0.04 Gulf-slice improvement target was not observed because both models achieved the maximum possible Macro-F1 score on the supplied Arabic evaluation split.


## Lab 5 — Bilingual Semantic Search

### Retrieval Results

Bi-encoder model:
`intfloat/multilingual-e5-base`

Cross-encoder reranker:
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

Corpus size:
20,000 cases

Index dimension:
768

### Measured Metrics

- Recall@10 without reranking: 0.0385
- MRR@10 without reranking: 0.0059
- Recall@10 with reranking: 0.0077
- MRR@10 with reranking: 0.0013

### Cross-lingual Slice

- Arabic Recall@10: 0.0167
- English Recall@10: 0.0000
- Cross-lingual slice gap: 0.0167

### No-answer Behaviour

- No-answer queries: 20
- Correct empty results: 20/20
- Best tested threshold: 0.0

The no-answer correctness target of at least 17/20 was achieved.

### Latency

- Bi-encoder average latency: 33.45 ms
- Two-stage average latency: 91.02 ms

### Evaluation Note

The retrieval targets were not achieved on the provided labelled query set.

Further inspection showed that 103 out of 130 answerable queries (79.23%) have exact-match cases in the 20,000-case corpus that are not included in their `relevant_case_ids`.

A total of 5,009 exact-match cases were found outside the labelled relevant sets.

For example, an Arabic query about a pothole had exact duplicate case texts retrieved with cosine similarity 1.0, but those case IDs were not included in the labelled relevant IDs.

Therefore, the measured retrieval metrics are strongly affected by duplicate cases and incomplete relevance labels. The reported values are kept unchanged to preserve honest evaluation.

### L2 Normalisation Diagnosis

The FAISS index uses `IndexFlatIP`, so both corpus vectors and query vectors are L2-normalised before similarity search.

Without L2 normalisation, inner-product scores are affected by vector magnitude rather than only semantic direction. This can produce plausible-looking nearest neighbours while significantly reducing retrieval metrics.

This demonstrates why retrieval quality should not be approved by manually inspecting a few results. The labelled evaluation set and retrieval metrics must be used.


## Lab 6 — Evaluation Report

### Bootstrap and Sliced Evaluation

- Validation rows: 2400
- Validation errors: 300
- English accuracy: 1.0000
- Arabic accuracy: 0.7500
- `parks` accuracy: 0.0000
- Medium-length accuracy: 0.8894
- Short-length accuracy: 0.8435

### Behavioural Evaluation

The supplied behavioural template file contains:

- Invariance tests: 200
- Directional tests: 200
- Explicit MFT rows: 0

The behavioural runner was implemented for invariance, directional behaviour, and optional MFT cases.

A sentiment predictor and explicit MFT test rows are not supplied in the repository, so unsupported pass rates were not fabricated.

### Manual Error Review

120 sampled validation errors were manually reviewed.

- Class confusion / under-learned class: 120/120
- Primary confusion: `parks -> roads`

The strongest observed fix is targeted improvement of the `parks` class. If all 300 current validation errors were corrected, validation accuracy would improve by approximately 12.5 percentage points.

Additional prioritised fixes are:

1. Add hard contrastive examples for `parks` versus `roads`.
2. Add Arabic orthographic and noisy-text augmentation.
3. Verify class mapping and class-balanced training for `parks`.


## Lab 7 — CPU Inference Optimisation

### Step 1 — FP32 Baseline

CPU threads pinned to: 4

| Configuration | p50 | p99 |
| --- | ---: | ---: |
| FP32, padded, max_length=512 | 1798.51 ms | 4976.57 ms |
| FP32, dynamic padding, max_length=128 | 131.17 ms | 232.10 ms |

Free-wins p99 speed-up from dynamic padding / shorter max length:

`21.44x`

The production latency evidence is CPU-based, as required by the lab.
