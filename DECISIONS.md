# Decision Records

## tokenizer
## Lab 1 — Tokenizer Decision

**Chosen checkpoint:** `xlm-roberta-base`

**Arabic fertility:** 1.672
**English fertility:** 1.434
**Arabic p95 sequence length:** 21
**English p95 sequence length:** 23

**Why this choice fits Bayan:**
XLM-R was selected because it provides the best overall balance between Arabic and English tokenization. It produces relatively low fertility and short sequence lengths in both languages. CAMeLBERT performs better on Arabic, but its English fertility is much higher, while DistilBERT performs well on English but poorly on Arabic. Since Bayan is a bilingual Arabic-English system, XLM-R is the most suitable balanced choice.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:


## Arabic Model

CAMeLBERT-mix and CAMeLBERT-DA both achieved 1.0000 Macro-F1 on the full Arabic, Gulf, and MSA slices.

Because there was no measurable Gulf-slice advantage for CAMeLBERT-DA, there is no evidence-based winner on the supplied dataset. CAMeLBERT-mix is retained as the default model for consistency with the existing Bayan pipeline.
