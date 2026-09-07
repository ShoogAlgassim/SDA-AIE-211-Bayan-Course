# Lab Notes

## Lab 1 — Defect Safari

Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes. For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1

* Class: Unicode forms
* Example: Arabic characters may appear in different Unicode forms.
* Why it matters: Different forms of the same character can be treated as different tokens.
* Decision: Clean

### Defect 2

* Class: Tatweel
* Example: الخدمــــة
* Why it matters: Tatweel adds unnecessary characters and can affect tokenization.
* Decision: Clean

### Defect 3

* Class: Code-switching Arabic ↔ English
* Example: الخدمة was very slow
* Why it matters: Bayan is bilingual, so both Arabic and English information can be useful.
* Decision: Preserve

### Defect 4

* Class: PII
* Example: 0551234567 or 1023456789
* Why it matters: Personal information should not be passed directly to the model.
* Decision: Clean

### Defect 5

* Class: Emoji
* Example: 😡 or ✅
* Why it matters: Emoji can carry useful sentiment information.
* Decision: Preserve

### Defect 6

* Class: HTML remnants
* Example: <br>
* Why it matters: HTML tags are formatting noise and may affect tokenization.
* Decision: Clean

## Lab 1 — Sentence Segmentation Spot-Check 
### Sentence Segmentation Spot-Check

* Checked 5 long examples from the dataset.
* Long complaints without clear sentence punctuation remained as one sentence, which was reasonable.
* Emoji and bilingual content were preserved after preprocessing.
* The numbered-list example was initially split incorrectly, with list numbers treated as separate sentences.
* The segmentation logic was updated to merge numbered-list markers with the following sentence.
* After the fix, the numbered-list complaint was segmented correctly:

  * `1. The streetlight is broken.`
  * `2. The road has a pothole.`
  * `3. The waste bin is full.`


## Lab 2 — Parameter Audit

### mBERT
- Embeddings: 92,208,384
- Attention: 28,366,848
- FFN: 56,669,184
- Norms: 18,432
- Pooler: 590,592
- Total: 177,853,440

### CAMeLBERT
- Embeddings: 23,436,288
- Attention: 28,366,848
- FFN: 56,669,184
- Norms: 18,432
- Pooler: 590,592
- Total: 109,081,344

### Why is the embedding share different?
mBERT has a much larger multilingual vocabulary, so its embedding matrix is much larger. CAMeLBERT focuses on Arabic, so it needs a smaller vocabulary and therefore fewer embedding parameters.

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
