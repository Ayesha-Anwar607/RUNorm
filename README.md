# RUNorm: A Roman Urdu Normalization and Code-Switching Parser for Low-Resource NLP

> **A preprocessing and infrastructure layer for Roman Urdu text **

**Author:** Ayesha Anwar
---

## Table of Contents

- [Motivation](#motivation)
- [Pipeline Overview](#pipeline-overview)
- [Components](#components)
  - [Normalizer (3-Layer)](#normalizer-3-layer)
  - [Code-Switch Tagger](#code-switch-tagger)
  - [SQLite Schema](#sqlite-schema)
- [Output Format](#output-format)
- [Data](#data)
  - [Primary Dataset](#primary-dataset)
  - [Supplementary Files](#supplementary-files)
  - [Backup Corpora](#backup-corpora)
- [Benchmark Strategy](#benchmark-strategy)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [Setup & Usage](#setup--usage)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Roman Urdu — Urdu written in the Latin script — is the dominant informal register on Pakistani social media. Two core problems make it extremely difficult for NLP systems to process:

1. **No standard spelling:** A single word surfaces in 5+ orthographic variants  
   (e.g., *nahi*, *nhi*, *nai*, *nahin* all mean "no").
2. **Pervasive code-switching:** Urdu and English are freely mixed within a single sentence  
   (e.g., *"yaar yeh movie bohat acha tha lol"*).

RUNorm provides a **reusable preprocessing pipeline** that normalizes spelling variants to canonical forms and tags each token with its language (Urdu or English), producing clean, structured data that any downstream NLP task can consume.

---

## Pipeline Overview

```
Raw Text
  │
  ▼
┌──────────────────────────────────┐
│  ETL Layer                       │
│  • Clean (PII removal, encoding) │
│  • Tokenize (elongation, emoji,  │
│    hashtags, handles)            │
│  • Store in SQLite               │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Normalization Engine            │
│  Layer 1: Dictionary lookup      │
│  Layer 2: Phonetic matching      │
│  Layer 3: T5-small fallback      │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Code-Switch Tagger              │
│  CRF (primary) / XLM-R (compare)│
└──────────────┬───────────────────┘
               │
               ▼
      Structured JSON Output
```

---

## Components

### Normalizer (3-Layer)

| Layer | Method | Coverage | Lookup Cost |
|-------|--------|----------|-------------|
| **1. Manual Dictionary** | Hand-curated JSON mapping of top 1,000 frequent words and their variants to canonical forms | High-frequency tokens | O(1) |
| **2. Phonetic Matching** | `jellyfish` library (Soundex / Metaphone) for near-miss variants not in the dictionary | Medium-frequency tokens | O(1) per comparison |
| **3. T5-small Fallback** | Fine-tuned T5-small model for rare or novel spellings that escape the first two layers | Long-tail / OOV tokens | O(n) inference |

Each layer is evaluated independently via **per-layer ablation** to quantify its contribution.

### Code-Switch Tagger

- **Annotation:** 1,500 sentences hand-labeled **per token** (`UR` or `EN`) in Label Studio.
  - Balanced across slang, formal, and code-mixed registers.
  - Spread across data sources with no overlap with dev/test splits.
  - 150 sentences double-labeled for **Cohen's κ** inter-annotator agreement.
- **Primary model:** CRF (`sklearn-crfsuite`) — strong baseline for sequence labeling on limited data.
- **Comparison model:** XLM-RoBERTa — to measure whether a large pretrained multilingual model adds value over a feature-engineered CRF on this task.

### SQLite Schema

```sql
CREATE TABLE sentences (
    sentence_id   INTEGER PRIMARY KEY,
    raw_text      TEXT NOT NULL,
    normalized_text TEXT,
    source        TEXT,
    domain        TEXT
);

CREATE TABLE tokens (
    token_id          INTEGER PRIMARY KEY,
    sentence_id       INTEGER REFERENCES sentences(sentence_id),
    position          INTEGER,
    raw_token         TEXT,
    normalized_token  TEXT,
    lang_tag          TEXT  -- 'UR' or 'EN'
);
```

---

## Output Format

```
Input:  yaar yeh movie bohat acha tha lol
Output: yaar(UR) yeh(UR) movie(EN) bohat(UR) acha(UR) tha(UR) lol(EN)
```

Structured JSON example:

```json
{
  "raw_text": "yaar yeh movie bohat acha tha lol",
  "normalized_text": "yaar yeh movie bohat acha tha lol",
  "tokens": [
    {"position": 0, "raw": "yaar",  "normalized": "yaar",  "lang": "UR"},
    {"position": 1, "raw": "yeh",   "normalized": "yeh",   "lang": "UR"},
    {"position": 2, "raw": "movie", "normalized": "movie", "lang": "EN"},
    {"position": 3, "raw": "bohat", "normalized": "bohat", "lang": "UR"},
    {"position": 4, "raw": "acha",  "normalized": "acha",  "lang": "UR"},
    {"position": 5, "raw": "tha",   "normalized": "tha",   "lang": "UR"},
    {"position": 6, "raw": "lol",   "normalized": "lol",   "lang": "EN"}
  ]
}
```

---

## Data

### Primary Dataset

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `anonymized_84k_for_labeling.csv` | 84,927 | `message`, `label` | Names masked as `PERSON`. Class imbalance: ~34k Neg, ~32k Neu, ~18.6k Pos |

**Source:** RUDaSA (Ahmad, 2026, Research Square) — 99k Roman Urdu social-media instances, obtained directly from the dataset owner.

### Supplementary Files

| File | Rows | Notes |
|------|------|-------|
| `Final_merged.csv` | 8,872 | Multi-attribute labels (sentiment, emotion, sarcasm, toxicity, domain, style). ⚠️ Contains 5 duplicated header rows and at least one broken row — **clean before use**. |
| `for__negative_samples.csv` | ~33,000 | Additional tweets. ⚠️ Contains some Indonesian text — **run language ID before use**. |
| `whatsapp_data__1_labeled.csv` | — | WhatsApp conversation data with labels |
| `whatsapp_others_cleaned.csv` | — | Additional cleaned WhatsApp data |

> **Privacy note:** Scan all files for leftover phone numbers, @handles, and other PII before processing.

### Backup Corpora

| Corpus | Size | Source |
|--------|------|--------|
| Smat26/Roman-Urdu-Dataset | ~20k | GitHub |
| roman_urdu | ~20k | HuggingFace |
| UCI RUSAD | ~11k | UCI ML Repository |
| RUSA-19 | — | Published dataset |

---

## Benchmark Strategy

To prove that RUNorm is **useful**, we run the same sentiment classification task on:
- **Raw text** (no preprocessing), and
- **RUNorm-cleaned text** (normalized + code-switch tagged).

The **accuracy gain (Δ)** between the two conditions is the primary evidence of the tool's value.

The benchmark uses the RUDaSA sentiment labels (Negative / Neutral / Positive) on the `anonymized_84k_for_labeling.csv` dataset.

---

## Project Structure

```
Runorm/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                  # Original unmodified source files
│   ├── processed/            # Cleaned outputs & SQLite database
│   └── splits/               # train.csv, dev.csv, test.csv indices
├── dictionary/
│   └── dictionary.json       # Canonical spelling mappings
├── notebooks/
│   └── 01_data_cleaning.ipynb
├── src/
│   ├── cleaning.py           # Regex rules, PII removal, noise filtering
│   ├── tokenizer.py          # Elongation, emoji, hashtag handling
│   ├── db.py                 # SQLite schema & ingestion pipeline
│   ├── normalizer.py         # 3-layer normalization engine
│   └── tagger.py             # CRF code-switch tagger
├── models/                   # Trained model artifacts
├── demo/                     # Streamlit app
└── RUDASA/                   # Raw dataset files (not committed)
```

---

## Roadmap

| Week | Milestone |
|------|-----------|
| **1** | Repo setup. Data cleaning pipeline with per-rule drop counts. Tokenizer (elongation, emoji, hashtags). SQLite ingestion. |
| **2** | Rerunnable SQLite ETL script. Add `split` column (train/dev/test) and persist split to disk. |
| **3** | Frequency analysis on top 3,000 tokens. Group variants, pick canonical forms with written rules. Save `dictionary.json`. Inter-annotator agreement check (100 random words, second annotator). |
| **4** | Sample 1,500 sentences (balanced slang / formal / code-mixed). Write annotation guide. Pilot 100 labels, then full annotation. 150 double-labeled for Cohen's κ. |
| **5+** | Dictionary-lookup baseline → CRF tagger → phonetic layer → per-layer ablation → sentiment benchmark → T5-small & XLM-R (last). |

---

## Setup & Usage

```bash
# Clone and enter the project
git init
git add .
git commit -m "Initial project setup"

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

> `requirements.txt` will be populated as the project progresses.

---

## Citation

If you use RUNorm or the underlying RUDaSA dataset, please cite:

```bibtex
@article{ahmad2026rudasa,
  title     = {RUDaSA: Roman Urdu Dataset for Sentiment Analysis},
  author    = {Ahmad, ...},
  year      = {2026},
  journal   = {Research Square (Preprint)},
  note      = {99k Roman Urdu social-media instances}
}
```

> **Note on the RUDaSA paper:**
> - Krippendorff's α is reported as both 0.81 and 0.812 in different sections.
> - Table 10 states +5.56 pp while the body text says +5.5.
> - Table 6 reports best epoch as 17, but early stopping is described as triggering at epoch 20.

---

## License

*To be determined.*
