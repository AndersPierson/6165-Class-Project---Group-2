# Twitter Sentiment Analysis — GRU & LSTM Comparison
**DASC 6165 | Group 2**

This repository contains the full codebase, notebooks, and supporting assets for a deep learning sentiment analysis project built on the [Twitter Entity Sentiment Analysis dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). The project compares two recurrent neural network architectures — Gated Recurrent Units (GRU) and Long Short-Term Memory networks (LSTM) — across three distinct data preprocessing strategies to evaluate how both architecture choice and data formatting decisions interact to influence model performance.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Preprocessing Strategies](#preprocessing-strategies)
- [Model Architectures](#model-architectures)
- [Experiment Matrix](#experiment-matrix)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

Sentiment analysis on social media text is a well-studied but genuinely difficult problem. Twitter data in particular introduces challenges: short and informal text, heavy use of slang and abbreviations, missing context, and significant class imbalance between sentiment categories. This project was designed around two central questions:

1. Does an LSTM's richer gating mechanism (forget gate, input gate, output gate) meaningfully outperform a simpler GRU on this kind of data?
2. How does the way you structure and preprocess the data affect model performance — sometimes more than the architecture itself?

To answer these questions, we ran a structured 3×2 experiment matrix: three preprocessing approaches applied independently to both GRU and LSTM architectures, yielding six comparable models. We also built a custom lexicon-based sentiment classifier as a supplementary exercise to understand the baseline difficulty of the task from first principles.

---

## Dataset

The dataset consists of labeled Twitter posts across four sentiment classes:

| Class | Description |
|---|---|
| **Positive** | Tweet expresses a positive sentiment toward the entity |
| **Negative** | Tweet expresses a negative sentiment toward the entity |
| **Neutral** | Tweet is factual or non-opinionated toward the entity |
| **Irrelevant** | Tweet is unrelated to the entity |

Each row contains a tweet ID, the target entity, a sentiment label, and the tweet text. The training file contains approximately 74,000 rows and the validation file contains 1,000 rows.

**Files expected in the project root:**
- `twitter_training (1).csv`
- `twitter_validation (1).csv`

---

## Repository Structure

```
├── Group_2_GRU_Simple_RawData.ipynb           # GRU Model 1 — raw data
├── Group_2_GRU_Simple_Concatenate_Final.ipynb # GRU Model 2 — simple concatenation
├── Group_2_GRU_Complex_Concatenate_Final.ipynb# GRU Model 3 — complex concatenation
├── Group_2_LSTM_Simple_RawData.ipynb          # LSTM Model 1 — raw data
├── Group_2_LSTM_Simple_Concatenate_Final.ipynb# LSTM Model 2 — simple concatenation
├── Group_2_LSTM_Complex_Concatenate_Final.ipynb# LSTM Model 3 — complex concatenation
├── twitter_training (1).csv                   # Training data (not tracked by Git LFS)
├── twitter_validation (1).csv                 # Validation data
├── glove.6B.100d.txt                          # GloVe embeddings (required for Model 3s)
└── outputs/
    ├── lstm_simple_rawdata/
    ├── lstm_simple_concatenate/
    ├── lstm_complex_concatenate/
    ├── gru_simple_rawdata/
    ├── gru_simple_concatenate/
    └── gru_complex_concatenate/
```

Each `outputs/` subdirectory is populated on run with:
- `training_history.png` — loss and accuracy curves
- `confusion_matrix.png` — per-class prediction breakdown
- `run_metadata.json` — validation loss, accuracy, vocab size, sequence length
- Saved model (`.keras`), tokenizer (`.pkl`), and label encoder (`.pkl`)

---

## Preprocessing Strategies

All models share a common text cleaning pipeline before branching into their respective strategies:

1. **Lowercasing** — normalize all text to lowercase
2. **URL/mention/hashtag removal** — strip `http://...`, `@username`, and `#hashtag` patterns
3. **Non-alphabetic character removal** — remove punctuation, numbers, and special characters
4. **Stopword filtering** — remove common English stopwords via NLTK
5. **Lemmatization** — reduce words to their base form using WordNetLemmatizer

After cleaning, the three strategies diverge:

### Strategy 1 — Raw Data
Each tweet is treated as an independent training sample. This preserves the full ~74,000 training rows and keeps sequences short (the 95th percentile of sequence length was used as `max_sequence_length`). This is the most straightforward setup and serves as the performance baseline.

### Strategy 2 — Simple Concatenation
Tweets that share the same `tweet_id` are concatenated into a single string before training. The intent was to give the model more context per sample by combining related tweets. In practice, this collapsed the training set from ~74,000 rows to approximately 12,000, and the resulting longer sequences proved too sparse for a simple architecture to learn from effectively. Both simple concatenation models failed to generalize.

### Strategy 3 — Complex Concatenation
Same concatenation approach as Strategy 2, but paired with a significantly more capable architecture designed to handle the longer, reduced-sample setting. This strategy is only used in Model 3 of each architecture family.

---

## Model Architectures

### Simple Architecture (Models 1 & 2)
```
Embedding(vocab_size=10000, output_dim=100)
    ↓
[GRU | LSTM](units=64)
    ↓
Dense(num_classes, activation='softmax')
```
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- EarlyStopping: patience=3, monitor=val_loss, restore_best_weights=True
- Max epochs: 10 (Model 1), 5 (Model 2)
- Batch size: 32

### Complex Architecture (Model 3)
```
Embedding(vocab_size, output_dim=100, weights=[GloVe], trainable=False)
    ↓
Bidirectional([GRU | LSTM](units=128))
    ↓
Dropout(0.4)
    ↓
Dense(num_classes, activation='softmax')
```
- Pre-trained GloVe embeddings: `glove.6B.100d.txt`
- Class weights applied during training to address sentiment imbalance
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- EarlyStopping: patience=3, monitor=val_loss, restore_best_weights=True
- Max epochs: 10

---

## Experiment Matrix

| Model | Architecture | Preprocessing | Key Design Notes |
|---|---|---|---|
| GRU Model 1 | GRU | Raw Data | Baseline; simple architecture |
| GRU Model 2 | GRU | Simple Concatenate | Collapsed to majority class prediction |
| GRU Model 3 | GRU | Complex Concatenate | GloVe + Bidirectional + class weights |
| LSTM Model 1 | LSTM | Raw Data | Best overall performer |
| LSTM Model 2 | LSTM | Simple Concatenate | Collapsed to majority class prediction |
| LSTM Model 3 | LSTM | Complex Concatenate | GloVe + Bidirectional + class weights |

---

## Results Summary

| Model | Val Accuracy | Val Loss | Macro F1 |
|---|---|---|---|
| GRU — Raw Data | 82.77% | 0.5066 | 0.83 |
| GRU — Simple Concatenate | ~27% | ~1.38 | ~0.11 |
| GRU — Complex Concatenate | 54.74% | ~1.02 | ~0.54 |
| **LSTM — Raw Data** | **94.90%** | **0.2034** | **0.95** |
| LSTM — Simple Concatenate | 27.70% | 1.3693 | 0.11 |
| LSTM — Complex Concatenate | 63.00% | 0.9444 | 0.62 |

---

## Key Findings

**Raw data wins.** The single biggest factor in model performance was keeping tweets as individual, short samples. Both architectures performed best on raw data, and the LSTM's 94.90% accuracy on that split was the standout result of the entire experiment.

**Concatenation without complexity is a trap.** The intuition behind concatenation — "more context per sample should help" — sounds reasonable but ignored a critical side effect: dramatically fewer training rows. Reducing the dataset from ~74,000 to ~12,000 samples while increasing sequence length left both simple models unable to learn anything beyond the majority class.

**Architecture upgrades can recover concatenated performance, but not fully.** GloVe embeddings, bidirectional processing, and class weighting together rescued the concatenated models from complete collapse. But even with these improvements, neither complex concatenation model approached the accuracy of the raw data models. The preprocessing decision introduced a ceiling.

**LSTM consistently outperformed GRU.** Across all three strategies, the LSTM matched or exceeded the GRU. The gap was largest on raw data (94.90% vs 82.77%) and present but smaller on complex concatenation (63.00% vs 54.74%). The LSTM's additional gating complexity appears to be genuinely advantageous for this classification task.

**The Irrelevant class was the hardest throughout.** Across all six models, Irrelevant had the lowest or tied-lowest performance. Tweets that are off-topic for a given entity share surface-level language with on-topic tweets, making them structurally difficult to separate without deeper semantic understanding.

---

## How to Run

1. Clone the repository and ensure the data CSVs and GloVe file are in the project root.
2. Install dependencies (see below).
3. Open any notebook in Jupyter or Google Colab and run all cells. Each notebook is self-contained — it loads data, preprocesses, trains, evaluates, saves plots, and serializes the model.

> **Note:** Model 3 notebooks require `glove.6B.100d.txt` (available from the [GloVe project page](https://nlp.stanford.edu/projects/glove/)). Download the `glove.6B.zip`, extract it, and place the `100d` file in the project root.

---

## Dependencies

```
tensorflow >= 2.x
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
```

Install via pip:
```bash
pip install tensorflow numpy pandas scikit-learn nltk matplotlib seaborn
```

NLTK corpora (downloaded automatically by the notebooks on first run):
```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```
