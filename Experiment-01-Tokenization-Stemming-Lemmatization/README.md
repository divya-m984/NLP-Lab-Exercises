# Experiment 01 — Tokenization, Stemming, and Lemmatization

## Experiment Title

Implement Tokenization and Compare the Effectiveness of Stemming Versus Lemmatization in Improving Text Preprocessing for Sentiment Analysis

---

## Student Details

| Field           | Details    |
|-----------------|------------|
| Student Name    | Divya M    |
| Register Number | 24AD0074   |

---

## Aim

To implement tokenization and compare stemming and lemmatization for
sentiment analysis text preprocessing.

---

## Algorithms Explored

| # | Method | Description |
|---|--------|-------------|
| A | **Baseline** | Lowercase conversion, NLTK word_tokenize, alphabetic filter, stopword removal — no further normalization |
| B | **Porter Stemmer** | Rule-based suffix stripping algorithm (NLTK `PorterStemmer`) |
| C | **Snowball Stemmer** | Improved Porter-family stemmer with language support (NLTK `SnowballStemmer`, English) |
| D | **Lancaster Stemmer** | Aggressive iterative stemmer that produces shorter stems (NLTK `LancasterStemmer`) |
| E | **POS-aware WordNet Lemmatizer** | WordNet lemmatization guided by Treebank POS tags mapped to WordNet categories (noun, verb, adjective, adverb) using NLTK `pos_tag` and `WordNetLemmatizer` |

All variants share the same basic preprocessing pipeline (lowercase → tokenize → keep alphabetic → remove stopwords) before normalization is applied.

---

## Methodology

1. **Resource preparation** — NLTK data packages (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`, `averaged_perceptron_tagger_eng`, `sentence_polarity`) are checked and downloaded only if missing.
2. **Dataset creation** — 1 000 positive and 1 000 negative sentences are sampled with `random.seed(42)` from the NLTK `sentence_polarity` corpus and split 80 / 20 (stratified, `random_state=42`) into training and test sets.
3. **Preprocessing** — Each of the five variants processes every sentence independently; preprocessing time is measured with `time.perf_counter`.
4. **Vectorization** — Processed documents are converted to TF-IDF feature matrices using `sklearn.feature_extraction.text.TfidfVectorizer`.
5. **Classification** — A `MultinomialNB` classifier is trained and evaluated for every variant on the same fixed train/test split.
6. **Evaluation** — Accuracy, macro precision, macro recall, macro F1-score, vocabulary size, and preprocessing time are recorded per variant.
7. **Best method selection** — The variant with the highest macro F1-score is selected.
8. **Custom sentence analysis** — A user-supplied (or default) sentence is processed through all five pipelines; the best model predicts its sentiment with class probabilities.
9. **Output generation** — Results are saved to `output/output.txt`, `output/comparison.csv`, and `output/comparison_plot.png`.

---

## Dataset or Sample Input

- **Corpus**: NLTK `sentence_polarity` (Pang & Lee, 2005)
- **Classes**: positive (label 1) and negative (label 0)
- **Sample size**: 1 000 sentences per class (2 000 total); if fewer are available the maximum balanced count is used
- **Split**: 1 600 training / 400 test (80 / 20, stratified)
- **Default analysis sentence**: *"The movie was beautifully directed, although some scenes were moving too slowly."*

---

## Requirements

```
nltk
numpy
scikit-learn
matplotlib
```

Install with:

```bash
pip install nltk scikit-learn matplotlib numpy
```

---

## Execution

From the repository root:

```bash
python Experiment-01-Tokenization-Stemming-Lemmatization/experiment_01.py
```

With custom input:

```bash
python Experiment-01-Tokenization-Stemming-Lemmatization/experiment_01.py \
  --text "The acting was excellent but the ending was disappointing."
```

---

## Evaluation Criteria

Each preprocessing variant is evaluated on the held-out test set using:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correctly classified test sentences |
| **Macro Precision** | Average precision across both classes (positive and negative) |
| **Macro Recall** | Average recall across both classes |
| **Macro F1-Score** | Harmonic mean of macro precision and recall; used to select the best method |
| **Vocabulary Size** | Number of unique terms retained after TF-IDF fitting |
| **Preprocessing Time (s)** | Wall-clock time to preprocess all training and test sentences |

The best preprocessing method is the one that achieves the highest macro F1-score.

---

## Output Files

Running the script creates the following files inside `output/`:

| File | Contents |
|------|----------|
| `output.txt` | Student details, experiment metadata, dataset statistics, full evaluation table, best method summary, custom sentence analysis |
| `comparison.csv` | One row per method with columns: `method`, `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `vocabulary_size`, `preprocessing_seconds` |
| `comparison_plot.png` | Bar chart of macro F1-score for all five variants; the best-performing bar is highlighted in orange |

---

## Observations

The comparison across five preprocessing methods reveals clear trade-offs between normalization aggressiveness, classification performance, and processing cost:

| Method | Accuracy | Macro F1 | Vocabulary | Time |
|--------|----------|----------|------------|------|
| Baseline | 0.6500 | 0.6489 | 6383 | 0.3060 s |
| Porter Stemmer | 0.6525 | 0.6515 | 4947 | 0.8877 s |
| Snowball Stemmer | 0.6525 | 0.6515 | 4855 | 0.4386 s |
| Lancaster Stemmer | 0.6325 | 0.6306 | 4312 | 0.5540 s |
| POS Lemmatizer | 0.6575 | 0.6569 | 5561 | 5.8257 s |

- **POS-aware lemmatization** achieved the highest accuracy (0.6575) and macro F1-score (0.6569). By using POS tags to guide lemmatization, it preserves linguistically meaningful base forms and retains more semantic precision than stemming.
- **POS lemmatization was the slowest method** (5.8257 s) because POS tagging was performed on every sentence before lemmatization.
- **Porter and Snowball stemmers produced identical classification performance** (accuracy 0.6525, macro F1 0.6515). Snowball had a slightly smaller vocabulary (4855 vs. 4947) and was faster than Porter (0.4386 s vs. 0.8877 s) in this execution.
- **Lancaster stemmer** created the smallest vocabulary (4312) but achieved the lowest classification performance (accuracy 0.6325, macro F1 0.6306), indicating that its aggressive stemming removed useful distinctions between semantically different terms.
- **The baseline** was the fastest method (0.3060 s) but performed slightly below Porter, Snowball, and POS-aware lemmatization in both accuracy and macro F1-score.

---

## Result

POS-aware WordNet lemmatization was the best-performing preprocessing method for this dataset, achieving the highest macro F1-score (0.6569) and accuracy (0.6575) among all five variants. However, it also incurred the highest processing cost (5.8257 s) due to the additional POS tagging step. This result is specific to the NLTK `sentence_polarity` corpus and the experimental setup described above; it does not necessarily generalize to every sentiment analysis dataset.

Complete results are available in:

- `output/output.txt` — full evaluation table, best method summary, and custom sentence analysis
- `output/comparison.csv` — per-method metrics in CSV format
- `output/comparison_plot.png` — bar chart comparing macro F1-scores across all variants
- `output/terminal-output.png` — screenshot of terminal output during execution
