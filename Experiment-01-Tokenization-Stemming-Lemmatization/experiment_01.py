"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 01
Title           : Implement Tokenization and Compare the Effectiveness of Stemming
                  Versus Lemmatization in Improving Text Preprocessing for Sentiment
                  Analysis
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SAMPLE_SIZE = 1000          # per class
TRAIN_RATIO = 0.80

STUDENT_NAME = "Divya M"
REGISTER_NUMBER = "24AD0074"
EXPERIMENT_TITLE = (
    "Implement Tokenization and Compare the Effectiveness of Stemming Versus\n"
    "Lemmatization in Improving Text Preprocessing for Sentiment Analysis"
)
EXPERIMENT_AIM = (
    "To implement tokenization and compare stemming and lemmatization for\n"
    "sentiment analysis text preprocessing."
)
DEFAULT_SENTENCE = (
    "The movie was beautifully directed, although some scenes were moving too slowly."
)

# ---------------------------------------------------------------------------
# NLTK resource management
# ---------------------------------------------------------------------------

def ensure_nltk_resources() -> None:
    """Download any NLTK resources that are not already present."""
    import nltk

    resources = [
        ("tokenizers/punkt",                    "punkt"),
        ("tokenizers/punkt_tab",                "punkt_tab"),
        ("corpora/stopwords",                   "stopwords"),
        ("corpora/wordnet",                     "wordnet"),
        ("corpora/omw-1.4",                     "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/sentence_polarity",           "sentence_polarity"),
    ]

    for resource_path, resource_id in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_id, quiet=True)
            except Exception as exc:
                print(f"Warning: could not download '{resource_id}': {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset() -> Tuple[List[str], List[int], int]:
    """
    Load the sentence_polarity corpus and return a balanced, reproducible sample.

    Returns:
        sentences : list of raw sentence strings
        labels    : parallel list of int labels (1 = positive, 0 = negative)
        n_used    : number of examples per class actually used
    """
    import random
    from nltk.corpus import sentence_polarity

    pos_sentences = list(sentence_polarity.sents(categories="pos"))
    neg_sentences = list(sentence_polarity.sents(categories="neg"))

    n_used = min(SAMPLE_SIZE, len(pos_sentences), len(neg_sentences))

    rng = random.Random(RANDOM_SEED)
    pos_sample = rng.sample(pos_sentences, n_used)
    neg_sample = rng.sample(neg_sentences, n_used)

    sentences: List[str] = []
    labels: List[int] = []
    for sent_tokens in pos_sample:
        sentences.append(" ".join(sent_tokens))
        labels.append(1)
    for sent_tokens in neg_sample:
        sentences.append(" ".join(sent_tokens))
        labels.append(0)

    return sentences, labels, n_used


def split_dataset(
    sentences: List[str],
    labels: List[int],
) -> Tuple[List[str], List[str], List[int], List[int]]:
    """Stratified 80/20 train-test split."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        sentences,
        labels,
        test_size=1.0 - TRAIN_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
    )


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def basic_tokens(text: str, stop_words: set) -> List[str]:
    """Lowercase, tokenize, keep alphabetic tokens, remove stopwords."""
    from nltk import word_tokenize

    return [
        tok for tok in word_tokenize(text.lower())
        if tok.isalpha() and tok not in stop_words
    ]


def _treebank_to_wordnet_pos(treebank_tag: str) -> str:
    """Map a Treebank POS tag to a WordNet POS constant."""
    from nltk.corpus import wordnet

    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_baseline(tokens: List[str]) -> List[str]:
    """Return tokens unchanged (baseline: no stemming or lemmatization)."""
    return tokens


def preprocess_porter(tokens: List[str], stemmer) -> List[str]:
    """Apply Porter stemmer."""
    return [stemmer.stem(tok) for tok in tokens]


def preprocess_snowball(tokens: List[str], stemmer) -> List[str]:
    """Apply Snowball stemmer."""
    return [stemmer.stem(tok) for tok in tokens]


def preprocess_lancaster(tokens: List[str], stemmer) -> List[str]:
    """Apply Lancaster stemmer."""
    return [stemmer.stem(tok) for tok in tokens]


def preprocess_lemmatizer(tokens: List[str], lemmatizer) -> List[str]:
    """POS-aware WordNet lemmatization."""
    from nltk import pos_tag

    tagged = pos_tag(tokens)
    return [
        lemmatizer.lemmatize(tok, pos=_treebank_to_wordnet_pos(tag))
        for tok, tag in tagged
    ]


# ---------------------------------------------------------------------------
# Pipeline for one preprocessing variant
# ---------------------------------------------------------------------------

def run_variant(
    name: str,
    train_texts: List[str],
    test_texts: List[str],
    train_labels: List[int],
    test_labels: List[int],
    stop_words: set,
    normalizer,          # callable(tokens) -> tokens
) -> Dict:
    """
    Preprocess data, fit TF-IDF + MultinomialNB, evaluate, and return metrics.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    t_start = time.perf_counter()

    def process(texts: List[str]) -> List[str]:
        return [
            " ".join(normalizer(basic_tokens(text, stop_words)))
            for text in texts
        ]

    train_processed = process(train_texts)
    test_processed  = process(test_texts)

    t_elapsed = time.perf_counter() - t_start

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_processed)
    X_test  = vectorizer.transform(test_processed)

    clf = MultinomialNB()
    clf.fit(X_train, train_labels)
    predictions = clf.predict(X_test)

    acc = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro", zero_division=0
    )
    vocab_size = len(vectorizer.vocabulary_)

    return {
        "name":                 name,
        "accuracy":             acc,
        "macro_precision":      precision,
        "macro_recall":         recall,
        "macro_f1":             f1,
        "vocabulary_size":      vocab_size,
        "preprocessing_seconds": t_elapsed,
        "classifier":           clf,
        "vectorizer":           vectorizer,
    }


# ---------------------------------------------------------------------------
# Custom sentence analysis
# ---------------------------------------------------------------------------

def analyse_sentence(
    text: str,
    stop_words: set,
    porter_stemmer,
    snowball_stemmer,
    lancaster_stemmer,
    lemmatizer,
    best_result: Dict,
) -> Dict:
    """
    Run all preprocessing variants on a single sentence and predict sentiment.

    Returns a dict with all intermediate outputs.
    """
    base_toks   = basic_tokens(text, stop_words)
    porter_toks = preprocess_porter(base_toks, porter_stemmer)
    snow_toks   = preprocess_snowball(base_toks, snowball_stemmer)
    lanc_toks   = preprocess_lancaster(base_toks, lancaster_stemmer)
    lemma_toks  = preprocess_lemmatizer(base_toks, lemmatizer)

    vectorizer = best_result["vectorizer"]
    clf        = best_result["classifier"]

    # Use the same preprocessing method as the best variant
    best_name = best_result["name"]
    if best_name == "Baseline":
        proc_toks = base_toks
    elif best_name == "Porter Stemmer":
        proc_toks = porter_toks
    elif best_name == "Snowball Stemmer":
        proc_toks = snow_toks
    elif best_name == "Lancaster Stemmer":
        proc_toks = lanc_toks
    else:
        proc_toks = lemma_toks

    vec = vectorizer.transform([" ".join(proc_toks)])
    prediction = clf.predict(vec)[0]
    sentiment  = "Positive" if prediction == 1 else "Negative"

    proba = None
    if hasattr(clf, "predict_proba"):
        proba_arr = clf.predict_proba(vec)[0]
        classes   = list(clf.classes_)
        neg_prob  = proba_arr[classes.index(0)]
        pos_prob  = proba_arr[classes.index(1)]
        proba     = {"positive": pos_prob, "negative": neg_prob}

    return {
        "original":    text,
        "base_tokens": base_toks,
        "porter":      porter_toks,
        "snowball":    snow_toks,
        "lancaster":   lanc_toks,
        "lemmatized":  lemma_toks,
        "sentiment":   sentiment,
        "proba":       proba,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _format_table(results: List[Dict]) -> str:
    """Return a plain-text comparison table."""
    header = (
        f"{'Method':<25} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>9} {'F1':>9} {'Vocab':>7} {'Time(s)':>9}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r['name']:<25} {r['accuracy']:>9.4f} {r['macro_precision']:>10.4f} "
            f"{r['macro_recall']:>9.4f} {r['macro_f1']:>9.4f} "
            f"{r['vocabulary_size']:>7} {r['preprocessing_seconds']:>9.4f}"
        )
    return "\n".join(rows)


def save_outputs(
    results: List[Dict],
    best: Dict,
    analysis: Dict,
    n_used: int,
    n_train: int,
    n_test: int,
    output_dir: Path,
) -> None:
    """Write output.txt, comparison.csv, and comparison_plot.png."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- output.txt --------------------------------------------------------
    table_str = _format_table(results)

    proba_line = ""
    if analysis["proba"]:
        p = analysis["proba"]
        proba_line = (
            f"  Positive probability : {p['positive']:.4f}\n"
            f"  Negative probability : {p['negative']:.4f}\n"
        )

    output_text = f"""============================================================
NLP Laboratory — Experiment 01
============================================================
Student Name    : {STUDENT_NAME}
Register Number : {REGISTER_NUMBER}

Title:
  {EXPERIMENT_TITLE}

Aim:
  {EXPERIMENT_AIM}

------------------------------------------------------------
Dataset Information
------------------------------------------------------------
  Corpus            : NLTK sentence_polarity
  Examples per class: {n_used}
  Total examples    : {n_used * 2}
  Training set      : {n_train}
  Test set          : {n_test}

------------------------------------------------------------
Evaluation Results (MultinomialNB with TF-IDF)
------------------------------------------------------------
{table_str}

------------------------------------------------------------
Best Preprocessing Method
------------------------------------------------------------
  Method : {best['name']}
  Macro F1-score : {best['macro_f1']:.4f}

------------------------------------------------------------
Custom Sentence Analysis
------------------------------------------------------------
  Original text    : {analysis['original']}
  Basic tokens     : {analysis['base_tokens']}
  Porter stemmed   : {analysis['porter']}
  Snowball stemmed : {analysis['snowball']}
  Lancaster stemmed: {analysis['lancaster']}
  Lemmatized (POS) : {analysis['lemmatized']}
  Predicted sentiment ({best['name']}): {analysis['sentiment']}
{proba_line}
------------------------------------------------------------
Final Result
------------------------------------------------------------
  The experiment compared five preprocessing methods for
  sentiment analysis using the NLTK sentence_polarity corpus.
  Based on macro F1-score, the best method was:

      {best['name']} (F1 = {best['macro_f1']:.4f})

  Full numeric results are recorded in:
    output/comparison.csv
    output/comparison_plot.png
============================================================
"""

    (output_dir / "output.txt").write_text(output_text, encoding="utf-8")

    # ---- comparison.csv ----------------------------------------------------
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method", "accuracy", "macro_precision",
                "macro_recall", "macro_f1",
                "vocabulary_size", "preprocessing_seconds",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "method":                 r["name"],
                "accuracy":               round(r["accuracy"], 6),
                "macro_precision":        round(r["macro_precision"], 6),
                "macro_recall":           round(r["macro_recall"], 6),
                "macro_f1":               round(r["macro_f1"], 6),
                "vocabulary_size":        r["vocabulary_size"],
                "preprocessing_seconds":  round(r["preprocessing_seconds"], 6),
            })

    # ---- comparison_plot.png -----------------------------------------------
    method_names = [r["name"] for r in results]
    f1_scores    = [r["macro_f1"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(method_names, f1_scores, color="steelblue", edgecolor="white")

    # Highlight best
    best_idx = method_names.index(best["name"])
    bars[best_idx].set_color("darkorange")

    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_title(
        "Macro F1-Score Comparison of Preprocessing Methods\n"
        "(orange = best method)",
        fontsize=12,
    )
    ax.set_xlabel("Preprocessing Method", fontsize=11)
    ax.set_ylabel("Macro F1-Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", labelsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "comparison_plot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_summary(results: List[Dict], best: Dict, n_used: int, n_train: int, n_test: int) -> None:
    """Print the main summary to the terminal."""
    print("=" * 64)
    print("NLP Laboratory — Experiment 01")
    print("=" * 64)
    print(f"Student : {STUDENT_NAME}   ({REGISTER_NUMBER})")
    print()
    print(f"Dataset : sentence_polarity  |  {n_used} per class  |  "
          f"train={n_train}  test={n_test}")
    print()
    print(_format_table(results))
    print()
    print(f"Best method (macro F1): {best['name']}  —  {best['macro_f1']:.4f}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 01 — Tokenization, Stemming, and Lemmatization"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_SENTENCE,
        help="Custom sentence to analyse (default: built-in example)",
    )
    args = parser.parse_args()

    # 1. Download missing NLTK resources
    ensure_nltk_resources()

    # 2. Imports that require NLTK data
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
    from nltk.stem import WordNetLemmatizer

    stop_words       = set(stopwords.words("english"))
    porter_stemmer   = PorterStemmer()
    snowball_stemmer = SnowballStemmer("english")
    lancaster_stemmer = LancasterStemmer()
    lemmatizer       = WordNetLemmatizer()

    # 3. Load and split dataset
    sentences, labels, n_used = load_dataset()
    train_texts, test_texts, train_labels, test_labels = split_dataset(sentences, labels)
    n_train, n_test = len(train_texts), len(test_texts)

    # 4. Run all five preprocessing variants
    variants = [
        (
            "Baseline",
            lambda toks: preprocess_baseline(toks),
        ),
        (
            "Porter Stemmer",
            lambda toks, s=porter_stemmer: preprocess_porter(toks, s),
        ),
        (
            "Snowball Stemmer",
            lambda toks, s=snowball_stemmer: preprocess_snowball(toks, s),
        ),
        (
            "Lancaster Stemmer",
            lambda toks, s=lancaster_stemmer: preprocess_lancaster(toks, s),
        ),
        (
            "POS Lemmatizer",
            lambda toks, l=lemmatizer: preprocess_lemmatizer(toks, l),
        ),
    ]

    results: List[Dict] = []
    for name, normalizer in variants:
        result = run_variant(
            name,
            train_texts, test_texts,
            train_labels, test_labels,
            stop_words,
            normalizer,
        )
        results.append(result)

    # 5. Identify best by macro F1
    best = max(results, key=lambda r: r["macro_f1"])

    # 6. Analyse custom/default sentence
    analysis = analyse_sentence(
        args.text,
        stop_words,
        porter_stemmer,
        snowball_stemmer,
        lancaster_stemmer,
        lemmatizer,
        best,
    )

    # 7. Save output files
    output_dir = Path(__file__).parent / "output"
    save_outputs(results, best, analysis, n_used, n_train, n_test, output_dir)

    # 8. Print summary
    print_summary(results, best, n_used, n_train, n_test)
    print(f"\nCustom sentence : {analysis['original']}")
    print(f"Predicted sentiment ({best['name']}): {analysis['sentiment']}")
    if analysis["proba"]:
        p = analysis["proba"]
        print(f"  Positive: {p['positive']:.4f}   Negative: {p['negative']:.4f}")
    print(f"\nOutput files written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
