"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 02
Title           : Develop a Part-of-Speech (POS) Tagging System Using NLTK and
                  Evaluate Its Accuracy on a Corpus of News Articles
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_RATIO = 0.80

STUDENT_NAME = "Divya M"
REGISTER_NUMBER = "24AD0074"
EXPERIMENT_TITLE = (
    "Develop a Part-of-Speech (POS) Tagging System Using NLTK and Evaluate\n"
    "Its Accuracy on a Corpus of News Articles"
)
EXPERIMENT_AIM = (
    "To implement Part-of-Speech tagging using NLTK and identify the\n"
    "grammatical categories of words in a sentence."
)
DEFAULT_SENTENCE = (
    "The government announced a new economic policy after Parliament "
    "approved the bill."
)

UNIVERSAL_TAG_ORDER = [
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET",
    "ADP", "NUM", "CONJ", "PRT", ".", "X",
]

TAG_NAMES: Dict[str, str] = {
    "NOUN": "Noun",
    "VERB": "Verb",
    "ADJ": "Adjective",
    "ADV": "Adverb",
    "PRON": "Pronoun",
    "DET": "Determiner",
    "ADP": "Adposition",
    "NUM": "Numeral",
    "CONJ": "Conjunction",
    "PRT": "Particle",
    ".": "Punctuation",
    "X": "Other",
}


# ---------------------------------------------------------------------------
# NLTK resource management
# ---------------------------------------------------------------------------
def ensure_nltk_resources() -> None:
    """Download only missing NLTK resources."""
    import nltk

    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/brown": "brown",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
        "taggers/universal_tagset": "universal_tagset",
    }

    # punkt_tab is needed in newer NLTK versions
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass  # not required in this NLTK version

    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading missing NLTK resource: {name}")
            try:
                nltk.download(name, quiet=True)
            except Exception as exc:
                print(f"ERROR: Failed to download '{name}': {exc}", file=sys.stderr)
                sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_split_data() -> Tuple[
    List[List[Tuple[str, str]]],
    List[List[Tuple[str, str]]],
    int,
]:
    """Load Brown news corpus with universal tags, shuffle, and split 80/20."""
    from nltk.corpus import brown

    tagged_sents = brown.tagged_sents(categories="news", tagset="universal")
    tagged_sents = list(tagged_sents)
    total = len(tagged_sents)

    random.seed(RANDOM_SEED)
    random.shuffle(tagged_sents)

    split_idx = int(total * TRAIN_RATIO)
    train_sents = tagged_sents[:split_idx]
    test_sents = tagged_sents[split_idx:]

    return train_sents, test_sents, total


def count_tokens(sentences: List[List[Tuple[str, str]]]) -> int:
    """Count total tokens across tagged sentences."""
    return sum(len(s) for s in sentences)


# ---------------------------------------------------------------------------
# Taggers
# ---------------------------------------------------------------------------
def build_default_tagger() -> "nltk.tag.DefaultTagger":
    """Build a DefaultTagger that assigns NOUN to every token."""
    from nltk.tag import DefaultTagger
    return DefaultTagger("NOUN")


def build_unigram_tagger(
    train_sents: List[List[Tuple[str, str]]],
    backoff: "nltk.tag.TaggerI",
) -> "nltk.tag.UnigramTagger":
    """Train a UnigramTagger with a backoff tagger."""
    from nltk.tag import UnigramTagger
    return UnigramTagger(train_sents, backoff=backoff)


def build_bigram_tagger(
    train_sents: List[List[Tuple[str, str]]],
    backoff: "nltk.tag.TaggerI",
) -> "nltk.tag.BigramTagger":
    """Train a BigramTagger with a backoff tagger."""
    from nltk.tag import BigramTagger
    return BigramTagger(train_sents, backoff=backoff)


def build_trigram_tagger(
    train_sents: List[List[Tuple[str, str]]],
    backoff: "nltk.tag.TaggerI",
) -> "nltk.tag.TrigramTagger":
    """Train a TrigramTagger with a backoff tagger."""
    from nltk.tag import TrigramTagger
    return TrigramTagger(train_sents, backoff=backoff)


def tag_with_perceptron(
    test_sents: List[List[Tuple[str, str]]],
) -> List[List[Tuple[str, str]]]:
    """Tag sentences using NLTK's pre-trained averaged perceptron tagger."""
    import nltk

    results = []
    for sent in test_sents:
        tokens = [w for w, _ in sent]
        tagged = nltk.pos_tag(tokens, tagset="universal")
        results.append(tagged)
    return results


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def flatten_tags(
    tagged_sents: List[List[Tuple[str, str]]],
) -> Tuple[List[str], List[str]]:
    """Flatten list of tagged sentences into parallel token/tag lists."""
    tokens, tags = [], []
    for sent in tagged_sents:
        for word, tag in sent:
            tokens.append(word)
            tags.append(tag)
    return tokens, tags


def evaluate_tagger(
    actual_tags: List[str],
    predicted_tags: List[str],
) -> Dict[str, float]:
    """Compute accuracy, macro precision/recall/F1."""
    acc = accuracy_score(actual_tags, predicted_tags)
    prec, rec, f1, _ = precision_recall_fscore_support(
        actual_tags, predicted_tags, average="macro", zero_division=0,
    )
    return {
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
    }


def run_all_taggers(
    train_sents: List[List[Tuple[str, str]]],
    test_sents: List[List[Tuple[str, str]]],
) -> List[Dict]:
    """Train/run all five taggers and collect metrics."""
    _, actual_tags = flatten_tags(test_sents)
    results = []

    # --- A. Default Tagger ---
    t0 = time.perf_counter()
    default_tagger = build_default_tagger()
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_sents = [default_tagger.tag([w for w, _ in s]) for s in test_sents]
    tag_time = time.perf_counter() - t0
    _, pred_tags = flatten_tags(pred_sents)

    metrics = evaluate_tagger(actual_tags, pred_tags)
    results.append({
        "method": "Default Tagger",
        "pred_tags": pred_tags,
        **metrics,
        "training_seconds": train_time,
        "tagging_seconds": tag_time,
    })

    # --- B. Unigram Tagger ---
    t0 = time.perf_counter()
    unigram_tagger = build_unigram_tagger(train_sents, default_tagger)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_sents = [unigram_tagger.tag([w for w, _ in s]) for s in test_sents]
    tag_time = time.perf_counter() - t0
    _, pred_tags = flatten_tags(pred_sents)

    metrics = evaluate_tagger(actual_tags, pred_tags)
    results.append({
        "method": "Unigram Tagger",
        "pred_tags": pred_tags,
        **metrics,
        "training_seconds": train_time,
        "tagging_seconds": tag_time,
    })

    # --- C. Bigram Tagger ---
    t0 = time.perf_counter()
    bigram_tagger = build_bigram_tagger(train_sents, unigram_tagger)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_sents = [bigram_tagger.tag([w for w, _ in s]) for s in test_sents]
    tag_time = time.perf_counter() - t0
    _, pred_tags = flatten_tags(pred_sents)

    metrics = evaluate_tagger(actual_tags, pred_tags)
    results.append({
        "method": "Bigram Tagger",
        "pred_tags": pred_tags,
        **metrics,
        "training_seconds": train_time,
        "tagging_seconds": tag_time,
    })

    # --- D. Trigram Tagger ---
    t0 = time.perf_counter()
    trigram_tagger = build_trigram_tagger(train_sents, bigram_tagger)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred_sents = [trigram_tagger.tag([w for w, _ in s]) for s in test_sents]
    tag_time = time.perf_counter() - t0
    _, pred_tags = flatten_tags(pred_sents)

    metrics = evaluate_tagger(actual_tags, pred_tags)
    results.append({
        "method": "Trigram Tagger",
        "pred_tags": pred_tags,
        **metrics,
        "training_seconds": train_time,
        "tagging_seconds": tag_time,
    })

    # --- E. Averaged Perceptron Tagger (pre-trained) ---
    t0 = time.perf_counter()
    pred_sents = tag_with_perceptron(test_sents)
    tag_time = time.perf_counter() - t0
    _, pred_tags = flatten_tags(pred_sents)

    metrics = evaluate_tagger(actual_tags, pred_tags)
    results.append({
        "method": "Averaged Perceptron Tagger",
        "pred_tags": pred_tags,
        **metrics,
        "training_seconds": 0.0,
        "tagging_seconds": tag_time,
    })

    return results


# ---------------------------------------------------------------------------
# Per-tag analysis for best tagger
# ---------------------------------------------------------------------------
def per_tag_analysis(
    actual_tags: List[str],
    predicted_tags: List[str],
) -> Tuple[List[Dict], np.ndarray, List[str]]:
    """Compute per-tag metrics and confusion matrix for the best tagger."""
    present_tags = sorted(
        set(actual_tags) | set(predicted_tags),
        key=lambda t: UNIVERSAL_TAG_ORDER.index(t) if t in UNIVERSAL_TAG_ORDER else len(UNIVERSAL_TAG_ORDER),
    )

    report = classification_report(
        actual_tags, predicted_tags,
        labels=present_tags,
        output_dict=True,
        zero_division=0,
    )

    per_tag = []
    for tag in present_tags:
        entry = report[tag]
        per_tag.append({
            "tag": tag,
            "precision": entry["precision"],
            "recall": entry["recall"],
            "f1_score": entry["f1-score"],
            "support": int(entry["support"]),
        })

    cm = confusion_matrix(actual_tags, predicted_tags, labels=present_tags)
    return per_tag, cm, present_tags


# ---------------------------------------------------------------------------
# Custom sentence tagging
# ---------------------------------------------------------------------------
def tag_custom_sentence(sentence: str, best_method: str, best_result: Dict,
                        train_sents: List[List[Tuple[str, str]]]) -> List[Tuple[str, str, str]]:
    """Tag a custom sentence with the best-performing approach."""
    import nltk

    tokens = nltk.word_tokenize(sentence)

    if best_method == "Averaged Perceptron Tagger":
        tagged = nltk.pos_tag(tokens, tagset="universal")
    else:
        # Rebuild the tagger chain
        default_t = build_default_tagger()
        unigram_t = build_unigram_tagger(train_sents, default_t)
        bigram_t = build_bigram_tagger(train_sents, unigram_t)
        trigram_t = build_trigram_tagger(train_sents, bigram_t)

        tagger_map = {
            "Default Tagger": default_t,
            "Unigram Tagger": unigram_t,
            "Bigram Tagger": bigram_t,
            "Trigram Tagger": trigram_t,
        }
        tagger = tagger_map[best_method]
        tagged = tagger.tag(tokens)

    return [(w, t, TAG_NAMES.get(t, t)) for w, t in tagged]


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def save_comparison_csv(results: List[Dict], path: Path) -> None:
    """Write the five-tagger comparison CSV."""
    fieldnames = [
        "method", "accuracy", "macro_precision", "macro_recall",
        "macro_f1", "training_seconds", "tagging_seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})


def save_per_tag_csv(per_tag: List[Dict], path: Path) -> None:
    """Write per-tag metrics CSV."""
    fieldnames = ["tag", "precision", "recall", "f1_score", "support"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in per_tag:
            writer.writerow(row)


def save_accuracy_plot(results: List[Dict], path: Path) -> None:
    """Bar chart comparing macro F1-score for all five taggers."""
    methods = [r["method"] for r in results]
    f1_scores = [r["macro_f1"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(methods)), f1_scores, color="steelblue", edgecolor="black")

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Macro F1-Score")
    ax.set_xlabel("POS Tagging Method")
    ax.set_title("POS Tagging — Macro F1-Score Comparison")
    ax.set_ylim(0, min(1.05, max(f1_scores) + 0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(
    cm: np.ndarray, labels: List[str], path: Path,
) -> None:
    """Confusion matrix heatmap for the best tagger."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)

    ax.set_xlabel("Predicted Tag")
    ax.set_ylabel("Actual Tag")
    ax.set_title("Confusion Matrix — Best POS Tagger (Universal Tagset)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_output_txt(
    total: int,
    train_sents: List,
    test_sents: List,
    train_tokens: int,
    test_tokens: int,
    results: List[Dict],
    best: Dict,
    per_tag: List[Dict],
    custom_sentence: str,
    custom_analysis: List[Tuple[str, str, str]],
    path: Path,
) -> str:
    """Generate the comprehensive output text report. Returns the text."""
    lines: List[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)

    section("STUDENT DETAILS")
    lines.append(f"Student Name    : {STUDENT_NAME}")
    lines.append(f"Register Number : {REGISTER_NUMBER}")

    section("EXPERIMENT TITLE")
    lines.append(EXPERIMENT_TITLE)

    section("AIM")
    lines.append(EXPERIMENT_AIM)

    section("DATASET INFORMATION")
    lines.append(f"Corpus          : Brown Corpus (category='news')")
    lines.append(f"Tagset          : Universal POS Tagset")
    lines.append(f"Total sentences : {total}")
    lines.append(f"Training split  : {len(train_sents)} sentences ({train_tokens} tokens)")
    lines.append(f"Testing split   : {len(test_sents)} sentences ({test_tokens} tokens)")

    section("EVALUATION RESULTS")
    header = (
        f"{'Method':<32} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
        f"{'F1':>8} {'Train(s)':>9} {'Tag(s)':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        note = " *" if r["training_seconds"] == 0.0 and r["method"] == "Averaged Perceptron Tagger" else ""
        lines.append(
            f"{r['method']:<32} {r['accuracy']:>9.4f} {r['macro_precision']:>10.4f} "
            f"{r['macro_recall']:>8.4f} {r['macro_f1']:>8.4f} "
            f"{r['training_seconds']:>9.4f} {r['tagging_seconds']:>8.4f}{note}"
        )
    lines.append("")
    lines.append("* Pre-trained model (training time = 0)")

    section("BEST-PERFORMING TAGGER")
    lines.append(f"Method : {best['method']}")
    lines.append(f"Macro F1-Score : {best['macro_f1']:.4f}")

    section("PER-TAG EVALUATION (Best Tagger)")
    tag_header = f"{'Tag':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}"
    lines.append(tag_header)
    lines.append("-" * len(tag_header))
    for pt in per_tag:
        lines.append(
            f"{pt['tag']:<8} {pt['precision']:>10.4f} {pt['recall']:>8.4f} "
            f"{pt['f1_score']:>8.4f} {pt['support']:>8d}"
        )

    section("CUSTOM SENTENCE ANALYSIS")
    lines.append(f"Sentence : {custom_sentence}")
    lines.append(f"Tagger   : {best['method']}")
    lines.append("")
    lines.append(f"{'Token':<20} {'Tag':<8} {'Category':<15}")
    lines.append("-" * 43)
    for word, tag, name in custom_analysis:
        lines.append(f"{word:<20} {tag:<8} {name:<15}")

    section("RESULT")
    lines.append(
        f"The {best['method']} achieved the highest macro F1-score of "
        f"{best['macro_f1']:.4f} among the five POS tagging approaches evaluated "
        f"on the Brown news corpus with the Universal tagset."
    )

    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------
def print_summary(results: List[Dict], best: Dict,
                  custom_sentence: str,
                  custom_analysis: List[Tuple[str, str, str]]) -> None:
    """Print a concise summary to the terminal."""
    print("\n" + "=" * 70)
    print("POS TAGGING — EVALUATION SUMMARY")
    print("=" * 70)

    header = f"{'Method':<32} {'Accuracy':>9} {'Macro F1':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['method']:<32} {r['accuracy']:>9.4f} {r['macro_f1']:>9.4f}")

    print(f"\nBest tagger: {best['method']} (Macro F1 = {best['macro_f1']:.4f})")

    print("\n" + "=" * 70)
    print("CUSTOM SENTENCE ANALYSIS")
    print("=" * 70)
    print(f"Sentence: {custom_sentence}")
    print(f"Tagger  : {best['method']}")
    print()
    print(f"{'Token':<20} {'Tag':<8} {'Category':<15}")
    print("-" * 43)
    for word, tag, name in custom_analysis:
        print(f"{word:<20} {tag:<8} {name:<15}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for Experiment 02."""
    parser = argparse.ArgumentParser(
        description="POS Tagging experiment using NLTK on the Brown corpus."
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Custom sentence to tag (default: built-in sample).",
    )
    args = parser.parse_args()
    custom_sentence = args.text if args.text else DEFAULT_SENTENCE

    # Resolve output directory relative to this script
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure NLTK data
    print("Checking NLTK resources...")
    ensure_nltk_resources()

    # Load and split data
    print("Loading Brown corpus (news, universal tagset)...")
    train_sents, test_sents, total = load_and_split_data()
    train_tokens = count_tokens(train_sents)
    test_tokens = count_tokens(test_sents)

    print(f"  Total sentences : {total}")
    print(f"  Training        : {len(train_sents)} sentences ({train_tokens} tokens)")
    print(f"  Testing         : {len(test_sents)} sentences ({test_tokens} tokens)")

    # Run all taggers
    print("\nTraining and evaluating taggers...")
    results = run_all_taggers(train_sents, test_sents)

    # Determine best by macro F1
    best = max(results, key=lambda r: r["macro_f1"])

    # Per-tag analysis on best tagger
    _, actual_tags = flatten_tags(test_sents)
    per_tag, cm, cm_labels = per_tag_analysis(actual_tags, best["pred_tags"])

    # Custom sentence
    custom_analysis = tag_custom_sentence(
        custom_sentence, best["method"], best, train_sents,
    )

    # Save outputs
    print("Saving output files...")
    save_comparison_csv(results, output_dir / "comparison.csv")
    save_per_tag_csv(per_tag, output_dir / "per_tag_metrics.csv")
    save_accuracy_plot(results, output_dir / "accuracy_plot.png")
    save_confusion_matrix_plot(cm, cm_labels, output_dir / "confusion_matrix.png")

    report_text = generate_output_txt(
        total, train_sents, test_sents, train_tokens, test_tokens,
        results, best, per_tag, custom_sentence, custom_analysis,
        output_dir / "output.txt",
    )

    # Terminal output
    print_summary(results, best, custom_sentence, custom_analysis)
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
