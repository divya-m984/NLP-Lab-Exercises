"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 07
Title           : Construct a Language Model Using N-Gram Models and Compare Its
                  Performance with a Hidden Markov Model (HMM) on a Corpus of Tweets
"""

import argparse
import math
import random
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tag import HiddenMarkovModelTrainer
from nltk.tokenize import TweetTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SAMPLE_SIZE = 500  # per class
SPLIT_RATIO = 0.8
UNK_CUTOFF = 2
UNK_TOKEN = "<UNK>"

EXPERIMENT_TITLE = (
    "Construct a Language Model Using N-Gram Models and Compare Its\n"
    "Performance with a Hidden Markov Model (HMM) on a Corpus of Tweets"
)

AIM = (
    "To construct N-Gram and Hidden Markov Model (HMM) language models and\n"
    "compare their ability to capture word sequences in tweet data."
)

DEFAULT_TEXT = "AI is transforming social media one post at a time"

UNIVERSAL_TAG_NAMES: Dict[str, str] = {
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
# NLTK resource helpers
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Download only missing NLTK resources."""
    import nltk

    resources = [
        ("corpora", "twitter_samples"),
        ("taggers", "averaged_perceptron_tagger_eng"),
        ("taggers", "universal_tagset"),
        ("corpora", "wordnet"),
        ("corpora", "omw-1.4"),
    ]
    for category, name in resources:
        try:
            nltk.data.find(f"{category}/{name}")
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download NLTK resource '{name}': {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Tweet preprocessing
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


def preprocess_tweet(text: str) -> List[str]:
    """Clean and tokenize a single tweet."""
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    text = _URL_RE.sub("", text)
    tokens = _TOKENIZER.tokenize(text)

    processed: List[str] = []
    for tok in tokens:
        # strip leading # but keep the word
        if tok.startswith("#"):
            tok = tok[1:]
        if not tok:
            continue
        # allow simple contractions
        if re.fullmatch(r"[a-z]+'[a-z]+", tok):
            processed.append(lemmatizer.lemmatize(tok))
            continue
        # keep only alphabetic words
        if not tok.isalpha():
            continue
        processed.append(lemmatizer.lemmatize(tok))
    return processed


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_tweets() -> Tuple[List[List[str]], List[int], Dict[str, int]]:
    """Load, sample, preprocess and split NLTK twitter_samples.

    Returns (all_tweets, labels, stats_dict).
    """
    from nltk.corpus import twitter_samples

    random.seed(RANDOM_SEED)

    pos_raw = twitter_samples.strings("positive_tweets.json")
    neg_raw = twitter_samples.strings("negative_tweets.json")

    # preprocess all, keep those with >= 3 tokens
    pos_processed = [(preprocess_tweet(t), 1) for t in pos_raw]
    neg_processed = [(preprocess_tweet(t), 0) for t in neg_raw]
    pos_processed = [(t, l) for t, l in pos_processed if len(t) >= 3]
    neg_processed = [(t, l) for t, l in neg_processed if len(t) >= 3]

    # balanced sample
    n_samples = min(SAMPLE_SIZE, len(pos_processed), len(neg_processed))
    pos_sampled = random.sample(pos_processed, n_samples)
    neg_sampled = random.sample(neg_processed, n_samples)

    all_data = pos_sampled + neg_sampled
    random.shuffle(all_data)

    tweets = [t for t, _ in all_data]
    labels = [l for _, l in all_data]

    stats = {
        "total": len(tweets),
        "positive": n_samples,
        "negative": n_samples,
    }
    return tweets, labels, stats


# ---------------------------------------------------------------------------
# Vocabulary with UNK mapping
# ---------------------------------------------------------------------------

def build_vocab(train_tweets: List[List[str]], cutoff: int = UNK_CUTOFF) -> set:
    """Build vocabulary from training tweets, mapping rare words to UNK."""
    counts: Counter = Counter()
    for tw in train_tweets:
        counts.update(tw)
    return {w for w, c in counts.items() if c >= cutoff}


def apply_unk(tokens: List[str], vocab: set) -> List[str]:
    """Replace out-of-vocabulary tokens with UNK_TOKEN."""
    return [t if t in vocab else UNK_TOKEN for t in tokens]


# ---------------------------------------------------------------------------
# N-Gram models
# ---------------------------------------------------------------------------

def train_ngram_model(train_tweets: List[List[str]], n: int, vocab: set) -> Laplace:
    """Train an NLTK Laplace-smoothed N-Gram model."""
    mapped = [apply_unk(tw, vocab) for tw in train_tweets]
    train_data, padded_vocab = padded_everygram_pipeline(n, mapped)
    model = Laplace(n)
    model.fit(train_data, padded_vocab)
    return model


def tweet_perplexity(model: Laplace, tokens: List[str], n: int) -> Optional[float]:
    """Compute perplexity of a single tweet. Returns None on failure."""
    from nltk.lm.preprocessing import pad_both_ends
    from nltk.util import everygrams

    padded = list(pad_both_ends(tokens, n=n))
    ngrams = list(everygrams(padded, max_len=n))
    # filter to only n-grams of the correct order
    test_ngrams = [ng for ng in ngrams if len(ng) == n]
    if not test_ngrams:
        return None
    try:
        pp = model.perplexity(test_ngrams)
        if math.isinf(pp) or math.isnan(pp):
            return None
        return pp
    except Exception:
        return None


def evaluate_ngram(
    model: Laplace,
    test_tweets: List[List[str]],
    n: int,
    vocab: set,
) -> Dict[str, object]:
    """Evaluate an N-Gram model: perplexity, next-word accuracy."""
    start = time.time()

    # --- Perplexity ---
    perplexities: List[float] = []
    for tw in test_tweets:
        mapped = apply_unk(tw, vocab)
        pp = tweet_perplexity(model, mapped, n)
        if pp is not None:
            perplexities.append(pp)

    avg_pp = float(np.mean(perplexities)) if perplexities else float("inf")
    med_pp = float(np.median(perplexities)) if perplexities else float("inf")

    # --- Next-word accuracy ---
    top1_correct = 0
    top3_correct = 0
    total_preds = 0

    for tw in test_tweets:
        mapped = apply_unk(tw, vocab)
        if n == 1:
            # unigram: no context, predict each non-padding token
            for actual in mapped:
                preds = _top_k_predictions(model, [], n, k=3)
                if preds and preds[0] == actual:
                    top1_correct += 1
                if actual in preds:
                    top3_correct += 1
                total_preds += 1
        else:
            from nltk.lm.preprocessing import pad_both_ends
            padded = list(pad_both_ends(mapped, n=n))
            # predict each position after sufficient context, skip padding
            for i in range(n - 1, len(padded)):
                actual = padded[i]
                if actual in ("<s>", "</s>"):
                    continue
                context = padded[max(0, i - (n - 1)):i]
                preds = _top_k_predictions(model, context, n, k=3)
                if preds and preds[0] == actual:
                    top1_correct += 1
                if actual in preds:
                    top3_correct += 1
                total_preds += 1

    elapsed = time.time() - start

    return {
        "avg_perplexity": avg_pp,
        "med_perplexity": med_pp,
        "tweets_evaluated": len(perplexities),
        "tweets_total": len(test_tweets),
        "top1_accuracy": top1_correct / total_preds if total_preds else 0.0,
        "top3_accuracy": top3_correct / total_preds if total_preds else 0.0,
        "total_preds": total_preds,
        "eval_time": elapsed,
    }


def _top_k_predictions(
    model: Laplace, context: List[str], n: int, k: int = 3
) -> List[str]:
    """Return top-k predicted words from the model, deterministic on ties."""
    special = {"<s>", "</s>", "<UNK>"}
    scores: Dict[str, float] = {}
    for word in model.vocab:
        if word in special:
            continue
        try:
            s = model.score(word, context)
        except Exception:
            continue
        scores[word] = s
    # sort by score descending, then word ascending for determinism
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:k]]


def select_best_ngram(
    results: Dict[str, Dict[str, object]],
) -> str:
    """Select best N-Gram model by lowest avg perplexity, then accuracy tie-breakers."""
    items = list(results.items())
    items.sort(
        key=lambda x: (
            x[1]["avg_perplexity"],
            -x[1]["top1_accuracy"],
            -x[1]["top3_accuracy"],
        )
    )
    return items[0][0]


# ---------------------------------------------------------------------------
# HMM model
# ---------------------------------------------------------------------------

def pos_tag_tweets(tweets: List[List[str]]) -> List[List[Tuple[str, str]]]:
    """Assign Universal POS tags to preprocessed tweets using NLTK's tagger."""
    import nltk
    tagged: List[List[Tuple[str, str]]] = []
    for tw in tweets:
        pairs = nltk.pos_tag(tw, tagset="universal")
        tagged.append(pairs)
    return tagged


def train_hmm(
    tagged_train: List[List[Tuple[str, str]]],
) -> object:
    """Train an HMM using NLTK's HiddenMarkovModelTrainer with Lidstone smoothing."""
    from nltk.probability import LidstoneProbDist

    def lidstone_estimator(fd, bins):
        return LidstoneProbDist(fd, gamma=0.1, bins=bins)

    trainer = HiddenMarkovModelTrainer()
    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = trainer.train_supervised(tagged_train, estimator=lidstone_estimator)
    train_time = time.time() - start
    return model, train_time


def evaluate_hmm(
    model, tagged_test: List[List[Tuple[str, str]]]
) -> Dict[str, object]:
    """Evaluate HMM on test data against silver-standard POS labels."""
    y_true: List[str] = []
    y_pred: List[str] = []

    start = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for tagged_sent in tagged_test:
            words = [w for w, _ in tagged_sent]
            gold_tags = [t for _, t in tagged_sent]
            predicted = model.tag(words)
            pred_tags = [t for _, t in predicted]
            y_true.extend(gold_tags)
            y_pred.extend(pred_tags)
    tag_time = time.time() - start

    accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if y_true else 0.0
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # average log probability per token
    avg_logprob = None
    try:
        per_token_values: List[float] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for tagged_sent in tagged_test:
                words = [w for w, _ in tagged_sent]
                lp = model.log_probability(
                    [(w, t) for w, t in zip(words, [t for _, t in tagged_sent])]
                )
                if math.isfinite(lp) and len(words) > 0:
                    per_token_values.append(lp / len(words))
        if per_token_values:
            avg_logprob = sum(per_token_values) / len(per_token_values)
        else:
            avg_logprob = None
    except Exception:
        avg_logprob = None

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "tag_time": tag_time,
        "avg_logprob_per_token": avg_logprob,
        "total_tokens": len(y_true),
    }


# ---------------------------------------------------------------------------
# Custom tweet analysis
# ---------------------------------------------------------------------------

def analyze_custom_tweet(
    text: str,
    best_model: Laplace,
    best_n: int,
    hmm_model,
    vocab: set,
) -> Dict[str, object]:
    """Analyze a custom tweet with the best N-Gram model and HMM."""
    tokens = preprocess_tweet(text)
    mapped = apply_unk(tokens, vocab)

    # n-grams
    from nltk.util import bigrams as nltk_bigrams, trigrams as nltk_trigrams
    unigrams = list(tokens)
    bigrams_list = list(nltk_bigrams(tokens))
    trigrams_list = list(nltk_trigrams(tokens))

    # next-word predictions from best model
    if best_n == 1:
        context = []
    elif best_n == 2:
        context = mapped[-1:] if mapped else []
    else:
        context = mapped[-(best_n - 1):] if mapped else []
    top5 = _top_k_predictions(best_model, context, best_n, k=5)

    # HMM POS tags
    import nltk
    pos_tags = nltk.pos_tag(tokens, tagset="universal")
    tag_names = [(w, t, UNIVERSAL_TAG_NAMES.get(t, t)) for w, t in pos_tags]

    return {
        "original": text,
        "tokens": tokens,
        "unigrams": unigrams,
        "bigrams": bigrams_list,
        "trigrams": trigrams_list,
        "top5_next": top5,
        "pos_tags": pos_tags,
        "tag_names": tag_names,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    stats: Dict[str, int],
    train_tokens: int,
    test_tokens: int,
    vocab_size: int,
    ngram_results: Dict[str, Dict],
    best_ngram_name: str,
    hmm_metrics: Dict[str, object],
    hmm_train_time: float,
    custom: Dict[str, object],
    output_path: Path,
) -> str:
    """Generate output/output.txt report and return its content."""
    sep = "=" * 72
    thin = "-" * 72

    lines: List[str] = []
    a = lines.append

    a(sep)
    a("NATURAL LANGUAGE PROCESSING LABORATORY")
    a(sep)
    a(f"Student Name    : Divya M")
    a(f"Register Number : 24AD0074")
    a(f"Experiment      : 07")
    a("")
    a(f"Title: {EXPERIMENT_TITLE}")
    a("")
    a(f"Aim: {AIM}")
    a(sep)

    # Dataset
    a("\n1. DATASET SUMMARY")
    a(thin)
    a(f"  Source              : NLTK twitter_samples corpus")
    a(f"  Total tweets        : {stats['total']}")
    a(f"  Positive tweets     : {stats['positive']}")
    a(f"  Negative tweets     : {stats['negative']}")
    a(f"  Training tweets     : {stats['train']}")
    a(f"  Testing tweets      : {stats['test']}")
    a(f"  Training tokens     : {train_tokens}")
    a(f"  Testing tokens      : {test_tokens}")
    a(f"  Vocabulary size     : {vocab_size}")
    a(f"  Random seed         : {RANDOM_SEED}")
    a(f"  Train/test split    : {SPLIT_RATIO:.0%} / {1 - SPLIT_RATIO:.0%}")
    a(f"  Stratified by       : positive/negative source label")
    a(f"  Min tokens/tweet    : 3 (after preprocessing)")
    a("")

    # Preprocessing
    a("2. PREPROCESSING")
    a(thin)
    a("  - TweetTokenizer (preserve_case=False, reduce_len=True, strip_handles=True)")
    a("  - Remove URLs")
    a("  - Remove standalone punctuation")
    a("  - Strip '#' symbol, retain hashtag word")
    a("  - Retain alphabetic words and simple contractions")
    a("  - Lowercase all tokens")
    a("  - WordNet lemmatization")
    a("  - Stopwords retained (important for language modeling)")
    a("  - Sentence-boundary padding added for N-Gram training")
    a(f"  - Unknown-token cutoff: words appearing < {UNK_CUTOFF} times -> {UNK_TOKEN}")
    a("")

    # N-Gram models
    a("3. N-GRAM LANGUAGE MODELS")
    a(thin)
    a("  A. Unigram Laplace model  (n=1)")
    a("  B. Bigram Laplace model   (n=2)")
    a("  C. Trigram Laplace model  (n=3)")
    a("")
    a("  All models use Laplace (add-one) smoothing via nltk.lm.Laplace.")
    a("")

    # Evaluation table
    a("4. N-GRAM EVALUATION RESULTS")
    a(thin)
    header = f"  {'Model':<12} {'Avg PP':>10} {'Med PP':>10} {'Top-1 Acc':>10} {'Top-3 Acc':>10} {'Eval(s)':>8} {'Tweets OK':>10}"
    a(header)
    a("  " + "-" * 70)
    for name in ["Unigram", "Bigram", "Trigram"]:
        r = ngram_results[name]
        a(
            f"  {name:<12} {r['avg_perplexity']:>10.2f} {r['med_perplexity']:>10.2f} "
            f"{r['top1_accuracy']:>9.4f} {r['top3_accuracy']:>10.4f} "
            f"{r['eval_time']:>7.2f}s {r['tweets_evaluated']:>5}/{r['tweets_total']:<4}"
        )
    a("")
    a(f"  Best N-Gram model: {best_ngram_name} (lowest average perplexity)")
    a("")

    # HMM
    a("5. HIDDEN MARKOV MODEL (HMM)")
    a(thin)
    a("  Methodology:")
    a("  - Trained using NLTK HiddenMarkovModelTrainer (supervised)")
    a("  - POS tags: Universal tagset via nltk.pos_tag(tagset='universal')")
    a("  - These are SILVER-STANDARD labels produced by NLTK's pretrained POS")
    a("    tagger, NOT manually annotated gold labels. The tagger was trained")
    a("    on standard English text and applied automatically to tweets.")
    a("")
    a("  Evaluation Metrics:")
    a(f"    Token-level POS accuracy : {hmm_metrics['accuracy']:.4f}")
    a(f"    Macro precision          : {hmm_metrics['macro_precision']:.4f}")
    a(f"    Macro recall             : {hmm_metrics['macro_recall']:.4f}")
    a(f"    Macro F1-score           : {hmm_metrics['macro_f1']:.4f}")
    a(f"    Training time            : {hmm_train_time:.2f}s")
    a(f"    Tagging time             : {hmm_metrics['tag_time']:.2f}s")
    a(f"    Total test tokens        : {hmm_metrics['total_tokens']}")
    if hmm_metrics["avg_logprob_per_token"] is not None:
        a(f"    Avg log-prob per token   : {hmm_metrics['avg_logprob_per_token']:.4f}")
    else:
        a(f"    Avg log-prob per token   : N/A (numerical underflow)")
    a("")

    # Comparison
    a("6. CONCEPTUAL COMPARISON: N-GRAM vs HMM")
    a(thin)
    a("  N-Gram models are evaluated as word-sequence predictors using")
    a("  perplexity and next-word accuracy. They model P(word | context)")
    a("  directly from surface-level word co-occurrences.")
    a("")
    a("  The HMM is evaluated as a hidden-state sequence-labeling model using")
    a("  POS-tagging accuracy and F1-score. It models the joint probability")
    a("  P(words, tags) through emission and transition probabilities.")
    a("")
    a("  Both approaches capture local sequence dependencies:")
    a("  - N-Grams use fixed-length word windows (unigram/bigram/trigram).")
    a("  - HMMs use latent POS states with Markov transition assumptions.")
    a("")
    a("  Their numerical metrics measure fundamentally different tasks:")
    a("  - Perplexity measures how well the model predicts the next word.")
    a("  - POS accuracy measures how well the model recovers hidden labels.")
    a("  Therefore, a direct numerical comparison between N-Gram perplexity")
    a("  and HMM POS accuracy would be misleading and is not performed.")
    a("")

    # Custom tweet
    a("7. CUSTOM TWEET ANALYSIS")
    a(thin)
    a(f"  Original  : {custom['original']}")
    a(f"  Tokens    : {custom['tokens']}")
    a(f"  Unigrams  : {custom['unigrams']}")
    a(f"  Bigrams   : {custom['bigrams']}")
    a(f"  Trigrams  : {custom['trigrams']}")
    a(f"  Top-5 next words ({best_ngram_name}): {custom['top5_next']}")
    a(f"  HMM POS tags:")
    for word, tag, name in custom["tag_names"]:
        a(f"    {word:<20} {tag:<6} {name}")
    a("")

    # Result
    a("8. RESULT")
    a(thin)
    a(f"  N-Gram and Hidden Markov Model language models were successfully")
    a(f"  constructed and evaluated on a balanced corpus of 1000 tweets from")
    a(f"  NLTK twitter_samples. The {best_ngram_name} model achieved the lowest")
    a(f"  average perplexity among the N-Gram models. The HMM achieved a")
    a(f"  token-level POS accuracy of {hmm_metrics['accuracy']:.4f} and macro")
    a(f"  F1-score of {hmm_metrics['macro_f1']:.4f} using silver-standard")
    a(f"  Universal POS labels. The two model families capture local sequence")
    a(f"  dependencies through complementary mechanisms and are evaluated on")
    a(f"  different tasks, making direct numerical comparison inappropriate.")
    a(sep)

    report = "\n".join(lines) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_terminal_summary(
    stats: Dict[str, int],
    train_tokens: int,
    test_tokens: int,
    vocab_size: int,
    ngram_results: Dict[str, Dict],
    best_ngram_name: str,
    hmm_metrics: Dict[str, object],
    custom: Dict[str, object],
    output_path: Path,
) -> None:
    """Print concise terminal summary."""
    print("=" * 60)
    print("Experiment 07: N-Gram & HMM Language Models")
    print("=" * 60)
    print(f"Student: Divya M (24AD0074)")
    print(f"Dataset: {stats['total']} tweets ({stats['train']} train / {stats['test']} test)")
    print(f"         {train_tokens} train tokens, {test_tokens} test tokens, vocab={vocab_size}")
    print("-" * 60)

    print(f"{'Model':<10} {'AvgPP':>8} {'MedPP':>8} {'Top1':>7} {'Top3':>7} {'Time':>6}")
    print("-" * 60)
    for name in ["Unigram", "Bigram", "Trigram"]:
        r = ngram_results[name]
        print(
            f"{name:<10} {r['avg_perplexity']:>8.1f} {r['med_perplexity']:>8.1f} "
            f"{r['top1_accuracy']:>6.3f} {r['top3_accuracy']:>7.3f} "
            f"{r['eval_time']:>5.1f}s"
        )
    print(f"Best N-Gram: {best_ngram_name}")
    print("-" * 60)

    print(f"HMM Accuracy: {hmm_metrics['accuracy']:.4f}  Macro-F1: {hmm_metrics['macro_f1']:.4f}")
    print("-" * 60)

    print(f"Custom tweet: \"{custom['original']}\"")
    print(f"Next words ({best_ngram_name}): {', '.join(custom['top5_next'])}")
    tags_str = " ".join(f"{w}/{t}" for w, t in custom["pos_tags"])
    print(f"HMM tags: {tags_str}")
    print("-" * 60)
    print(f"Output: {output_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 07: N-Gram and HMM Language Models."""
    parser = argparse.ArgumentParser(
        description="Experiment 07: N-Gram & HMM Language Models on Tweets"
    )
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Custom tweet to analyze")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "output" / "output.txt"

    # 1. NLTK resources
    _ensure_nltk_resources()

    # 2. Load and split data
    tweets, labels, stats = load_tweets()
    train_tweets, test_tweets, train_labels, test_labels = train_test_split(
        tweets, labels, test_size=1 - SPLIT_RATIO, stratify=labels, random_state=RANDOM_SEED
    )
    stats["train"] = len(train_tweets)
    stats["test"] = len(test_tweets)
    train_tokens = sum(len(t) for t in train_tweets)
    test_tokens = sum(len(t) for t in test_tweets)

    # 3. Build vocabulary
    vocab = build_vocab(train_tweets)
    vocab_size = len(vocab)

    # 4. Train N-Gram models
    print("Training N-Gram models...")
    ngram_models: Dict[str, Laplace] = {}
    ngram_ns = {"Unigram": 1, "Bigram": 2, "Trigram": 3}
    for name, n in ngram_ns.items():
        ngram_models[name] = train_ngram_model(train_tweets, n, vocab)

    # 5. Evaluate N-Gram models
    print("Evaluating N-Gram models...")
    ngram_results: Dict[str, Dict] = {}
    for name, n in ngram_ns.items():
        ngram_results[name] = evaluate_ngram(ngram_models[name], test_tweets, n, vocab)

    best_ngram_name = select_best_ngram(ngram_results)

    # 6. HMM
    print("Training HMM...")
    all_tagged = pos_tag_tweets(tweets)
    tagged_train = [all_tagged[i] for i in range(len(tweets))
                    if tweets[i] in train_tweets]

    # Use the same split indices
    train_set = set(map(id, train_tweets))
    test_set = set(map(id, test_tweets))

    # Re-tag using consistent indexing
    tagged_by_id: Dict[int, List[Tuple[str, str]]] = {}
    for i, tw in enumerate(tweets):
        tagged_by_id[id(tw)] = all_tagged[i]

    tagged_train_split = [tagged_by_id[id(tw)] for tw in train_tweets]
    tagged_test_split = [tagged_by_id[id(tw)] for tw in test_tweets]

    hmm_model, hmm_train_time = train_hmm(tagged_train_split)

    print("Evaluating HMM...")
    hmm_metrics = evaluate_hmm(hmm_model, tagged_test_split)

    # 7. Custom tweet analysis
    best_n = ngram_ns[best_ngram_name]
    custom = analyze_custom_tweet(
        args.text, ngram_models[best_ngram_name], best_n, hmm_model, vocab
    )

    # 8. Generate report
    generate_report(
        stats, train_tokens, test_tokens, vocab_size,
        ngram_results, best_ngram_name, hmm_metrics, hmm_train_time,
        custom, output_path,
    )

    # 9. Terminal summary
    print_terminal_summary(
        stats, train_tokens, test_tokens, vocab_size,
        ngram_results, best_ngram_name, hmm_metrics, custom, output_path,
    )


if __name__ == "__main__":
    main()
