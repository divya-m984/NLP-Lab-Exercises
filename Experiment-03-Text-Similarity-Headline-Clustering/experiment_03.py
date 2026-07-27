"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 03
Title           : Explore Various Text Similarity Metrics, Including WordNet-Based
                  Similarity, for Clustering News Headlines into Topics
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
HYBRID_TFIDF_WEIGHT = 0.70
HYBRID_WORDNET_WEIGHT = 0.30
NUM_CLUSTERS = 4

STUDENT_NAME = "Divya M"
REGISTER_NUMBER = "24AD0074"
EXPERIMENT_TITLE = (
    "Explore Various Text Similarity Metrics, Including WordNet-Based\n"
    "Similarity, for Clustering News Headlines into Topics"
)
EXPERIMENT_AIM = (
    "To calculate text similarity using TF-IDF and WordNet similarity and\n"
    "group similar news headlines into topics."
)

DEFAULT_HEADLINE_A = (
    "Technology companies release faster processors for mobile devices"
)
DEFAULT_HEADLINE_B = (
    "New computer chips improve smartphone performance"
)

# 48 labelled news headlines (originally from data/news_headlines.csv)
HEADLINES_DATA: List[Tuple[str, str]] = [
    ("Stock markets rally after central bank cuts interest rates", "business"),
    ("Major retailer announces expansion into three new countries", "business"),
    ("Oil prices surge amid rising global demand for energy", "business"),
    ("Automobile manufacturer reports record quarterly revenue", "business"),
    ("Small businesses struggle with rising supply chain costs", "business"),
    ("Banking sector faces new regulations on lending practices", "business"),
    ("Technology startup secures large funding round from investors", "business"),
    ("Trade agreement between nations boosts export volumes", "business"),
    ("Housing market slows as mortgage rates continue to climb", "business"),
    ("Airlines report increased passenger traffic this quarter", "business"),
    ("Pharmaceutical company merges with rival in billion dollar deal", "business"),
    ("Consumer spending rises despite inflation concerns", "business"),
    ("National football team wins championship after dramatic final", "sports"),
    ("Tennis star claims fifth grand slam title in straight sets", "sports"),
    ("Olympic committee announces host city for upcoming games", "sports"),
    ("Basketball league introduces new playoff format next season", "sports"),
    ("Marathon runner breaks world record at international event", "sports"),
    ("Cricket captain retires after two decades of professional play", "sports"),
    ("Swimming federation bans controversial training technique", "sports"),
    ("Golf tournament draws record number of spectators this year", "sports"),
    ("Rugby squad announces roster changes ahead of world cup", "sports"),
    ("Boxing champion defends title in unanimous decision victory", "sports"),
    ("Cycling race sees unexpected winner after mountain stage upset", "sports"),
    ("Hockey team trades star player before transfer deadline", "sports"),
    ("Smartphone makers unveil foldable screens at electronics expo", "technology"),
    ("Cloud computing services expand to meet enterprise demand", "technology"),
    ("Researchers develop faster algorithm for natural language processing", "technology"),
    ("Electric vehicle battery technology achieves major breakthrough", "technology"),
    ("Cybersecurity firm warns of increasing ransomware threats worldwide", "technology"),
    ("Social media platform introduces new content moderation tools", "technology"),
    ("Quantum computing lab demonstrates high qubit processor prototype", "technology"),
    ("Satellite internet provider extends coverage to rural regions", "technology"),
    ("Wearable fitness devices gain popularity among health enthusiasts", "technology"),
    ("Robotics company debuts autonomous warehouse delivery system", "technology"),
    ("Open source software community releases major framework update", "technology"),
    ("Artificial intelligence tool helps doctors analyze medical scans", "technology"),
    ("New vaccine shows strong results in large clinical trial", "health"),
    ("Researchers discover high protein diet reduces heart disease risk", "health"),
    ("Mental health awareness campaign reaches millions of young adults", "health"),
    ("Hospitals adopt telemedicine platforms for remote patient care", "health"),
    ("Study links regular exercise to improved cognitive function", "health"),
    ("World health organization issues guidelines on air pollution exposure", "health"),
    ("Scientists identify high gene variant associated with diabetes", "health"),
    ("Childhood obesity rates decline after school nutrition programs", "health"),
    ("Sleep deprivation found to weaken immune system response", "health"),
    ("Doctors recommend updated screening guidelines for lung cancer", "health"),
    ("Herbal supplement recalled after laboratory detects high contamination", "health"),
    ("Public health officials urge expanded access to clean drinking water", "health"),
]


# ---------------------------------------------------------------------------
# NLTK resource management
# ---------------------------------------------------------------------------
def ensure_nltk_resources() -> None:
    """Download only missing NLTK resources."""
    import nltk

    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
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
# Text preprocessing
# ---------------------------------------------------------------------------
def get_wordnet_pos(treebank_tag: str) -> str:
    """Map treebank POS tag to WordNet POS for lemmatization."""
    from nltk.corpus import wordnet
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess(text: str) -> Tuple[List[str], str]:
    """Preprocess text: lowercase, tokenize, remove stopwords, POS-aware lemmatize.

    Returns:
        Tuple of (token list, joined string for TF-IDF).
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]

    # POS-aware lemmatization
    tagged = nltk.pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in tagged]

    return tokens, " ".join(tokens)


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------
def compute_tfidf_cosine_matrix(processed_strings: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Build TF-IDF matrix and compute pairwise cosine similarity."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_strings)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix, vectorizer


def compute_jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def compute_jaccard_matrix(token_lists: List[List[str]]) -> np.ndarray:
    """Compute full pairwise Jaccard similarity matrix."""
    n = len(token_lists)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = compute_jaccard_similarity(token_lists[i], token_lists[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
    return matrix


def compute_wordnet_similarity_pair(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Compute WordNet Wu-Palmer similarity between two token lists.

    For each word in one list, finds the best Wu-Palmer similarity with
    any word in the other list, then averages both directions for symmetry.
    """
    from nltk.corpus import wordnet

    def best_similarity(word: str, other_words: List[str]) -> float:
        synsets_a = wordnet.synsets(word)
        if not synsets_a:
            return 0.0
        best = 0.0
        for other in other_words:
            synsets_b = wordnet.synsets(other)
            for sa in synsets_a:
                for sb in synsets_b:
                    score = sa.wup_similarity(sb)
                    if score is not None and score > best:
                        best = score
        return best

    if not tokens_a or not tokens_b:
        return 0.0

    # Direction A -> B
    scores_ab = [best_similarity(w, tokens_b) for w in tokens_a]
    avg_ab = sum(scores_ab) / len(scores_ab) if scores_ab else 0.0

    # Direction B -> A
    scores_ba = [best_similarity(w, tokens_a) for w in tokens_b]
    avg_ba = sum(scores_ba) / len(scores_ba) if scores_ba else 0.0

    return (avg_ab + avg_ba) / 2.0


def compute_wordnet_matrix(token_lists: List[List[str]]) -> np.ndarray:
    """Compute full pairwise WordNet similarity matrix."""
    n = len(token_lists)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = compute_wordnet_similarity_pair(token_lists[i], token_lists[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
    return matrix


def compute_hybrid_matrix(tfidf_matrix: np.ndarray, wordnet_matrix: np.ndarray) -> np.ndarray:
    """Combine TF-IDF cosine and WordNet similarity.

    Hybrid = 70% TF-IDF cosine similarity + 30% WordNet semantic similarity.
    """
    return HYBRID_TFIDF_WEIGHT * tfidf_matrix + HYBRID_WORDNET_WEIGHT * wordnet_matrix


# ---------------------------------------------------------------------------
# Pairwise demonstration
# ---------------------------------------------------------------------------
def pairwise_demonstration(headline_a: str, headline_b: str) -> Dict[str, float]:
    """Compute all four similarity metrics for two headlines."""
    tokens_a, str_a = preprocess(headline_a)
    tokens_b, str_b = preprocess(headline_b)

    # TF-IDF cosine similarity (fit on the pair)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([str_a, str_b])
    tfidf_cos = cosine_similarity(tfidf_matrix)[0, 1]

    jaccard = compute_jaccard_similarity(tokens_a, tokens_b)
    wordnet_sim = compute_wordnet_similarity_pair(tokens_a, tokens_b)
    hybrid = HYBRID_TFIDF_WEIGHT * tfidf_cos + HYBRID_WORDNET_WEIGHT * wordnet_sim

    return {
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "tfidf_cosine": float(tfidf_cos),
        "jaccard": jaccard,
        "wordnet": wordnet_sim,
        "hybrid": hybrid,
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def run_clustering(
    processed_strings: List[str],
    tfidf_cosine_matrix: np.ndarray,
    hybrid_matrix: np.ndarray,
    n_clusters: int,
) -> Dict[str, np.ndarray]:
    """Run three clustering approaches and return label arrays."""
    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(processed_strings)

    # A. K-Means with TF-IDF vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    kmeans_labels = kmeans.fit_predict(tfidf_vectors)

    # B. Agglomerative with TF-IDF cosine distance
    cosine_dist = 1.0 - tfidf_cosine_matrix
    np.fill_diagonal(cosine_dist, 0.0)
    cosine_dist = np.clip(cosine_dist, 0.0, None)

    agg_cosine = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    agg_cosine_labels = agg_cosine.fit_predict(cosine_dist)

    # C. Agglomerative with hybrid distance
    hybrid_dist = 1.0 - hybrid_matrix
    np.fill_diagonal(hybrid_dist, 0.0)
    hybrid_dist = np.clip(hybrid_dist, 0.0, None)

    agg_hybrid = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    agg_hybrid_labels = agg_hybrid.fit_predict(hybrid_dist)

    return {
        "K-Means TF-IDF": kmeans_labels,
        "Agglomerative Cosine": agg_cosine_labels,
        "Agglomerative Hybrid": agg_hybrid_labels,
        "_tfidf_vectors": tfidf_vectors,
        "_cosine_dist": cosine_dist,
        "_hybrid_dist": hybrid_dist,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_clustering(
    true_labels: List[str],
    clustering_results: Dict[str, np.ndarray],
) -> List[Dict]:
    """Evaluate each clustering method using ARI, NMI, and silhouette score."""
    methods = [
        ("K-Means TF-IDF", "_tfidf_vectors", "cosine"),
        ("Agglomerative Cosine", "_cosine_dist", "precomputed"),
        ("Agglomerative Hybrid", "_hybrid_dist", "precomputed"),
    ]

    evaluations = []
    for method_name, data_key, sil_metric in methods:
        pred_labels = clustering_results[method_name]
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)

        try:
            if sil_metric == "cosine":
                sil = silhouette_score(clustering_results[data_key], pred_labels, metric="cosine")
            else:
                sil = silhouette_score(clustering_results[data_key], pred_labels, metric="precomputed")
        except ValueError:
            sil = float("nan")

        evaluations.append({
            "method": method_name,
            "adjusted_rand_index": ari,
            "normalized_mutual_information": nmi,
            "silhouette_score": sil,
        })

    return evaluations


def select_best_method(evaluations: List[Dict]) -> Dict:
    """Select best method by highest ARI, then NMI as tie-breaker."""
    return max(evaluations, key=lambda e: (e["adjusted_rand_index"], e["normalized_mutual_information"]))


# ---------------------------------------------------------------------------
# Cluster interpretation
# ---------------------------------------------------------------------------
def get_cluster_keywords(
    processed_strings: List[str],
    labels: np.ndarray,
    n_terms: int = 5,
) -> Dict[int, List[str]]:
    """Extract top TF-IDF terms per cluster for interpretation."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_strings)
    feature_names = vectorizer.get_feature_names_out()

    keywords = {}
    for cluster_id in sorted(set(labels)):
        indices = np.where(labels == cluster_id)[0]
        cluster_tfidf = tfidf_matrix[indices].toarray().mean(axis=0)
        top_indices = cluster_tfidf.argsort()[-n_terms:][::-1]
        keywords[cluster_id] = [feature_names[i] for i in top_indices]

    return keywords


def map_clusters_to_topics(
    labels: np.ndarray, true_topics: List[str],
) -> Dict[int, str]:
    """Map each cluster to its majority true topic (for interpretation only)."""
    from collections import Counter
    mapping = {}
    for cluster_id in sorted(set(labels)):
        indices = np.where(labels == cluster_id)[0]
        topic_counts = Counter(true_topics[i] for i in indices)
        mapping[cluster_id] = topic_counts.most_common(1)[0][0]
    return mapping


# ---------------------------------------------------------------------------
# Output text report
# ---------------------------------------------------------------------------
def generate_output_txt(
    headlines: List[str],
    topics: List[str],
    pairwise: Dict,
    headline_a: str,
    headline_b: str,
    evaluations: List[Dict],
    best_eval: Dict,
    best_keywords: Dict[int, List[str]],
    best_cluster_map: Dict[int, str],
    clustering_results: Dict[str, np.ndarray],
    path: Path,
) -> str:
    """Generate the comprehensive output text report."""
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
    unique_topics = sorted(set(topics))
    lines.append(f"Total headlines : {len(headlines)}")
    lines.append(f"Topics          : {', '.join(unique_topics)}")
    for t in unique_topics:
        lines.append(f"  {t:<15} : {topics.count(t)} headlines")

    section("SIMILARITY METHODS")
    lines.append("A. TF-IDF Cosine Similarity")
    lines.append("   Builds a TF-IDF matrix from preprocessed headlines and computes")
    lines.append("   pairwise cosine similarity between document vectors.")
    lines.append("")
    lines.append("B. Jaccard Similarity")
    lines.append("   Compares sets of preprocessed tokens using intersection over union.")
    lines.append("")
    lines.append("C. WordNet Semantic Similarity (Wu-Palmer)")
    lines.append("   For each word pair across two headlines, computes Wu-Palmer similarity")
    lines.append("   using WordNet synsets. Averages best matches in both directions for")
    lines.append("   a symmetric score between 0 and 1.")
    lines.append("")
    lines.append("D. Hybrid Similarity")
    lines.append(f"   Combines {HYBRID_TFIDF_WEIGHT:.0%} TF-IDF cosine + {HYBRID_WORDNET_WEIGHT:.0%} WordNet semantic similarity.")

    section("PAIRWISE DEMONSTRATION")
    lines.append(f"Headline A : {headline_a}")
    lines.append(f"Headline B : {headline_b}")
    lines.append(f"Tokens A   : {pairwise['tokens_a']}")
    lines.append(f"Tokens B   : {pairwise['tokens_b']}")
    lines.append("")
    lines.append(f"TF-IDF Cosine Similarity  : {pairwise['tfidf_cosine']:.4f}")
    lines.append(f"Jaccard Similarity        : {pairwise['jaccard']:.4f}")
    lines.append(f"WordNet Similarity        : {pairwise['wordnet']:.4f}")
    lines.append(f"Hybrid Similarity         : {pairwise['hybrid']:.4f}")

    section("CLUSTERING EVALUATION")
    header = f"{'Method':<28} {'ARI':>8} {'NMI':>8} {'Silhouette':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    for e in evaluations:
        sil_str = f"{e['silhouette_score']:.4f}" if not np.isnan(e["silhouette_score"]) else "N/A"
        lines.append(
            f"{e['method']:<28} {e['adjusted_rand_index']:>8.4f} "
            f"{e['normalized_mutual_information']:>8.4f} {sil_str:>11}"
        )

    section("BEST CLUSTERING METHOD")
    lines.append(f"Method                       : {best_eval['method']}")
    lines.append(f"Adjusted Rand Index          : {best_eval['adjusted_rand_index']:.4f}")
    lines.append(f"Normalized Mutual Information : {best_eval['normalized_mutual_information']:.4f}")
    sil_str = f"{best_eval['silhouette_score']:.4f}" if not np.isnan(best_eval["silhouette_score"]) else "N/A"
    lines.append(f"Silhouette Score             : {sil_str}")

    section("REPRESENTATIVE KEYWORDS (Best Method)")
    for cluster_id in sorted(best_keywords.keys()):
        majority_topic = best_cluster_map.get(cluster_id, "unknown")
        kw = ", ".join(best_keywords[cluster_id])
        lines.append(f"Cluster {cluster_id} (majority topic: {majority_topic}): {kw}")

    section("CLUSTER ASSIGNMENTS")
    method_names = ["K-Means TF-IDF", "Agglomerative Cosine", "Agglomerative Hybrid"]
    for method_name in method_names:
        lines.append("")
        lines.append(f"--- {method_name} ---")
        lines.append(f"{'#':<5} {'Headline':<60} {'True Topic':<12} {'Cluster':>7}")
        lines.append("-" * 86)
        pred = clustering_results[method_name]
        for i, (h, t) in enumerate(zip(headlines, topics)):
            label = f"H{i+1:02d}"
            lines.append(f"{label:<5} {h:<60} {t:<12} {pred[i]:>7}")

    section("RESULT")
    lines.append(
        f"The {best_eval['method']} method achieved the highest Adjusted Rand Index "
        f"of {best_eval['adjusted_rand_index']:.4f} among the three clustering approaches "
        f"evaluated for grouping news headlines into topics using text similarity metrics."
    )

    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------
def print_summary(
    pairwise: Dict,
    headline_a: str,
    headline_b: str,
    evaluations: List[Dict],
    best_eval: Dict,
    best_keywords: Dict[int, List[str]],
    best_cluster_map: Dict[int, str],
) -> None:
    """Print concise summary to the terminal."""
    print("\n" + "=" * 70)
    print("PAIRWISE DEMONSTRATION")
    print("=" * 70)
    print(f"Headline A : {headline_a}")
    print(f"Headline B : {headline_b}")
    print(f"Tokens A   : {pairwise['tokens_a']}")
    print(f"Tokens B   : {pairwise['tokens_b']}")
    print(f"\nTF-IDF Cosine Similarity  : {pairwise['tfidf_cosine']:.4f}")
    print(f"Jaccard Similarity        : {pairwise['jaccard']:.4f}")
    print(f"WordNet Similarity        : {pairwise['wordnet']:.4f}")
    print(f"Hybrid Similarity         : {pairwise['hybrid']:.4f}")

    print("\n" + "=" * 70)
    print("CLUSTERING EVALUATION")
    print("=" * 70)
    header = f"{'Method':<28} {'ARI':>8} {'NMI':>8} {'Silhouette':>11}"
    print(header)
    print("-" * len(header))
    for e in evaluations:
        sil_str = f"{e['silhouette_score']:.4f}" if not np.isnan(e["silhouette_score"]) else "N/A"
        print(
            f"{e['method']:<28} {e['adjusted_rand_index']:>8.4f} "
            f"{e['normalized_mutual_information']:>8.4f} {sil_str:>11}"
        )

    print(f"\nBest method: {best_eval['method']} (ARI = {best_eval['adjusted_rand_index']:.4f})")

    print("\n" + "=" * 70)
    print("REPRESENTATIVE KEYWORDS (Best Method)")
    print("=" * 70)
    for cluster_id in sorted(best_keywords.keys()):
        majority_topic = best_cluster_map.get(cluster_id, "unknown")
        kw = ", ".join(best_keywords[cluster_id])
        print(f"Cluster {cluster_id} (majority topic: {majority_topic}): {kw}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for Experiment 03."""
    parser = argparse.ArgumentParser(
        description="Text similarity and headline clustering experiment."
    )
    parser.add_argument(
        "--headline-a", type=str, default=None,
        help="First headline for pairwise demonstration.",
    )
    parser.add_argument(
        "--headline-b", type=str, default=None,
        help="Second headline for pairwise demonstration.",
    )
    args = parser.parse_args()

    headline_a = args.headline_a if args.headline_a else DEFAULT_HEADLINE_A
    headline_b = args.headline_b if args.headline_b else DEFAULT_HEADLINE_B

    # Resolve paths relative to this script
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure NLTK data
    print("Checking NLTK resources...")
    ensure_nltk_resources()

    # Load dataset from embedded constant
    headlines = [h for h, _ in HEADLINES_DATA]
    topics = [t for _, t in HEADLINES_DATA]
    unique_topics = sorted(set(topics))
    n_clusters = len(unique_topics)
    print(f"Loaded {len(headlines)} headlines across {n_clusters} topics: {', '.join(unique_topics)}")

    # Preprocess all headlines
    print("Preprocessing headlines...")
    token_lists: List[List[str]] = []
    processed_strings: List[str] = []
    for h in headlines:
        tokens, proc_str = preprocess(h)
        token_lists.append(tokens)
        processed_strings.append(proc_str)

    # Pairwise demonstration
    print("Computing pairwise similarity demonstration...")
    pairwise = pairwise_demonstration(headline_a, headline_b)

    # Compute similarity matrices
    print("Computing TF-IDF cosine similarity matrix...")
    tfidf_cosine_matrix, _ = compute_tfidf_cosine_matrix(processed_strings)

    print("Computing Jaccard similarity matrix...")
    jaccard_matrix = compute_jaccard_matrix(token_lists)

    print("Computing WordNet similarity matrix (this may take a moment)...")
    wordnet_matrix = compute_wordnet_matrix(token_lists)

    print("Computing hybrid similarity matrix...")
    hybrid_matrix = compute_hybrid_matrix(tfidf_cosine_matrix, wordnet_matrix)

    # Clustering
    print("Running clustering algorithms...")
    clustering_results = run_clustering(processed_strings, tfidf_cosine_matrix, hybrid_matrix, n_clusters)

    # Evaluation
    print("Evaluating clustering results...")
    evaluations = evaluate_clustering(topics, clustering_results)
    best_eval = select_best_method(evaluations)
    best_method_name = best_eval["method"]

    # Cluster interpretation
    best_labels = clustering_results[best_method_name]
    best_keywords = get_cluster_keywords(processed_strings, best_labels, n_terms=5)
    best_cluster_map = map_clusters_to_topics(best_labels, topics)

    # Save output
    print("Saving output file...")
    generate_output_txt(
        headlines, topics, pairwise, headline_a, headline_b,
        evaluations, best_eval, best_keywords, best_cluster_map,
        clustering_results, output_dir / "output.txt",
    )

    # Terminal output
    print_summary(
        pairwise, headline_a, headline_b,
        evaluations, best_eval, best_keywords, best_cluster_map,
    )

    print(f"Output saved to: {output_dir / 'output.txt'}")


if __name__ == "__main__":
    main()
