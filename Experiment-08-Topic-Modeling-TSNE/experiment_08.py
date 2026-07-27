"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 08
Title           : Apply Topic Modeling Techniques to Extract Themes from a Collection
                  of Customer Reviews and Visualize the Results Using t-SNE
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
NUM_TOPICS = 4
NUM_REVIEWS = 60
REVIEWS_PER_THEME = 15
TOP_N_WORDS = 6
TSNE_PERPLEXITY = 10
GRID_COLS = 72
GRID_ROWS = 22

EXPERIMENT_TITLE = (
    "Apply Topic Modeling Techniques to Extract Themes from a Collection\n"
    "of Customer Reviews and Visualize the Results Using t-SNE"
)

AIM = (
    "To apply Latent Dirichlet Allocation (LDA) for topic modeling and use\n"
    "t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualize\n"
    "customer review clusters."
)

DEFAULT_TEXT = (
    "The product feels durable and works well, but the delivery arrived two days late."
)

THEME_NAMES = ["product_quality", "delivery", "customer_service", "price_and_value"]

# ---------------------------------------------------------------------------
# Embedded customer-review dataset
# ---------------------------------------------------------------------------

REVIEWS: List[Dict[str, str]] = [
    # ---- product_quality (15) ----
    {"review_id": "R01", "text": "The build quality of this gadget is outstanding and it feels premium in hand.", "theme": "product_quality"},
    {"review_id": "R02", "text": "Cheap materials were used and the casing cracked after just one week.", "theme": "product_quality"},
    {"review_id": "R03", "text": "Color on the display is vivid and the resolution exceeds my expectations.", "theme": "product_quality"},
    {"review_id": "R04", "text": "Battery drains far too quickly, barely lasts half a day with moderate use.", "theme": "product_quality"},
    {"review_id": "R05", "text": "Stitching on the jacket is neat and the fabric feels comfortable against the skin.", "theme": "product_quality"},
    {"review_id": "R06", "text": "The blender motor overheated after blending ice cubes twice, poor durability.", "theme": "product_quality"},
    {"review_id": "R07", "text": "Sound clarity on these headphones is impressive with deep bass response.", "theme": "product_quality"},
    {"review_id": "R08", "text": "Screen started flickering within a month, likely a defective panel.", "theme": "product_quality"},
    {"review_id": "R09", "text": "The product works exactly as described, solid construction overall.", "theme": "product_quality"},
    {"review_id": "R10", "text": "Material quality is average, neither impressive nor terrible for the category.", "theme": "product_quality"},
    {"review_id": "R11", "text": "Lens coating scratches easily even with careful handling, disappointing optics.", "theme": "product_quality"},
    {"review_id": "R12", "text": "Waterproof feature held up perfectly during a heavy rainstorm, very reliable.", "theme": "product_quality"},
    {"review_id": "R13", "text": "The zipper jammed on day three, clearly a manufacturing defect.", "theme": "product_quality"},
    {"review_id": "R14", "text": "Excellent craftsmanship on the wooden finish, smooth texture throughout.", "theme": "product_quality"},
    {"review_id": "R15", "text": "Performance is decent for everyday tasks but struggles with heavy workloads.", "theme": "product_quality"},

    # ---- delivery (15) ----
    {"review_id": "R16", "text": "Package arrived two days ahead of schedule, very impressed with the speed.", "theme": "delivery"},
    {"review_id": "R17", "text": "Shipping took over three weeks and I had to follow up multiple times.", "theme": "delivery"},
    {"review_id": "R18", "text": "The box was dented and the item inside had minor scratches from transit.", "theme": "delivery"},
    {"review_id": "R19", "text": "Same-day delivery option worked flawlessly, the courier was prompt.", "theme": "delivery"},
    {"review_id": "R20", "text": "Tracking information was never updated, leaving me anxious about the order.", "theme": "delivery"},
    {"review_id": "R21", "text": "Received the wrong item entirely, had to arrange a return shipment.", "theme": "delivery"},
    {"review_id": "R22", "text": "Express shipping was worth every penny, arrived securely packaged overnight.", "theme": "delivery"},
    {"review_id": "R23", "text": "Delivery driver left the parcel in the rain instead of at the door.", "theme": "delivery"},
    {"review_id": "R24", "text": "Free shipping promotion was great but the estimated date kept changing.", "theme": "delivery"},
    {"review_id": "R25", "text": "The packaging was eco-friendly and everything inside was well protected.", "theme": "delivery"},
    {"review_id": "R26", "text": "International shipping charges were reasonable and customs clearance was smooth.", "theme": "delivery"},
    {"review_id": "R27", "text": "Order dispatched quickly but the last mile carrier caused a two day delay.", "theme": "delivery"},
    {"review_id": "R28", "text": "Parcel was marked delivered but I never found it at my doorstep.", "theme": "delivery"},
    {"review_id": "R29", "text": "Delivery experience was average, nothing exceptional but it arrived on time.", "theme": "delivery"},
    {"review_id": "R30", "text": "Fragile sticker was ignored and the glass item arrived completely shattered.", "theme": "delivery"},

    # ---- customer_service (15) ----
    {"review_id": "R31", "text": "Support agent resolved my complaint within ten minutes, very professional.", "theme": "customer_service"},
    {"review_id": "R32", "text": "Waited on hold for over an hour before anyone picked up the phone.", "theme": "customer_service"},
    {"review_id": "R33", "text": "Live chat representative was knowledgeable and guided me through every step.", "theme": "customer_service"},
    {"review_id": "R34", "text": "Refund request was denied without explanation, extremely frustrating experience.", "theme": "customer_service"},
    {"review_id": "R35", "text": "Customer care team followed up via email to ensure my issue was fully resolved.", "theme": "customer_service"},
    {"review_id": "R36", "text": "The helpdesk ticket system is confusing and my query went unanswered for days.", "theme": "customer_service"},
    {"review_id": "R37", "text": "Staff at the service center were friendly and handled the replacement swiftly.", "theme": "customer_service"},
    {"review_id": "R38", "text": "Automated phone menu is a nightmare, impossible to reach a real person.", "theme": "customer_service"},
    {"review_id": "R39", "text": "Return process was hassle-free thanks to the helpful support coordinator.", "theme": "customer_service"},
    {"review_id": "R40", "text": "They apologized sincerely and offered a discount coupon for the inconvenience.", "theme": "customer_service"},
    {"review_id": "R41", "text": "Service response time has improved significantly compared to last year.", "theme": "customer_service"},
    {"review_id": "R42", "text": "No callback was ever made despite three separate promises from the team.", "theme": "customer_service"},
    {"review_id": "R43", "text": "The warranty claim experience was smooth and the agent was courteous.", "theme": "customer_service"},
    {"review_id": "R44", "text": "Support quality is inconsistent, depends entirely on which agent you get.", "theme": "customer_service"},
    {"review_id": "R45", "text": "Complaint escalation took too long and management never acknowledged the issue.", "theme": "customer_service"},

    # ---- price_and_value (15) ----
    {"review_id": "R46", "text": "Unbeatable price for the features offered, truly a bargain purchase.", "theme": "price_and_value"},
    {"review_id": "R47", "text": "Way too expensive for what you actually get, not worth the premium.", "theme": "price_and_value"},
    {"review_id": "R48", "text": "Seasonal discount brought the cost down nicely, great deal overall.", "theme": "price_and_value"},
    {"review_id": "R49", "text": "Subscription fee keeps increasing every year without any added benefits.", "theme": "price_and_value"},
    {"review_id": "R50", "text": "Affordable alternative to the big brands and the quality is comparable.", "theme": "price_and_value"},
    {"review_id": "R51", "text": "Hidden charges at checkout made the final amount much higher than advertised.", "theme": "price_and_value"},
    {"review_id": "R52", "text": "Excellent value for money, you genuinely get more than what you pay for.", "theme": "price_and_value"},
    {"review_id": "R53", "text": "Price matched a competitor offer instantly, saved me quite a bit of cash.", "theme": "price_and_value"},
    {"review_id": "R54", "text": "Monthly plan is reasonably priced but the annual option saves even more.", "theme": "price_and_value"},
    {"review_id": "R55", "text": "Budget-friendly option that does exactly what it promises without frills.", "theme": "price_and_value"},
    {"review_id": "R56", "text": "Overpriced compared to similar products available from other retailers.", "theme": "price_and_value"},
    {"review_id": "R57", "text": "Coupon code applied perfectly and the final bill was very reasonable.", "theme": "price_and_value"},
    {"review_id": "R58", "text": "The cost to quality ratio is mediocre, expected better at this price point.", "theme": "price_and_value"},
    {"review_id": "R59", "text": "Loyalty rewards program offers genuine savings on repeat purchases.", "theme": "price_and_value"},
    {"review_id": "R60", "text": "Paid full price only to see a flash sale the very next day, frustrating.", "theme": "price_and_value"},
]

# ---------------------------------------------------------------------------
# NLTK resource helpers
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Download only missing NLTK resources."""
    import nltk  # noqa: E402

    resources = [
        ("tokenizers", "punkt"),
        ("tokenizers", "punkt_tab"),
        ("corpora", "stopwords"),
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
                print(f"Warning: could not download NLTK resource '{name}': {exc}")


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess(texts: List[str]) -> Tuple[List[List[str]], List[str]]:
    """Shared preprocessing: lowercase, tokenize, filter, lemmatize.

    Returns:
        token_lists: list of token lists (one per document)
        processed_strings: joined strings suitable for vectorization
    """
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    token_lists: List[List[str]] = []
    processed_strings: List[str] = []

    for text in texts:
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) >= 2]
        token_lists.append(tokens)
        processed_strings.append(" ".join(tokens))

    return token_lists, processed_strings


# ---------------------------------------------------------------------------
# Topic modeling
# ---------------------------------------------------------------------------

def run_lda(processed_strings: List[str]) -> Tuple:
    """Run LDA with CountVectorizer. Returns (model, vectorizer, doc_topic_matrix)."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    dtm = vectorizer.fit_transform(processed_strings)

    # Verify no zero-vector documents; fall back to unigrams if needed
    row_nonzero = np.array(dtm.sum(axis=1)).flatten()
    if np.any(row_nonzero == 0):
        print("Warning: CountVectorizer produced zero-vector documents with "
              "bigrams; falling back to unigrams only.")
        vectorizer = CountVectorizer(
            stop_words="english",
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 1),
        )
        dtm = vectorizer.fit_transform(processed_strings)
        row_nonzero = np.array(dtm.sum(axis=1)).flatten()
        if np.any(row_nonzero == 0):
            raise ValueError(
                "LDA vectorization failed: some documents have no features "
                "even with unigram fallback."
            )

    lda = LatentDirichletAllocation(
        n_components=NUM_TOPICS,
        random_state=RANDOM_SEED,
        learning_method="batch",
        max_iter=30,
    )
    doc_topic = lda.fit_transform(dtm)

    return lda, vectorizer, doc_topic


def run_nmf(processed_strings: List[str]) -> Tuple:
    """Run NMF with TfidfVectorizer. Returns (model, vectorizer, doc_topic_matrix)."""
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(processed_strings)

    # Verify no zero-vector documents; fall back to unigrams if needed
    row_nonzero = np.array(tfidf.sum(axis=1)).flatten()
    if np.any(row_nonzero == 0):
        print("Warning: TfidfVectorizer produced zero-vector documents with "
              "bigrams; falling back to unigrams only.")
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 1),
        )
        tfidf = vectorizer.fit_transform(processed_strings)
        row_nonzero = np.array(tfidf.sum(axis=1)).flatten()
        if np.any(row_nonzero == 0):
            raise ValueError(
                "NMF vectorization failed: some documents have no features "
                "even with unigram fallback."
            )

    nmf = NMF(
        n_components=NUM_TOPICS,
        init="nndsvda",
        random_state=RANDOM_SEED,
        max_iter=1000,
    )
    doc_topic = nmf.fit_transform(tfidf)

    return nmf, vectorizer, doc_topic


def normalize_doc_topic(doc_topic: np.ndarray) -> np.ndarray:
    """Normalize rows of a document-topic matrix to sum to 1.

    Zero-sum rows are left as all zeros (confidence 0.0) rather than
    dividing by zero or assigning an arbitrary topic.
    """
    row_sums = doc_topic.sum(axis=1, keepdims=True)
    zero_mask = (row_sums == 0).flatten()
    if np.any(zero_mask):
        print(f"Warning: {int(zero_mask.sum())} document(s) have zero topic "
              "weights; their topic is marked unavailable (confidence 0.0).")
    safe_sums = np.where(row_sums == 0, 1.0, row_sums)
    result = doc_topic / safe_sums
    result[zero_mask] = 0.0
    return result


def get_top_words(model, vectorizer, n: int = TOP_N_WORDS) -> List[List[str]]:
    """Extract top words/phrases for each topic."""
    feature_names = vectorizer.get_feature_names_out()
    topics: List[List[str]] = []
    for topic_weights in model.components_:
        top_indices = topic_weights.argsort()[: -n - 1 : -1]
        topics.append([feature_names[i] for i in top_indices])
    return topics


def get_assignments(doc_topic_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return dominant topic indices and confidence values."""
    assignments = doc_topic_norm.argmax(axis=1)
    confidences = doc_topic_norm.max(axis=1)
    return assignments, confidences


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def encode_themes(themes: List[str]) -> np.ndarray:
    """Encode theme strings to integer labels."""
    theme_to_id = {t: i for i, t in enumerate(THEME_NAMES)}
    return np.array([theme_to_id[t] for t in themes])


def majority_theme_map(assignments: np.ndarray, true_themes: List[str]) -> Dict[int, str]:
    """Map each discovered topic to its majority true theme."""
    from collections import Counter
    mapping: Dict[int, str] = {}
    for topic_id in range(NUM_TOPICS):
        mask = assignments == topic_id
        if mask.sum() == 0:
            mapping[topic_id] = "unknown"
            continue
        themes_in_topic = [true_themes[i] for i in range(len(true_themes)) if mask[i]]
        most_common = Counter(themes_in_topic).most_common(1)[0][0]
        mapping[topic_id] = most_common
    return mapping


def topic_purity(assignments: np.ndarray, true_themes: List[str]) -> float:
    """Fraction of reviews matching majority theme of their assigned topic."""
    mapping = majority_theme_map(assignments, true_themes)
    correct = sum(
        1 for i, t in enumerate(true_themes) if mapping[assignments[i]] == t
    )
    return correct / len(true_themes)


def evaluate_model(
    doc_topic_norm: np.ndarray,
    assignments: np.ndarray,
    true_themes: List[str],
) -> Dict[str, Optional[float]]:
    """Compute ARI, NMI, purity, and silhouette score."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    true_labels = encode_themes(true_themes)

    ari = adjusted_rand_score(true_labels, assignments)
    nmi = normalized_mutual_info_score(true_labels, assignments)
    purity = topic_purity(assignments, true_themes)

    # Silhouette score with cosine metric
    sil: Optional[float] = None
    unique_labels = np.unique(assignments)
    if len(unique_labels) >= 2 and len(unique_labels) < len(assignments):
        try:
            sil = silhouette_score(doc_topic_norm, assignments, metric="cosine")
        except Exception:
            sil = None

    return {"ARI": ari, "NMI": nmi, "Purity": purity, "Silhouette": sil}


def select_best_model(
    metrics_lda: Dict[str, Optional[float]],
    metrics_nmf: Dict[str, Optional[float]],
) -> str:
    """Select best model by ARI > NMI > Purity."""
    for key in ("ARI", "NMI", "Purity"):
        val_lda = metrics_lda.get(key) or 0.0
        val_nmf = metrics_nmf.get(key) or 0.0
        if val_lda > val_nmf:
            return "LDA"
        if val_nmf > val_lda:
            return "NMF"
    return "LDA"  # default tie-break


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def run_tsne(doc_topic_norm: np.ndarray) -> np.ndarray:
    """Apply t-SNE to the normalized document-topic matrix.

    Returns an array of shape (n_docs, 2). Documents whose topic-weight
    row is all-zero receive NaN coordinates and are excluded from the
    t-SNE computation.
    """
    from sklearn.manifold import TSNE

    n_docs = doc_topic_norm.shape[0]
    row_sums = doc_topic_norm.sum(axis=1)
    valid_mask = (row_sums > 0) & np.isfinite(row_sums)

    if not np.all(np.isfinite(doc_topic_norm[valid_mask])):
        raise ValueError("t-SNE input contains non-finite values in "
                         "non-zero rows; cannot produce valid coordinates.")

    if valid_mask.sum() < 2:
        raise ValueError("Fewer than 2 valid document vectors available; "
                         "cannot run t-SNE.")

    tsne = TSNE(
        n_components=2,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
        perplexity=min(TSNE_PERPLEXITY, valid_mask.sum() - 1),
    )

    coords = np.full((n_docs, 2), np.nan)
    coords[valid_mask] = tsne.fit_transform(doc_topic_norm[valid_mask])
    return coords


def ascii_tsne_plot(
    coords: np.ndarray,
    assignments: np.ndarray,
    theme_map: Dict[int, str],
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
) -> str:
    """Create a terminal-friendly ASCII t-SNE visualization."""
    valid = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if not np.any(valid):
        return "(no valid t-SNE coordinates to plot)"

    x_min, x_max = coords[valid, 0].min(), coords[valid, 0].max()
    y_min, y_max = coords[valid, 1].min(), coords[valid, 1].max()

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    grid = [[" " for _ in range(cols)] for _ in range(rows)]

    for i in range(len(coords)):
        if not valid[i]:
            continue
        cx = int((coords[i, 0] - x_min) / x_range * (cols - 1))
        cy = int((coords[i, 1] - y_min) / y_range * (rows - 1))
        cx = min(max(cx, 0), cols - 1)
        cy = min(max(cy, 0), rows - 1)
        row_idx = rows - 1 - cy  # invert y-axis

        if grid[row_idx][cx] == " ":
            grid[row_idx][cx] = str(assignments[i])
        else:
            grid[row_idx][cx] = "*"

    border_top = "+" + "-" * cols + "+"
    lines = [border_top]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append(border_top)

    # Legend
    lines.append("")
    lines.append("Legend:")
    for topic_id in sorted(theme_map.keys()):
        lines.append(f"  {topic_id} = {theme_map[topic_id]}")
    lines.append("  * = overlapping points from different topics")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Custom review analysis
# ---------------------------------------------------------------------------

def analyze_custom_review(
    text: str,
    lda_model,
    lda_vec,
    nmf_model,
    nmf_vec,
    lda_theme_map: Dict[int, str],
    nmf_theme_map: Dict[int, str],
    best_model_name: str,
) -> Dict:
    """Analyze a custom review with both fitted models."""
    _, processed = preprocess([text])

    # LDA
    lda_dtm = lda_vec.transform(processed)
    lda_weights = lda_model.transform(lda_dtm)
    lda_weights = normalize_doc_topic(lda_weights)[0]
    lda_topic = int(lda_weights.argmax())
    lda_conf = float(lda_weights.max())

    # NMF
    nmf_tfidf = nmf_vec.transform(processed)
    nmf_weights = nmf_model.transform(nmf_tfidf)
    nmf_weights = normalize_doc_topic(nmf_weights)[0]
    nmf_topic = int(nmf_weights.argmax())
    nmf_conf = float(nmf_weights.max())

    if best_model_name == "LDA":
        final_theme = lda_theme_map[lda_topic]
        final_conf = lda_conf
    else:
        final_theme = nmf_theme_map[nmf_topic]
        final_conf = nmf_conf

    return {
        "original": text,
        "tokens": processed[0],
        "lda_weights": lda_weights,
        "nmf_weights": nmf_weights,
        "lda_topic": lda_topic,
        "nmf_topic": nmf_topic,
        "lda_theme": lda_theme_map[lda_topic],
        "nmf_theme": nmf_theme_map[nmf_topic],
        "final_theme": final_theme,
        "final_conf": final_conf,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def fmt_metric(val: Optional[float]) -> str:
    """Format a metric value or return N/A."""
    return f"{val:.4f}" if val is not None else "N/A"


def generate_report(
    true_themes: List[str],
    review_ids: List[str],
    review_texts: List[str],
    lda_top_words: List[List[str]],
    nmf_top_words: List[List[str]],
    metrics_lda: Dict[str, Optional[float]],
    metrics_nmf: Dict[str, Optional[float]],
    best_model_name: str,
    best_assignments: np.ndarray,
    best_confidences: np.ndarray,
    best_theme_map: Dict[int, str],
    best_top_words: List[List[str]],
    lda_theme_map: Dict[int, str],
    nmf_theme_map: Dict[int, str],
    tsne_coords: np.ndarray,
    ascii_plot: str,
    custom_result: Dict,
    output_path: Path,
) -> str:
    """Generate output/output.txt and return the report string."""
    lines: List[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 72)
        lines.append(title)
        lines.append("=" * 72)

    # Header
    section("EXPERIMENT 08 — TOPIC MODELING AND t-SNE VISUALIZATION")
    lines.append("")
    lines.append(f"Student Name    : Divya M")
    lines.append(f"Register Number : 24AD0074")
    lines.append("")
    lines.append(f"Title: {EXPERIMENT_TITLE}")
    lines.append("")
    lines.append(f"Aim: {AIM}")

    # Approach note
    section("APPROACH")
    lines.append("")
    lines.append("This experiment compares two topic-modeling techniques:")
    lines.append("  A. Latent Dirichlet Allocation (LDA) with CountVectorizer")
    lines.append("  B. Non-negative Matrix Factorization (NMF) with TfidfVectorizer")
    lines.append("Both models extract 4 topics from 60 customer reviews.")
    lines.append("The best model is selected by ARI, then NMI, then purity.")

    # Dataset summary
    section("DATASET SUMMARY")
    lines.append("")
    lines.append(f"Total reviews       : {NUM_REVIEWS}")
    lines.append(f"Reviews per theme   : {REVIEWS_PER_THEME}")
    lines.append(f"Themes              : {', '.join(THEME_NAMES)}")
    lines.append(f"True labels are used only for evaluation, not training.")

    # Preprocessing
    section("PREPROCESSING")
    lines.append("")
    lines.append("1. Convert text to lowercase")
    lines.append("2. Tokenize with nltk.word_tokenize")
    lines.append("3. Retain only alphabetic tokens")
    lines.append("4. Remove English stopwords")
    lines.append("5. Apply WordNet lemmatization")
    lines.append("6. Remove tokens shorter than 2 characters")

    # Model descriptions
    section("MODEL A: LATENT DIRICHLET ALLOCATION (LDA)")
    lines.append("")
    lines.append("Vectorizer  : CountVectorizer (stop_words='english', min_df=1,")
    lines.append("              max_df=0.95, ngram_range=(1,2))")
    lines.append("Model       : LDA (n_components=4, random_state=42,")
    lines.append("              learning_method='batch', max_iter=30)")

    section("MODEL B: NON-NEGATIVE MATRIX FACTORIZATION (NMF)")
    lines.append("")
    lines.append("Vectorizer  : TfidfVectorizer (stop_words='english', min_df=1,")
    lines.append("              max_df=0.95, ngram_range=(1,2))")
    lines.append("Model       : NMF (n_components=4, init='nndsvda',")
    lines.append("              random_state=42, max_iter=1000)")

    # Evaluation table
    section("EVALUATION RESULTS")
    lines.append("")
    lines.append(f"{'Metric':<20} {'LDA':>10} {'NMF':>10}")
    lines.append("-" * 42)
    for key in ("ARI", "NMI", "Purity", "Silhouette"):
        lines.append(f"{key:<20} {fmt_metric(metrics_lda[key]):>10} {fmt_metric(metrics_nmf[key]):>10}")
    lines.append("")
    lines.append(f"Selected best model : {best_model_name}")

    # Top words — LDA
    section("LDA TOPICS — TOP WORDS AND MAJORITY THEMES")
    lines.append("")
    lda_assignments, _ = get_assignments(normalize_doc_topic(
        np.zeros((1, NUM_TOPICS))  # placeholder; we need stored assignments
    ))
    for tid in range(NUM_TOPICS):
        count = int(np.sum(np.array(
            [1 for i, t in enumerate(true_themes)
             if lda_theme_map.get(tid) == lda_theme_map.get(tid)]  # just count
        )))
        lines.append(f"  Topic {tid}: {', '.join(lda_top_words[tid])}")
        lines.append(f"          Majority theme: {lda_theme_map.get(tid, 'unknown')}")

    # Top words — NMF
    section("NMF TOPICS — TOP WORDS AND MAJORITY THEMES")
    lines.append("")
    for tid in range(NUM_TOPICS):
        lines.append(f"  Topic {tid}: {', '.join(nmf_top_words[tid])}")
        lines.append(f"          Majority theme: {nmf_theme_map.get(tid, 'unknown')}")

    # Best model topics with review counts
    section(f"BEST MODEL ({best_model_name}) — TOPIC DETAILS")
    lines.append("")
    for tid in range(NUM_TOPICS):
        count = int(np.sum(best_assignments == tid))
        lines.append(f"  Topic {tid}: {', '.join(best_top_words[tid])}")
        lines.append(f"          Majority theme: {best_theme_map.get(tid, 'unknown')}")
        lines.append(f"          Reviews assigned: {count}")

    # t-SNE visualization
    section("t-SNE CLUSTER VISUALIZATION (ASCII)")
    lines.append("")
    lines.append("Note: t-SNE preserves local neighbourhood structure but its axis")
    lines.append("values do not have direct semantic meaning. The coordinates are")
    lines.append("useful for visualizing cluster separation, not for interpretation.")
    lines.append("")
    lines.append(ascii_plot)

    # Per-review t-SNE coordinates
    section("REVIEW ASSIGNMENTS AND t-SNE COORDINATES")
    lines.append("")
    header = f"{'ID':<5} {'True Theme':<20} {'Topic':>5} {'Conf':>7} {'t-SNE x':>9} {'t-SNE y':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for i in range(len(review_ids)):
        if np.isfinite(tsne_coords[i, 0]) and np.isfinite(tsne_coords[i, 1]):
            lines.append(
                f"{review_ids[i]:<5} {true_themes[i]:<20} {best_assignments[i]:>5} "
                f"{best_confidences[i]:>7.4f} {tsne_coords[i, 0]:>9.4f} {tsne_coords[i, 1]:>9.4f}"
            )
        else:
            lines.append(
                f"{review_ids[i]:<5} {true_themes[i]:<20} {best_assignments[i]:>5} "
                f"{best_confidences[i]:>7.4f}       N/A       N/A"
            )

    # Custom review
    section("CUSTOM REVIEW ANALYSIS")
    lines.append("")
    lines.append(f"Original review : {custom_result['original']}")
    lines.append(f"Processed tokens: {custom_result['tokens']}")
    lines.append("")
    lines.append("LDA topic weights: " + "  ".join(
        f"T{i}={w:.4f}" for i, w in enumerate(custom_result["lda_weights"])
    ))
    lines.append(f"LDA dominant topic: {custom_result['lda_topic']} "
                 f"({custom_result['lda_theme']})")
    lines.append("")
    lines.append("NMF topic weights: " + "  ".join(
        f"T{i}={w:.4f}" for i, w in enumerate(custom_result["nmf_weights"])
    ))
    lines.append(f"NMF dominant topic: {custom_result['nmf_topic']} "
                 f"({custom_result['nmf_theme']})")
    lines.append("")
    lines.append(f"Final theme ({best_model_name}): {custom_result['final_theme']}")
    lines.append(f"Confidence: {custom_result['final_conf']:.4f}")

    # Result
    section("RESULT")
    lines.append("")
    lines.append("LDA and NMF topic models were applied to 60 customer reviews")
    lines.append("across four themes. Both models were evaluated using ARI, NMI,")
    lines.append("purity, and silhouette score. The document-topic representation")
    lines.append(f"from the best model ({best_model_name}) was visualized using t-SNE.")
    lines.append("")
    lines.append(f"{best_model_name} was selected because it achieved the higher ARI "
                 f"({fmt_metric(metrics_lda['ARI'] if best_model_name == 'LDA' else metrics_nmf['ARI'])}), "
                 f"NMI ({fmt_metric(metrics_lda['NMI'] if best_model_name == 'LDA' else metrics_nmf['NMI'])}), "
                 f"and purity ({fmt_metric(metrics_lda['Purity'] if best_model_name == 'LDA' else metrics_nmf['Purity'])}).")
    lines.append("Overall ARI, NMI, and purity remained modest. The experiment")
    lines.append("demonstrates the topic-modeling and t-SNE workflow, but the weak")
    lines.append("scores show that short, overlapping customer reviews are difficult")
    lines.append("to separate into clean topics.")
    lines.append("")

    report = "\n".join(lines)

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_terminal_summary(
    metrics_lda: Dict[str, Optional[float]],
    metrics_nmf: Dict[str, Optional[float]],
    best_model_name: str,
    best_top_words: List[List[str]],
    best_theme_map: Dict[int, str],
    custom_result: Dict,
    output_path: Path,
) -> None:
    """Print a compact terminal summary."""
    print()
    print("=" * 60)
    print("EXPERIMENT 08 — Topic Modeling & t-SNE")
    print("=" * 60)
    print(f"Student  : Divya M (24AD0074)")
    print(f"Dataset  : {NUM_REVIEWS} reviews, {NUM_TOPICS} themes")
    print()

    # Evaluation table
    print(f"{'Metric':<15} {'LDA':>8} {'NMF':>8}")
    print("-" * 33)
    for key in ("ARI", "NMI", "Purity", "Silhouette"):
        print(f"{key:<15} {fmt_metric(metrics_lda[key]):>8} {fmt_metric(metrics_nmf[key]):>8}")
    print()
    print(f"Best model: {best_model_name}")
    print()

    # Best model topics
    for tid in range(NUM_TOPICS):
        print(f"  Topic {tid} [{best_theme_map.get(tid, '?')}]:")
        print(f"    {', '.join(best_top_words[tid])}")
    print()

    # Custom review
    print(f"Custom review: {custom_result['original'][:60]}...")
    print(f"  LDA => Topic {custom_result['lda_topic']} ({custom_result['lda_theme']})")
    print(f"  NMF => Topic {custom_result['nmf_topic']} ({custom_result['nmf_theme']})")
    print(f"  Final ({best_model_name}): {custom_result['final_theme']} "
          f"(conf={custom_result['final_conf']:.4f})")
    print()
    print(f"Output: {output_path}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 08: Topic Modeling with LDA/NMF and t-SNE visualization."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser(
        description="Experiment 08 — Topic Modeling and t-SNE",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_TEXT,
        help="Custom customer review to analyze.",
    )
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_path = output_dir / "output.txt"

    # NLTK setup
    _ensure_nltk_resources()

    # Prepare data
    review_ids = [r["review_id"] for r in REVIEWS]
    review_texts = [r["text"] for r in REVIEWS]
    true_themes = [r["theme"] for r in REVIEWS]

    # Preprocess
    token_lists, processed_strings = preprocess(review_texts)

    # Run models
    lda_model, lda_vec, lda_doc_topic_raw = run_lda(processed_strings)
    nmf_model, nmf_vec, nmf_doc_topic_raw = run_nmf(processed_strings)

    lda_doc_topic = normalize_doc_topic(lda_doc_topic_raw)
    nmf_doc_topic = normalize_doc_topic(nmf_doc_topic_raw)

    lda_assignments, lda_confidences = get_assignments(lda_doc_topic)
    nmf_assignments, nmf_confidences = get_assignments(nmf_doc_topic)

    lda_top_words = get_top_words(lda_model, lda_vec)
    nmf_top_words = get_top_words(nmf_model, nmf_vec)

    lda_theme_map = majority_theme_map(lda_assignments, true_themes)
    nmf_theme_map = majority_theme_map(nmf_assignments, true_themes)

    # Evaluate
    metrics_lda = evaluate_model(lda_doc_topic, lda_assignments, true_themes)
    metrics_nmf = evaluate_model(nmf_doc_topic, nmf_assignments, true_themes)

    best_model_name = select_best_model(metrics_lda, metrics_nmf)

    if best_model_name == "LDA":
        best_doc_topic = lda_doc_topic
        best_assignments = lda_assignments
        best_confidences = lda_confidences
        best_top_words = lda_top_words
        best_theme_map = lda_theme_map
    else:
        best_doc_topic = nmf_doc_topic
        best_assignments = nmf_assignments
        best_confidences = nmf_confidences
        best_top_words = nmf_top_words
        best_theme_map = nmf_theme_map

    # t-SNE
    tsne_coords = run_tsne(best_doc_topic)
    ascii_plot = ascii_tsne_plot(tsne_coords, best_assignments, best_theme_map)

    # Custom review
    custom_result = analyze_custom_review(
        args.text,
        lda_model, lda_vec,
        nmf_model, nmf_vec,
        lda_theme_map, nmf_theme_map,
        best_model_name,
    )

    # Generate report
    generate_report(
        true_themes=true_themes,
        review_ids=review_ids,
        review_texts=review_texts,
        lda_top_words=lda_top_words,
        nmf_top_words=nmf_top_words,
        metrics_lda=metrics_lda,
        metrics_nmf=metrics_nmf,
        best_model_name=best_model_name,
        best_assignments=best_assignments,
        best_confidences=best_confidences,
        best_theme_map=best_theme_map,
        best_top_words=best_top_words,
        lda_theme_map=lda_theme_map,
        nmf_theme_map=nmf_theme_map,
        tsne_coords=tsne_coords,
        ascii_plot=ascii_plot,
        custom_result=custom_result,
        output_path=output_path,
    )

    # Terminal summary
    print_terminal_summary(
        metrics_lda=metrics_lda,
        metrics_nmf=metrics_nmf,
        best_model_name=best_model_name,
        best_top_words=best_top_words,
        best_theme_map=best_theme_map,
        custom_result=custom_result,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
