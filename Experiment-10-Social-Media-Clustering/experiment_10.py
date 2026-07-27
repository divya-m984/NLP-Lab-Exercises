"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 10
Title           : UTILIZE WORD AND PHRASE-BASED CLUSTERING ALGORITHMS TO IDENTIFY PATTERNS
                  IN SOCIAL MEDIA CONVERSATIONS AND ANALYZE THEIR IMPLICATIONS FOR
                  MARKETING STRATEGIES.
"""

import argparse
import re
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
N_CLUSTERS: int = 5
N_INIT: int = 20
MAX_ITER: int = 500
TOP_TERMS: int = 8
FLOAT_TOL: float = 1e-9

THEME_ORDER: List[str] = [
    "product_quality",
    "delivery_experience",
    "customer_support",
    "pricing_promotions",
    "sustainability_brand",
]

SENTIMENT_ORDER: List[str] = ["positive", "negative", "neutral"]

STUDENT_NAME: str = "Divya M"
REGISTER_NUMBER: str = "24AD0074"

EXPERIMENT_TITLE: str = (
    "UTILIZE WORD AND PHRASE-BASED CLUSTERING ALGORITHMS TO IDENTIFY PATTERNS\n"
    "IN SOCIAL MEDIA CONVERSATIONS AND ANALYZE THEIR IMPLICATIONS FOR\n"
    "MARKETING STRATEGIES."
)

AIM: str = (
    "To apply clustering techniques on social media posts using TF-IDF\n"
    "(Term Frequency\u2013Inverse Document Frequency) and K-Means to identify\n"
    "customer trends and marketing insights."
)

DEFAULT_CUSTOM_TEXT: str = (
    "Love the new design, but my order arrived late and support never replied."
)

METHOD_NAMES: List[str] = [
    "Word-Based K-Means",
    "Word-and-Phrase K-Means",
    "Agglomerative Word-and-Phrase",
]

# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------


def _ensure_nltk_resources() -> None:
    """Download only missing NLTK resources with readable error handling."""
    import nltk

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as exc:
                print(f"Warning: could not download NLTK resource '{name}': {exc}")


# ---------------------------------------------------------------------------
# Embedded dataset  (75 fictional social-media posts)
# ---------------------------------------------------------------------------


def build_dataset() -> List[Dict[str, str]]:
    """Return 75 social-media posts, 15 per theme, 5 per sentiment per theme."""

    data: List[Dict[str, str]] = [
        # ===== product_quality — positive =====
        {"id": "S01", "text": "Just unboxed my ZenithPro tablet and wow the build quality is incredible \U0001f44d #quality", "theme": "product_quality", "sentiment": "positive"},
        {"id": "S02", "text": "@NovaTech your latest headphones have amazing sound clarity, best purchase this year", "theme": "product_quality", "sentiment": "positive"},
        {"id": "S03", "text": "6 months with the ArcLight blender and not a single issue, super durable product #reliable", "theme": "product_quality", "sentiment": "positive"},
        {"id": "S04", "text": "The stitching on my VeloShoes is top notch, quality you can actually feel", "theme": "product_quality", "sentiment": "positive"},
        {"id": "S05", "text": "Switched to BrightEdge monitors for work and the display quality blew me away honestly", "theme": "product_quality", "sentiment": "positive"},
        # ===== product_quality — negative =====
        {"id": "S06", "text": "My CrestLine laptop screen cracked after two weeks, terrible build quality #disappointed", "theme": "product_quality", "sentiment": "negative"},
        {"id": "S07", "text": "@PulseGear the zipper on your backpack broke on day one, very poor craftsmanship", "theme": "product_quality", "sentiment": "negative"},
        {"id": "S08", "text": "Bought the OrbitFan and it started rattling within a month, feels cheaply made", "theme": "product_quality", "sentiment": "negative"},
        {"id": "S09", "text": "Paint peeling off my GlideBoard after a few rides, unacceptable product defect", "theme": "product_quality", "sentiment": "negative"},
        {"id": "S10", "text": "Third replacement from @StellarAudio and still getting static, quality control is nonexistent", "theme": "product_quality", "sentiment": "negative"},
        # ===== product_quality — neutral =====
        {"id": "S11", "text": "The PeakLens camera takes decent photos, nothing groundbreaking but does the job", "theme": "product_quality", "sentiment": "neutral"},
        {"id": "S12", "text": "Has anyone compared NovaTech earbuds with the older model? Curious about build changes", "theme": "product_quality", "sentiment": "neutral"},
        {"id": "S13", "text": "ZenithPro tablet feels similar to last gen, screen is fine battery is average", "theme": "product_quality", "sentiment": "neutral"},
        {"id": "S14", "text": "The SummitPack material seems standard for the price range, nothing special", "theme": "product_quality", "sentiment": "neutral"},
        {"id": "S15", "text": "Got the new ArcLight mixer, build seems ok will update after more use #firstimpressions", "theme": "product_quality", "sentiment": "neutral"},

        # ===== delivery_experience — positive =====
        {"id": "S16", "text": "Order arrived a full day early with perfect packaging, kudos @SwiftShip \U0001f4e6", "theme": "delivery_experience", "sentiment": "positive"},
        {"id": "S17", "text": "Fastest delivery I have ever had, two days coast to coast #impressed #shipping", "theme": "delivery_experience", "sentiment": "positive"},
        {"id": "S18", "text": "@QuickRoute your tracking updates are so detailed, loved knowing exactly when my parcel would arrive", "theme": "delivery_experience", "sentiment": "positive"},
        {"id": "S19", "text": "Same day delivery on a holiday weekend, did not expect that from NovaMart", "theme": "delivery_experience", "sentiment": "positive"},
        {"id": "S20", "text": "Package came in mint condition with eco friendly wrapping, great shipping experience", "theme": "delivery_experience", "sentiment": "positive"},
        # ===== delivery_experience — negative =====
        {"id": "S21", "text": "Still waiting for my order placed three weeks ago, @SwiftShip zero updates on tracking", "theme": "delivery_experience", "sentiment": "negative"},
        {"id": "S22", "text": "Delivery driver left my package in the rain and it was completely soaked #terrible", "theme": "delivery_experience", "sentiment": "negative"},
        {"id": "S23", "text": "Paid for express shipping and it took longer than standard, total waste of money", "theme": "delivery_experience", "sentiment": "negative"},
        {"id": "S24", "text": "@QuickRoute delivered to wrong address twice now, this is getting ridiculous", "theme": "delivery_experience", "sentiment": "negative"},
        {"id": "S25", "text": "Box arrived crushed and the item inside was damaged, worst delivery experience ever", "theme": "delivery_experience", "sentiment": "negative"},
        # ===== delivery_experience — neutral =====
        {"id": "S26", "text": "Package arrived on the estimated date nothing special, standard shipping", "theme": "delivery_experience", "sentiment": "neutral"},
        {"id": "S27", "text": "Does anyone know the average delivery time for NovaMart orders to the west coast?", "theme": "delivery_experience", "sentiment": "neutral"},
        {"id": "S28", "text": "Tracking said out for delivery this morning, still showing in transit now", "theme": "delivery_experience", "sentiment": "neutral"},
        {"id": "S29", "text": "Got my order in about a week, packaging was plain but item was fine", "theme": "delivery_experience", "sentiment": "neutral"},
        {"id": "S30", "text": "Shipping was neither fast nor slow just the usual timeline #delivery", "theme": "delivery_experience", "sentiment": "neutral"},

        # ===== customer_support — positive =====
        {"id": "S31", "text": "@NovaTech support resolved my issue in under ten minutes, amazing service \U0001f64f", "theme": "customer_support", "sentiment": "positive"},
        {"id": "S32", "text": "Called customer service and they actually listened and fixed everything on the first call", "theme": "customer_support", "sentiment": "positive"},
        {"id": "S33", "text": "Huge shoutout to the live chat team at VeloShoes, super helpful and friendly #support", "theme": "customer_support", "sentiment": "positive"},
        {"id": "S34", "text": "Got a full refund plus a discount code for the trouble, top tier customer care", "theme": "customer_support", "sentiment": "positive"},
        {"id": "S35", "text": "Support agent went above and beyond to track my missing order, really impressed", "theme": "customer_support", "sentiment": "positive"},
        # ===== customer_support — negative =====
        {"id": "S36", "text": "Been on hold for over an hour with @PulseGear support, this is unacceptable", "theme": "customer_support", "sentiment": "negative"},
        {"id": "S37", "text": "Customer support keeps giving me scripted responses, nobody actually reads my complaint", "theme": "customer_support", "sentiment": "negative"},
        {"id": "S38", "text": "Emailed support three times and got zero reply, worst customer service I have seen", "theme": "customer_support", "sentiment": "negative"},
        {"id": "S39", "text": "@StellarAudio your chatbot is useless and there is no way to reach a real person #frustrated", "theme": "customer_support", "sentiment": "negative"},
        {"id": "S40", "text": "Waited two weeks for a replacement and support stopped responding completely", "theme": "customer_support", "sentiment": "negative"},
        # ===== customer_support — neutral =====
        {"id": "S41", "text": "Contacted NovaTech support about warranty terms, got a standard reply with a link", "theme": "customer_support", "sentiment": "neutral"},
        {"id": "S42", "text": "Anyone know if BrightEdge has weekend support hours or only weekdays?", "theme": "customer_support", "sentiment": "neutral"},
        {"id": "S43", "text": "Support ticket is open for five days now, no resolution yet but they acknowledged it", "theme": "customer_support", "sentiment": "neutral"},
        {"id": "S44", "text": "The FAQ section covers most issues, did not need to call support directly", "theme": "customer_support", "sentiment": "neutral"},
        {"id": "S45", "text": "Used the help form on the website, response time was about 48 hours #customerservice", "theme": "customer_support", "sentiment": "neutral"},

        # ===== pricing_promotions — positive =====
        {"id": "S46", "text": "Grabbed the ZenithPro at 40 percent off during the flash sale, absolute steal \U0001f525", "theme": "pricing_promotions", "sentiment": "positive"},
        {"id": "S47", "text": "@NovaMart your loyalty rewards program is honestly the best, saved so much this year", "theme": "pricing_promotions", "sentiment": "positive"},
        {"id": "S48", "text": "Bundle deal on ArcLight appliances is unbeatable value, got three items for the price of two", "theme": "pricing_promotions", "sentiment": "positive"},
        {"id": "S49", "text": "Love that they offer student discounts, makes quality products actually affordable #deals", "theme": "pricing_promotions", "sentiment": "positive"},
        {"id": "S50", "text": "Used a coupon code from their newsletter and got free shipping plus 20 pct off nice", "theme": "pricing_promotions", "sentiment": "positive"},
        # ===== pricing_promotions — negative =====
        {"id": "S51", "text": "The sale price is barely cheaper than the regular price, feels like a fake discount", "theme": "pricing_promotions", "sentiment": "negative"},
        {"id": "S52", "text": "@PulseGear raised prices right before the sale so the discount means nothing #scam", "theme": "pricing_promotions", "sentiment": "negative"},
        {"id": "S53", "text": "Promo code did not work at checkout even though it should still be valid, frustrating", "theme": "pricing_promotions", "sentiment": "negative"},
        {"id": "S54", "text": "Paid full price last week and now it is on sale, no price adjustment offered at all", "theme": "pricing_promotions", "sentiment": "negative"},
        {"id": "S55", "text": "Membership fee went up but the discounts got worse, not worth renewing anymore", "theme": "pricing_promotions", "sentiment": "negative"},
        # ===== pricing_promotions — neutral =====
        {"id": "S56", "text": "Prices seem about average compared to other brands, nothing shocking either way", "theme": "pricing_promotions", "sentiment": "neutral"},
        {"id": "S57", "text": "Does anyone know when the next NovaMart seasonal sale is? Want to compare prices first", "theme": "pricing_promotions", "sentiment": "neutral"},
        {"id": "S58", "text": "Checked the loyalty points balance and it covers maybe a small accessory, meh", "theme": "pricing_promotions", "sentiment": "neutral"},
        {"id": "S59", "text": "The promo says up to 30 pct off but most items are only 10 pct, read the fine print", "theme": "pricing_promotions", "sentiment": "neutral"},
        {"id": "S60", "text": "Price list updated on the site, some things went up some went down #pricing", "theme": "pricing_promotions", "sentiment": "neutral"},

        # ===== sustainability_brand — positive =====
        {"id": "S61", "text": "Love that @GreenVista uses fully recyclable packaging for every order #ecofriendly", "theme": "sustainability_brand", "sentiment": "positive"},
        {"id": "S62", "text": "NovaTech committed to carbon neutral operations by next year, great to see brands step up", "theme": "sustainability_brand", "sentiment": "positive"},
        {"id": "S63", "text": "Switched to EcoBlend products because they actually publish their sustainability reports \U0001f331", "theme": "sustainability_brand", "sentiment": "positive"},
        {"id": "S64", "text": "The refill program from ArcLight reduces so much waste, genuinely good initiative #green", "theme": "sustainability_brand", "sentiment": "positive"},
        {"id": "S65", "text": "Bought the organic line from VeloShoes and love that they source materials ethically", "theme": "sustainability_brand", "sentiment": "positive"},
        # ===== sustainability_brand — negative =====
        {"id": "S66", "text": "@PulseGear claiming eco friendly but still using plastic wrap on everything, classic greenwashing", "theme": "sustainability_brand", "sentiment": "negative"},
        {"id": "S67", "text": "Their sustainability page is all buzzwords and zero data, show me the actual numbers", "theme": "sustainability_brand", "sentiment": "negative"},
        {"id": "S68", "text": "Ordered the green collection and it came in non recyclable styrofoam, so much for being eco", "theme": "sustainability_brand", "sentiment": "negative"},
        {"id": "S69", "text": "Brand talks about saving the planet but ships individual items in massive boxes #wasteful", "theme": "sustainability_brand", "sentiment": "negative"},
        {"id": "S70", "text": "No transparency on supply chain ethics from @StellarAudio despite their green branding", "theme": "sustainability_brand", "sentiment": "negative"},
        # ===== sustainability_brand — neutral =====
        {"id": "S71", "text": "GreenVista released a new sustainability report, haven't read it yet but it's available", "theme": "sustainability_brand", "sentiment": "neutral"},
        {"id": "S72", "text": "Curious how NovaTech measures their carbon footprint, does anyone have details?", "theme": "sustainability_brand", "sentiment": "neutral"},
        {"id": "S73", "text": "The eco label is on the box but I am not sure what certification it actually refers to", "theme": "sustainability_brand", "sentiment": "neutral"},
        {"id": "S74", "text": "Some products in the green range cost more but unclear if manufacturing is different", "theme": "sustainability_brand", "sentiment": "neutral"},
        {"id": "S75", "text": "Brand mentions ethical sourcing on their about page, standard language though #sustainability", "theme": "sustainability_brand", "sentiment": "neutral"},
    ]

    return data


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(text: str) -> Tuple[List[str], str]:
    """Shared preprocessing for social-media posts.

    Steps: lowercase, remove URLs, remove @mentions, strip # but keep
    hashtag word, tokenize, keep alphabetic, remove stopwords, lemmatize,
    remove tokens < 2 chars.

    Returns (token_list, joined_string).
    """
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]

    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]

    return tokens, " ".join(tokens)


# ---------------------------------------------------------------------------
# Marketing insight generation
# ---------------------------------------------------------------------------

MARKETING_INSIGHTS: Dict[str, Dict[str, str]] = {
    "product_quality": {
        "positive": "Promote reliability and customer testimonials.",
        "negative": "Prioritize defect analysis and product improvements.",
        "neutral": "Publish clearer feature demonstrations.",
    },
    "delivery_experience": {
        "positive": "Highlight reliable shipping performance.",
        "negative": "Investigate delays and improve tracking communication.",
        "neutral": "Communicate delivery expectations more clearly.",
    },
    "customer_support": {
        "positive": "Showcase responsive service.",
        "negative": "Reduce response time and improve escalation.",
        "neutral": "Publish support channels and service-hour information.",
    },
    "pricing_promotions": {
        "positive": "Continue effective offers and loyalty campaigns.",
        "negative": "Review pricing clarity and perceived value.",
        "neutral": "Explain promotion conditions more clearly.",
    },
    "sustainability_brand": {
        "positive": "Strengthen sustainability-focused messaging.",
        "negative": "Provide evidence for environmental claims.",
        "neutral": "Publish measurable sustainability progress.",
    },
}


def get_marketing_insight(theme: str, sentiment: str) -> str:
    """Deterministic marketing insight from cluster theme and sentiment."""
    theme_map = MARKETING_INSIGHTS.get(theme)
    if theme_map is None:
        return "No specific insight available for this theme."
    return theme_map.get(sentiment, "No specific insight available for this sentiment.")


# ---------------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------------


def validate_features(matrix, post_ids: List[str], method_label: str) -> None:
    """Validate TF-IDF matrix: non-empty vocab, no zero-vector docs, finite."""
    if matrix.shape[1] == 0:
        raise ValueError(f"[{method_label}] Vocabulary is empty after vectorization.")
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    zero_rows = np.where(row_sums == 0)[0]
    if len(zero_rows) > 0:
        bad_ids = [post_ids[i] for i in zero_rows]
        raise ValueError(
            f"[{method_label}] Zero-vector documents found: {bad_ids}. "
            "Cannot assign a meaningful cluster."
        )
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    if not np.all(np.isfinite(dense)):
        raise ValueError(f"[{method_label}] Non-finite values found in TF-IDF matrix.")


# ---------------------------------------------------------------------------
# Clustering evaluation
# ---------------------------------------------------------------------------


def compute_purity(labels_true: List[str], labels_pred: List[int]) -> float:
    """Cluster purity: fraction of posts matching majority true theme."""
    clusters: Dict[int, List[str]] = {}
    for true, pred in zip(labels_true, labels_pred):
        clusters.setdefault(pred, []).append(true)
    correct = sum(Counter(members).most_common(1)[0][1] for members in clusters.values())
    return correct / len(labels_true)


def cluster_majority_theme(labels_true: List[str], labels_pred: List[int]) -> Dict[int, str]:
    """Map each cluster ID to the majority true theme."""
    clusters: Dict[int, List[str]] = {}
    for true, pred in zip(labels_true, labels_pred):
        clusters.setdefault(pred, []).append(true)
    return {cid: Counter(members).most_common(1)[0][0] for cid, members in clusters.items()}


def cluster_dominant_sentiment(
    sentiments: List[str], labels_pred: List[int]
) -> Dict[int, str]:
    """Map each cluster ID to the dominant sentiment."""
    clusters: Dict[int, List[str]] = {}
    for sent, pred in zip(sentiments, labels_pred):
        clusters.setdefault(pred, []).append(sent)
    return {cid: Counter(members).most_common(1)[0][0] for cid, members in clusters.items()}


def cluster_sentiment_dist(
    sentiments: List[str], labels_pred: List[int]
) -> Dict[int, Dict[str, int]]:
    """Sentiment distribution per cluster."""
    clusters: Dict[int, List[str]] = {}
    for sent, pred in zip(sentiments, labels_pred):
        clusters.setdefault(pred, []).append(sent)
    return {cid: dict(Counter(members)) for cid, members in clusters.items()}


def evaluate_clustering(
    labels_true: List[str],
    labels_pred: List[int],
    matrix,
    elapsed: float,
) -> Dict[str, float]:
    """Compute ARI, NMI, purity, silhouette, n_clusters, time."""
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )

    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    n_unique = len(set(labels_pred))

    sil = float("nan")
    if 2 <= n_unique <= len(labels_true) - 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sil = silhouette_score(dense, labels_pred, metric="cosine")

    return {
        "ARI": adjusted_rand_score(labels_true, labels_pred),
        "NMI": normalized_mutual_info_score(labels_true, labels_pred),
        "Purity": compute_purity(labels_true, labels_pred),
        "Silhouette": sil,
        "N_Clusters": n_unique,
        "Time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Best method selection
# ---------------------------------------------------------------------------


def select_best_method(
    results: List[Tuple[str, Dict[str, float]]],
) -> Tuple[str, Dict[str, float], str]:
    """Select best method using ARI > NMI > Purity > Silhouette > order.

    Returns (method_name, scores, tie_info).
    """
    criteria = ["ARI", "NMI", "Purity", "Silhouette"]

    def sort_key(item: Tuple[int, str, Dict[str, float]]) -> Tuple:
        idx, _, scores = item
        vals = []
        for c in criteria:
            v = scores[c]
            vals.append(v if np.isfinite(v) else -999.0)
        vals.append(-idx)  # deterministic tie-break by method order
        return tuple(vals)

    indexed = [(i, name, scores) for i, (name, scores) in enumerate(results)]
    indexed.sort(key=sort_key, reverse=True)

    best_idx, best_name, best_scores = indexed[0]
    # check for ties
    tie_info = "No tie."
    if len(indexed) > 1:
        _, second_name, second_scores = indexed[1]
        tied = all(
            abs(best_scores[c] - second_scores[c]) < FLOAT_TOL
            if np.isfinite(best_scores[c]) and np.isfinite(second_scores[c])
            else best_scores[c] == second_scores[c]
            for c in criteria
        )
        if tied:
            tie_info = f"Tie between {best_name} and {second_name}; resolved by method order."

    return best_name, best_scores, tie_info


# ---------------------------------------------------------------------------
# Top TF-IDF terms for a cluster
# ---------------------------------------------------------------------------


def top_tfidf_terms(
    matrix, labels_pred: List[int], feature_names: List[str], n_top: int = TOP_TERMS
) -> Dict[int, List[str]]:
    """Top TF-IDF terms per cluster by averaging document vectors."""
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    cluster_ids = sorted(set(labels_pred))
    result: Dict[int, List[str]] = {}
    for cid in cluster_ids:
        mask = np.array([i for i, l in enumerate(labels_pred) if l == cid])
        if len(mask) == 0:
            result[cid] = []
            continue
        mean_vec = dense[mask].mean(axis=0)
        top_idx = mean_vec.argsort()[::-1][:n_top]
        result[cid] = [feature_names[i] for i in top_idx if mean_vec[i] > 0]
    return result


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def cosine_sim(a, b) -> float:
    """Cosine similarity between two 1-D arrays."""
    a = np.asarray(a).ravel().astype(float)
    b = np.asarray(b).ravel().astype(float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    return max(-1.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 10: Social Media Clustering"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_CUSTOM_TEXT,
        help="Custom social media post for prediction.",
    )
    args = parser.parse_args()

    # --- NLTK ---
    _ensure_nltk_resources()

    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    # --- Dataset ---
    dataset = build_dataset()
    post_ids = [d["id"] for d in dataset]
    texts = [d["text"] for d in dataset]
    true_themes = [d["theme"] for d in dataset]
    sentiments = [d["sentiment"] for d in dataset]

    # --- Preprocess ---
    processed = [preprocess(t) for t in texts]
    processed_tokens = [p[0] for p in processed]
    processed_strings = [p[1] for p in processed]

    # -----------------------------------------------------------------------
    # Method A: Word-Based K-Means (unigram)
    # -----------------------------------------------------------------------
    vec_word = TfidfVectorizer(
        ngram_range=(1, 1), min_df=1, max_df=0.95,
        sublinear_tf=True, norm="l2",
    )
    tfidf_word = vec_word.fit_transform(processed_strings)
    validate_features(tfidf_word, post_ids, METHOD_NAMES[0])

    t0 = time.time()
    km_word = KMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_SEED,
        n_init=N_INIT, max_iter=MAX_ITER,
    )
    labels_word = km_word.fit_predict(tfidf_word)
    time_word = time.time() - t0

    # -----------------------------------------------------------------------
    # Method B: Word-and-Phrase K-Means (unigram + bigram)
    # -----------------------------------------------------------------------
    vec_phrase = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_df=0.95,
        sublinear_tf=True, norm="l2",
    )
    tfidf_phrase = vec_phrase.fit_transform(processed_strings)
    validate_features(tfidf_phrase, post_ids, METHOD_NAMES[1])

    t0 = time.time()
    km_phrase = KMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_SEED,
        n_init=N_INIT, max_iter=MAX_ITER,
    )
    labels_phrase = km_phrase.fit_predict(tfidf_phrase)
    time_phrase = time.time() - t0

    # -----------------------------------------------------------------------
    # Method C: Agglomerative Word-and-Phrase
    # -----------------------------------------------------------------------
    tfidf_phrase_dense = tfidf_phrase.toarray()

    try:
        agg = AgglomerativeClustering(
            n_clusters=N_CLUSTERS, metric="cosine", linkage="average",
        )
        t0 = time.time()
        labels_agg = agg.fit_predict(tfidf_phrase_dense)
        time_agg = time.time() - t0
    except TypeError:
        agg = AgglomerativeClustering(
            n_clusters=N_CLUSTERS, affinity="cosine", linkage="average",
        )
        t0 = time.time()
        labels_agg = agg.fit_predict(tfidf_phrase_dense)
        time_agg = time.time() - t0

    labels_word_list = labels_word.tolist()
    labels_phrase_list = labels_phrase.tolist()
    labels_agg_list = labels_agg.tolist()

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    eval_word = evaluate_clustering(true_themes, labels_word_list, tfidf_word, time_word)
    eval_phrase = evaluate_clustering(true_themes, labels_phrase_list, tfidf_phrase, time_phrase)
    eval_agg = evaluate_clustering(true_themes, labels_agg_list, tfidf_phrase, time_agg)

    all_results: List[Tuple[str, Dict[str, float]]] = [
        (METHOD_NAMES[0], eval_word),
        (METHOD_NAMES[1], eval_phrase),
        (METHOD_NAMES[2], eval_agg),
    ]

    best_name, best_scores, tie_info = select_best_method(all_results)

    # -----------------------------------------------------------------------
    # Cluster interpretation
    # -----------------------------------------------------------------------
    feat_word = vec_word.get_feature_names_out().tolist()
    feat_phrase = vec_phrase.get_feature_names_out().tolist()

    top_word = top_tfidf_terms(tfidf_word, labels_word_list, feat_word)
    top_phrase = top_tfidf_terms(tfidf_phrase, labels_phrase_list, feat_phrase)
    top_agg = top_tfidf_terms(tfidf_phrase, labels_agg_list, feat_phrase)

    maj_word = cluster_majority_theme(true_themes, labels_word_list)
    maj_phrase = cluster_majority_theme(true_themes, labels_phrase_list)
    maj_agg = cluster_majority_theme(true_themes, labels_agg_list)

    dom_sent_word = cluster_dominant_sentiment(sentiments, labels_word_list)
    dom_sent_phrase = cluster_dominant_sentiment(sentiments, labels_phrase_list)
    dom_sent_agg = cluster_dominant_sentiment(sentiments, labels_agg_list)

    sent_dist_word = cluster_sentiment_dist(sentiments, labels_word_list)
    sent_dist_phrase = cluster_sentiment_dist(sentiments, labels_phrase_list)
    sent_dist_agg = cluster_sentiment_dist(sentiments, labels_agg_list)

    # -----------------------------------------------------------------------
    # Word vs Phrase comparison
    # -----------------------------------------------------------------------
    ari_diff = eval_phrase["ARI"] - eval_word["ARI"]
    nmi_diff = eval_phrase["NMI"] - eval_word["NMI"]
    pur_diff = eval_phrase["Purity"] - eval_word["Purity"]
    sil_diff = (eval_phrase["Silhouette"] - eval_word["Silhouette"]
                if np.isfinite(eval_phrase["Silhouette"]) and np.isfinite(eval_word["Silhouette"])
                else float("nan"))

    if abs(ari_diff) < FLOAT_TOL and abs(nmi_diff) < FLOAT_TOL:
        phrase_verdict = "Including bigram phrases produced a tied clustering quality."
    elif ari_diff > FLOAT_TOL:
        phrase_verdict = "Including bigram phrases improved the clustering quality."
    else:
        phrase_verdict = "Including bigram phrases reduced the clustering quality."

    # -----------------------------------------------------------------------
    # Best method data for assignment table
    # -----------------------------------------------------------------------
    if best_name == METHOD_NAMES[0]:
        best_labels = labels_word_list
        best_maj = maj_word
        best_matrix = tfidf_word
        best_centroids = km_word.cluster_centers_
        best_is_kmeans = True
    elif best_name == METHOD_NAMES[1]:
        best_labels = labels_phrase_list
        best_maj = maj_phrase
        best_matrix = tfidf_phrase
        best_centroids = km_phrase.cluster_centers_
        best_is_kmeans = True
    else:
        best_labels = labels_agg_list
        best_maj = maj_agg
        best_matrix = tfidf_phrase
        best_is_kmeans = False
        # compute mean vectors for agglomerative clusters
        best_centroids_dict: Dict[int, np.ndarray] = {}
        for cid in sorted(set(labels_agg_list)):
            mask = [i for i, l in enumerate(labels_agg_list) if l == cid]
            best_centroids_dict[cid] = tfidf_phrase_dense[mask].mean(axis=0)

    # Compute similarities for all 75 posts
    def post_similarity(idx: int) -> float:
        dense_row = best_matrix[idx].toarray().ravel() if hasattr(best_matrix[idx], "toarray") else np.asarray(best_matrix[idx]).ravel()
        cid = best_labels[idx]
        if best_is_kmeans:
            centroid = best_centroids[cid]
        else:
            centroid = best_centroids_dict[cid]
        return cosine_sim(dense_row, centroid)

    # -----------------------------------------------------------------------
    # Custom post prediction
    # -----------------------------------------------------------------------
    custom_text = args.text
    custom_tokens, custom_string = preprocess(custom_text)

    def predict_custom_kmeans(vectorizer, km_model, method_label: str) -> Tuple[int, str, float]:
        vec = vectorizer.transform([custom_string])
        if vec.nnz == 0:
            print(f"[{method_label}] Custom post has no known vocabulary terms.")
            return -1, "unknown", 0.0
        pred = km_model.predict(vec)[0]
        maj = cluster_majority_theme(true_themes, km_model.labels_.tolist())
        sim = cosine_sim(vec.toarray().ravel(), km_model.cluster_centers_[pred])
        return int(pred), maj.get(pred, "unknown"), sim

    cust_word_cluster, cust_word_theme, cust_word_sim = predict_custom_kmeans(
        vec_word, km_word, METHOD_NAMES[0]
    )
    cust_phrase_cluster, cust_phrase_theme, cust_phrase_sim = predict_custom_kmeans(
        vec_phrase, km_phrase, METHOD_NAMES[1]
    )

    # Agglomerative custom prediction
    cust_agg_vec = vec_phrase.transform([custom_string])
    if cust_agg_vec.nnz == 0:
        cust_agg_cluster, cust_agg_theme, cust_agg_sim = -1, "unknown", 0.0
        print("[Agglomerative] Custom post has no known vocabulary terms.")
    else:
        agg_mean_vecs: Dict[int, np.ndarray] = {}
        for cid in sorted(set(labels_agg_list)):
            mask = [i for i, l in enumerate(labels_agg_list) if l == cid]
            agg_mean_vecs[cid] = tfidf_phrase_dense[mask].mean(axis=0)
        cust_agg_dense = cust_agg_vec.toarray().ravel()
        best_cid = -1
        best_sim_val = -2.0
        for cid, mvec in agg_mean_vecs.items():
            s = cosine_sim(cust_agg_dense, mvec)
            if s > best_sim_val:
                best_sim_val = s
                best_cid = cid
        cust_agg_cluster = best_cid
        cust_agg_theme = maj_agg.get(best_cid, "unknown")
        cust_agg_sim = best_sim_val

    # Final theme from best method
    if best_name == METHOD_NAMES[0]:
        final_theme = cust_word_theme
    elif best_name == METHOD_NAMES[1]:
        final_theme = cust_phrase_theme
    else:
        final_theme = cust_agg_theme

    # Determine dominant sentiment of custom post cluster in best method
    if best_name == METHOD_NAMES[0]:
        final_dom_sent = dom_sent_word.get(cust_word_cluster, "neutral")
    elif best_name == METHOD_NAMES[1]:
        final_dom_sent = dom_sent_phrase.get(cust_phrase_cluster, "neutral")
    else:
        final_dom_sent = dom_sent_agg.get(cust_agg_cluster, "neutral")

    final_insight = get_marketing_insight(final_theme, final_dom_sent)

    # -----------------------------------------------------------------------
    # Generate report  (output/output.txt)
    # -----------------------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "output.txt"

    lines: List[str] = []
    W = 78

    def sep() -> None:
        lines.append("=" * W)

    def heading(title: str) -> None:
        sep()
        lines.append(title.center(W))
        sep()

    heading("EXPERIMENT 10 — SOCIAL MEDIA CLUSTERING")
    lines.append(f"Student Name    : {STUDENT_NAME}")
    lines.append(f"Register Number : {REGISTER_NUMBER}")
    lines.append("")
    lines.append("Title:")
    lines.append(EXPERIMENT_TITLE)
    lines.append("")
    lines.append("Aim:")
    lines.append(AIM)
    lines.append("")

    # Dataset summary
    heading("DATASET SUMMARY")
    lines.append(f"Total posts       : {len(dataset)}")
    lines.append(f"Themes            : {N_CLUSTERS}")
    lines.append(f"Posts per theme    : 15")
    lines.append(f"Sentiments/theme  : 5 positive, 5 negative, 5 neutral")
    lines.append("")
    lines.append("Theme counts:")
    for theme in THEME_ORDER:
        count = sum(1 for d in dataset if d["theme"] == theme)
        lines.append(f"  {theme:30s} : {count}")
    lines.append("")
    lines.append("Sentiment counts:")
    for sent in SENTIMENT_ORDER:
        count = sum(1 for d in dataset if d["sentiment"] == sent)
        lines.append(f"  {sent:30s} : {count}")
    lines.append("")

    # Preprocessing
    heading("PREPROCESSING")
    lines.append("Shared preprocessing pipeline:")
    lines.append("  1. Convert to lowercase")
    lines.append("  2. Remove URLs")
    lines.append("  3. Remove @mentions")
    lines.append("  4. Strip # symbol, retain hashtag word")
    lines.append("  5. Tokenize with nltk.word_tokenize")
    lines.append("  6. Retain alphabetic tokens only")
    lines.append("  7. Remove English stopwords")
    lines.append("  8. WordNet lemmatization")
    lines.append("  9. Remove tokens shorter than 2 characters")
    lines.append("")
    lines.append("True themes and sentiment labels are NOT used during")
    lines.append("preprocessing or clustering. They are used only for")
    lines.append("post-hoc evaluation and interpretation.")
    lines.append("")

    # Method descriptions
    heading("CLUSTERING METHODS")
    lines.append("A. Word-Based K-Means")
    lines.append("   TF-IDF: ngram_range=(1,1), sublinear_tf, L2 norm")
    lines.append(f"   KMeans: k={N_CLUSTERS}, n_init={N_INIT}, max_iter={MAX_ITER}, seed={RANDOM_SEED}")
    lines.append("")
    lines.append("B. Word-and-Phrase K-Means  [REQUIRED METHOD]")
    lines.append("   TF-IDF: ngram_range=(1,2), sublinear_tf, L2 norm")
    lines.append(f"   KMeans: k={N_CLUSTERS}, n_init={N_INIT}, max_iter={MAX_ITER}, seed={RANDOM_SEED}")
    lines.append("   Uses both unigram and bigram phrase features.")
    lines.append("")
    lines.append("C. Agglomerative Word-and-Phrase")
    lines.append("   Same TF-IDF as method B (unigram + bigram)")
    lines.append(f"   AgglomerativeClustering: k={N_CLUSTERS}, cosine, average linkage")
    lines.append("")
    lines.append("Known themes are used ONLY to benchmark discovered clusters")
    lines.append("and are NOT provided to the algorithms during training.")
    lines.append("")

    # Evaluation table
    heading("EVALUATION RESULTS")
    header = f"{'Method':<35s} {'ARI':>7s} {'NMI':>7s} {'Purity':>7s} {'Silh.':>7s} {'k':>3s} {'Time':>7s}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, ev in all_results:
        sil_str = f"{ev['Silhouette']:.4f}" if np.isfinite(ev["Silhouette"]) else "  N/A"
        lines.append(
            f"{name:<35s} {ev['ARI']:>7.4f} {ev['NMI']:>7.4f} "
            f"{ev['Purity']:>7.4f} {sil_str:>7s} {int(ev['N_Clusters']):>3d} "
            f"{ev['Time_s']:>6.3f}s"
        )
    lines.append("")
    lines.append(f"Best method: {best_name}")
    lines.append(f"Tie info   : {tie_info}")
    lines.append("")

    # Word vs Phrase comparison
    heading("WORD-BASED vs WORD-AND-PHRASE K-MEANS COMPARISON")
    lines.append(f"Word-Based vocabulary size        : {len(feat_word)}")
    lines.append(f"Word-and-Phrase vocabulary size    : {len(feat_phrase)}")
    lines.append(f"ARI difference  (phrase - word)    : {ari_diff:+.4f}")
    lines.append(f"NMI difference  (phrase - word)    : {nmi_diff:+.4f}")
    lines.append(f"Purity difference                 : {pur_diff:+.4f}")
    sil_diff_str = f"{sil_diff:+.4f}" if np.isfinite(sil_diff) else "N/A"
    lines.append(f"Silhouette difference             : {sil_diff_str}")
    lines.append(f"Verdict: {phrase_verdict}")
    lines.append("")

    # Cluster summaries for each method
    method_info = [
        (METHOD_NAMES[0], labels_word_list, maj_word, dom_sent_word, sent_dist_word, top_word),
        (METHOD_NAMES[1], labels_phrase_list, maj_phrase, dom_sent_phrase, sent_dist_phrase, top_phrase),
        (METHOD_NAMES[2], labels_agg_list, maj_agg, dom_sent_agg, sent_dist_agg, top_agg),
    ]

    for mname, mlabels, mmaj, mdom, mdist, mtop in method_info:
        heading(f"CLUSTER DETAILS — {mname.upper()}")
        cluster_ids_sorted = sorted(set(mlabels))
        for cid in cluster_ids_sorted:
            members = [i for i, l in enumerate(mlabels) if l == cid]
            purity_c = sum(1 for i in members if true_themes[i] == mmaj[cid]) / len(members)
            lines.append(f"  Cluster {cid}:")
            lines.append(f"    Posts          : {len(members)}")
            lines.append(f"    Majority theme : {mmaj[cid]}")
            lines.append(f"    Dominant sent. : {mdom[cid]}")
            lines.append(f"    Sent. dist.    : {mdist[cid]}")
            lines.append(f"    Top terms      : {', '.join(mtop.get(cid, []))}")
            lines.append(f"    Cluster purity : {purity_c:.4f}")
            insight = get_marketing_insight(mmaj[cid], mdom[cid])
            lines.append(f"    Marketing insight: {insight}")
            lines.append("")

    # Marketing insights note
    heading("MARKETING INSIGHTS NOTE")
    lines.append("These are exploratory recommendations inferred from the")
    lines.append("embedded posts. They do not claim causal business conclusions.")
    lines.append("")

    # Assignment table for best method
    heading(f"ASSIGNMENT TABLE — {best_name.upper()} (BEST)")
    atbl_hdr = f"{'ID':>4s} {'True Theme':<25s} {'Sent.':<10s} {'Cl.':>3s} {'Interp. Theme':<25s} {'Sim.':>7s}"
    lines.append(atbl_hdr)
    lines.append("-" * len(atbl_hdr))
    for idx in range(len(dataset)):
        pid = post_ids[idx]
        tt = true_themes[idx]
        ss = sentiments[idx]
        cl = best_labels[idx]
        it = best_maj[cl]
        sim = post_similarity(idx)
        lines.append(f"{pid:>4s} {tt:<25s} {ss:<10s} {cl:>3d} {it:<25s} {sim:>7.4f}")
    lines.append("")

    # Custom post
    heading("CUSTOM POST ANALYSIS")
    lines.append(f"Original text     : {custom_text}")
    lines.append(f"Processed tokens  : {custom_tokens}")
    lines.append("")
    lines.append(f"Word-Based K-Means:")
    lines.append(f"  Cluster         : {cust_word_cluster}")
    lines.append(f"  Interp. theme   : {cust_word_theme}")
    lines.append(f"  Similarity      : {cust_word_sim:.4f}")
    lines.append("")
    lines.append(f"Word-and-Phrase K-Means:")
    lines.append(f"  Cluster         : {cust_phrase_cluster}")
    lines.append(f"  Interp. theme   : {cust_phrase_theme}")
    lines.append(f"  Similarity      : {cust_phrase_sim:.4f}")
    lines.append("")
    lines.append(f"Agglomerative Word-and-Phrase:")
    lines.append(f"  Cluster         : {cust_agg_cluster}")
    lines.append(f"  Interp. theme   : {cust_agg_theme}")
    lines.append(f"  Similarity      : {cust_agg_sim:.4f}")
    lines.append("")
    lines.append(f"Final theme (from {best_name}): {final_theme}")
    lines.append(f"Marketing insight: {final_insight}")
    lines.append("")

    # Final result
    heading("RESULT")
    lines.append(f"Best clustering method: {best_name}")
    lines.append(f"  ARI={best_scores['ARI']:.4f}  NMI={best_scores['NMI']:.4f}  "
                 f"Purity={best_scores['Purity']:.4f}  "
                 f"Silhouette={best_scores['Silhouette']:.4f}" if np.isfinite(best_scores['Silhouette'])
                 else f"  ARI={best_scores['ARI']:.4f}  NMI={best_scores['NMI']:.4f}  "
                      f"Purity={best_scores['Purity']:.4f}  Silhouette=N/A")
    lines.append("")
    lines.append(f"Phrase features effect: {phrase_verdict}")
    lines.append("")
    lines.append("Main exploratory customer trends identified:")
    best_method_maj = maj_word if best_name == METHOD_NAMES[0] else (maj_phrase if best_name == METHOD_NAMES[1] else maj_agg)
    best_method_dom = dom_sent_word if best_name == METHOD_NAMES[0] else (dom_sent_phrase if best_name == METHOD_NAMES[1] else dom_sent_agg)
    for cid in sorted(best_method_maj.keys()):
        lines.append(f"  Cluster {cid}: {best_method_maj[cid]} ({best_method_dom[cid]})")
    lines.append("")
    lines.append("NOTE: Results are specific to the embedded dataset of 75 posts.")
    lines.append("Cluster interpretations do not prove causal marketing effects.")
    lines.append("Scores should not be overstated when ARI, NMI, purity or")
    lines.append("silhouette values are modest.")
    sep()

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")

    # -----------------------------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  EXPERIMENT 10 — SOCIAL MEDIA CLUSTERING")
    print("=" * 60)
    print(f"Student  : {STUDENT_NAME}  ({REGISTER_NUMBER})")
    print(f"Dataset  : {len(dataset)} posts, {N_CLUSTERS} themes")
    print()

    print("Evaluation:")
    print(f"  {'Method':<35s} {'ARI':>6s} {'NMI':>6s} {'Pur.':>6s} {'Silh.':>6s}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for name, ev in all_results:
        sil_s = f"{ev['Silhouette']:.3f}" if np.isfinite(ev["Silhouette"]) else " N/A"
        print(f"  {name:<35s} {ev['ARI']:>6.3f} {ev['NMI']:>6.3f} {ev['Purity']:>6.3f} {sil_s:>6s}")
    print(f"\n  Best: {best_name}")
    print(f"  Word vs Phrase: {phrase_verdict}")
    print()

    # Best model cluster themes
    best_method_top = top_word if best_name == METHOD_NAMES[0] else (top_phrase if best_name == METHOD_NAMES[1] else top_agg)
    print("Best model clusters:")
    for cid in sorted(best_method_maj.keys()):
        terms_str = ", ".join(best_method_top.get(cid, [])[:5])
        print(f"  C{cid}: {best_method_maj[cid]:<25s} [{terms_str}]")
    print()

    print(f"Custom post: \"{custom_text}\"")
    print(f"  Word K-Means  : C{cust_word_cluster} -> {cust_word_theme} ({cust_word_sim:.3f})")
    print(f"  Phrase K-Means: C{cust_phrase_cluster} -> {cust_phrase_theme} ({cust_phrase_sim:.3f})")
    print(f"  Agglomerative : C{cust_agg_cluster} -> {cust_agg_theme} ({cust_agg_sim:.3f})")
    print(f"  Final theme   : {final_theme}")
    print(f"  Insight       : {final_insight}")
    print()
    print(f"Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
