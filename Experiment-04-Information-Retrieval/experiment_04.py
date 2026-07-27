"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 04
Title           : Build an Information Retrieval System Using Classical and
                  Nonclassical Models and Compare Their Performance on a Dataset
                  of Scientific Papers
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOP_K = 5
BM25_K1 = 1.5
BM25_B = 0.75
RANDOM_STATE = 42
MAX_LSA_COMPONENTS = 10

EXPERIMENT_TITLE = (
    "Build an Information Retrieval System Using Classical and Nonclassical "
    "Models and Compare Their Performance on a Dataset of Scientific Papers"
)

AIM = (
    "To implement an information retrieval system using TF-IDF and LSA "
    "techniques and retrieve relevant documents based on a user query."
)

STUDENT_NAME = "Divya M"
REGISTER_NUMBER = "24AD0074"

DEFAULT_QUERY = "Deep neural networks for detecting disease from medical images"

# ---------------------------------------------------------------------------
# Dataset: 40 scientific-paper abstracts (8 per topic)
# ---------------------------------------------------------------------------
DOCUMENTS: List[Dict[str, str]] = [
    # ── Machine Learning (D01–D08) ──────────────────────────────────────
    {
        "id": "D01",
        "title": "Gradient Boosting Ensembles for Tabular Prediction Tasks",
        "abstract": (
            "This study evaluates gradient boosting frameworks on heterogeneous "
            "tabular datasets. We compare XGBoost, LightGBM, and CatBoost using "
            "cross-validated accuracy and training throughput. Results demonstrate "
            "that careful hyperparameter tuning narrows performance gaps across "
            "all three frameworks."
        ),
        "topic": "machine_learning",
    },
    {
        "id": "D02",
        "title": "Transfer Learning with Pre-trained Convolutional Networks",
        "abstract": (
            "We investigate fine-tuning strategies for deep convolutional neural "
            "networks originally trained on ImageNet. Freezing early layers while "
            "adapting later blocks improves convergence on small target datasets. "
            "Our experiments on medical imaging benchmarks show significant gains "
            "over training from scratch."
        ),
        "topic": "machine_learning",
    },
    {
        "id": "D03",
        "title": "Semi-supervised Clustering with Graph Neural Networks",
        "abstract": (
            "Graph-based semi-supervised approaches leverage both labelled and "
            "unlabelled nodes to learn expressive representations. We propose a "
            "spectral attention mechanism that selectively aggregates neighbour "
            "features. The method achieves state-of-the-art results on citation "
            "network benchmarks."
        ),
        "topic": "machine_learning",
    },
    {
        "id": "D04",
        "title": "Bayesian Optimization for Neural Architecture Search",
        "abstract": (
            "Automated architecture search reduces reliance on expert design "
            "choices. We employ Gaussian process surrogate models within a "
            "Bayesian optimization loop to efficiently explore architecture "
            "spaces. The discovered configurations outperform manually designed "
            "baselines on image classification tasks."
        ),
        "topic": "machine_learning",
    },
    {
        "id": "D05",
        "title": "Federated Learning under Non-IID Data Distributions",
        "abstract": (
            "Federated training distributes model updates across edge devices "
            "without sharing raw data. We analyze convergence behaviour when "
            "client data is highly heterogeneous and propose a variance-reduced "
            "aggregation scheme. Privacy guarantees are maintained via "
            "differential privacy noise injection."
        ),
        "topic": "machine_learning",
    },
    {
        "id": "D06",
        "title": "Attention Mechanisms in Recurrent Sequence Models",
        "abstract": (
            "Incorporating attention into recurrent architectures enables the "
            "decoder to focus on salient encoder states. We benchmark several "
            "attention variants on machine translation and text summarization "
            "tasks. Multi-head self-attention achieves the highest BLEU and "
            "ROUGE scores across all tested language pairs."
        ),
        "topic": "machine_learning",
    },
    {
        "id": "D07",
        "title": "Explainability Methods for Random Forest Classifiers",
        "abstract": (
            "Interpreting ensemble predictions is critical in regulated domains. "
            "We compare SHAP values, permutation importance, and partial "
            "dependence plots for random forest models applied to credit scoring. "
            "SHAP provides the most consistent feature attributions aligned with "
            "domain expert judgements."
        ),
        "topic": "machine_learning",
    },
    {
        "id": "D08",
        "title": "Reinforcement Learning for Robotic Manipulation",
        "abstract": (
            "Policy gradient methods enable robots to learn dexterous grasping "
            "from simulated experience. We train a proximal policy optimization "
            "agent in a physics engine and transfer the policy to a physical "
            "arm. Sim-to-real transfer is enhanced by domain randomization "
            "techniques."
        ),
        "topic": "machine_learning",
    },
    # ── Climate Science (D09–D16) ───────────────────────────────────────
    {
        "id": "D09",
        "title": "Arctic Sea-Ice Decline and Polar Amplification Feedback",
        "abstract": (
            "Satellite observations confirm accelerating sea-ice loss in the "
            "Arctic over the past four decades. We quantify the albedo feedback "
            "loop and its contribution to polar amplification using reanalysis "
            "data and coupled climate model simulations."
        ),
        "topic": "climate_science",
    },
    {
        "id": "D10",
        "title": "Monsoon Variability Linked to Indian Ocean Dipole Events",
        "abstract": (
            "Interannual fluctuations in South Asian monsoon rainfall correlate "
            "with Indian Ocean Dipole phases. We use sea-surface temperature "
            "anomaly composites and moisture flux diagnostics to reveal the "
            "teleconnection pathways governing precipitation extremes."
        ),
        "topic": "climate_science",
    },
    {
        "id": "D11",
        "title": "Urban Heat Island Intensity in Tropical Megacities",
        "abstract": (
            "Rapid urbanization increases nocturnal temperatures in tropical "
            "cities. We deploy dense sensor networks across three megacities "
            "and correlate temperature differentials with impervious surface "
            "fraction and vegetation cover derived from Landsat imagery."
        ),
        "topic": "climate_science",
    },
    {
        "id": "D12",
        "title": "Carbon Budget Estimation Using Atmospheric Inversion",
        "abstract": (
            "Atmospheric CO2 inversion techniques constrain regional carbon "
            "fluxes by assimilating flask and satellite observations. We present "
            "an updated global carbon budget highlighting the growing terrestrial "
            "sink in the Northern Hemisphere boreal forests."
        ),
        "topic": "climate_science",
    },
    {
        "id": "D13",
        "title": "Extreme Precipitation Trends under RCP 8.5 Scenarios",
        "abstract": (
            "General circulation models project intensified rainfall extremes "
            "under high-emission pathways. We apply statistical downscaling to "
            "CMIP6 outputs and assess return period shifts for flood-critical "
            "river basins in Southeast Asia."
        ),
        "topic": "climate_science",
    },
    {
        "id": "D14",
        "title": "Permafrost Thaw and Methane Release in Siberian Tundra",
        "abstract": (
            "Warming permafrost releases stored organic carbon as methane, a "
            "potent greenhouse gas. We measure methane fluxes from thermokarst "
            "lakes using eddy covariance towers and estimate cumulative emissions "
            "through the end of the century."
        ),
        "topic": "climate_science",
    },
    {
        "id": "D15",
        "title": "Aerosol-Cloud Interactions and Radiative Forcing Uncertainty",
        "abstract": (
            "Anthropogenic aerosols modify cloud microphysics, complicating "
            "estimates of net radiative forcing. We combine aircraft in-situ "
            "measurements with satellite retrievals to reduce uncertainty in "
            "the aerosol indirect effect over the North Atlantic."
        ),
        "topic": "climate_science",
    },
    {
        "id": "D16",
        "title": "Glacier Mass Balance Monitoring with InSAR and GRACE",
        "abstract": (
            "Interferometric synthetic aperture radar and gravity recovery "
            "satellites provide complementary mass-loss estimates for mountain "
            "glaciers. We reconcile both datasets for Himalayan catchments and "
            "project freshwater availability impacts for downstream communities."
        ),
        "topic": "climate_science",
    },
    # ── Biomedical (D17–D24) ────────────────────────────────────────────
    {
        "id": "D17",
        "title": "CRISPR-Cas9 Off-Target Profiling in Human Cell Lines",
        "abstract": (
            "Genome editing with CRISPR-Cas9 carries risks of unintended "
            "cleavage at off-target loci. We perform whole-genome sequencing on "
            "edited HEK293 and iPSC lines and catalogue off-target mutations "
            "relative to guide RNA mismatch tolerance."
        ),
        "topic": "biomedical",
    },
    {
        "id": "D18",
        "title": "Single-Cell RNA Sequencing Reveals Tumour Heterogeneity",
        "abstract": (
            "Intra-tumour cellular diversity drives therapy resistance and "
            "relapse. We apply droplet-based single-cell transcriptomics to "
            "glioblastoma biopsies and identify distinct malignant sub-clones "
            "with differential sensitivity to temozolomide."
        ),
        "topic": "biomedical",
    },
    {
        "id": "D19",
        "title": "Gut Microbiome Composition and Inflammatory Bowel Disease",
        "abstract": (
            "Dysbiosis of intestinal microbiota is implicated in Crohn's disease "
            "and ulcerative colitis. We perform 16S rRNA amplicon sequencing on "
            "stool samples from a longitudinal cohort and link microbial diversity "
            "indices to clinical remission outcomes."
        ),
        "topic": "biomedical",
    },
    {
        "id": "D20",
        "title": "Drug Repurposing via Protein-Ligand Docking Simulations",
        "abstract": (
            "Virtual screening of approved drug libraries accelerates therapeutic "
            "discovery. We dock candidate compounds against the SARS-CoV-2 main "
            "protease and validate top hits with in-vitro enzymatic inhibition "
            "assays, identifying two promising repurposed agents."
        ),
        "topic": "biomedical",
    },
    {
        "id": "D21",
        "title": "Alzheimer Biomarker Detection from Cerebrospinal Fluid",
        "abstract": (
            "Early diagnosis of Alzheimer disease relies on sensitive biomarkers. "
            "We measure amyloid-beta and phosphorylated tau concentrations in "
            "cerebrospinal fluid using ultrasensitive immunoassays and correlate "
            "levels with cognitive decline trajectories over five years."
        ),
        "topic": "biomedical",
    },
    {
        "id": "D22",
        "title": "Nanomedicine Delivery Systems for Targeted Cancer Therapy",
        "abstract": (
            "Lipid nanoparticle carriers improve drug bioavailability and reduce "
            "systemic toxicity. We engineer folate-conjugated liposomes loaded "
            "with doxorubicin and evaluate tumour regression in murine xenograft "
            "models compared to free-drug administration."
        ),
        "topic": "biomedical",
    },
    {
        "id": "D23",
        "title": "Antibiotic Resistance Gene Surveillance in Wastewater",
        "abstract": (
            "Municipal wastewater serves as a reservoir for antimicrobial "
            "resistance genes. We perform metagenomic shotgun sequencing on "
            "influent samples from treatment plants and track the prevalence of "
            "carbapenem and colistin resistance determinants over eighteen months."
        ),
        "topic": "biomedical",
    },
    {
        "id": "D24",
        "title": "Wearable Biosensors for Continuous Glucose Monitoring",
        "abstract": (
            "Non-invasive glucose sensing enables real-time diabetes management. "
            "We develop a flexible electrochemical patch that measures interstitial "
            "glucose via microneedle arrays and validate accuracy against venous "
            "blood reference measurements in a clinical trial."
        ),
        "topic": "biomedical",
    },
    # ── Renewable Energy (D25–D32) ──────────────────────────────────────
    {
        "id": "D25",
        "title": "Perovskite-Silicon Tandem Solar Cell Efficiency Records",
        "abstract": (
            "Tandem photovoltaic architectures surpass single-junction efficiency "
            "limits. We fabricate monolithic perovskite-silicon cells achieving "
            "over 30 percent power conversion efficiency and examine long-term "
            "operational stability under accelerated ageing protocols."
        ),
        "topic": "renewable_energy",
    },
    {
        "id": "D26",
        "title": "Offshore Wind Farm Wake Modelling and Layout Optimization",
        "abstract": (
            "Turbine wake interactions reduce aggregate power output in wind "
            "farms. We couple large-eddy simulations with genetic algorithm "
            "optimization to determine turbine placements that minimize wake "
            "losses while respecting seabed and cable routing constraints."
        ),
        "topic": "renewable_energy",
    },
    {
        "id": "D27",
        "title": "Solid-State Lithium Batteries for Grid Energy Storage",
        "abstract": (
            "Replacing liquid electrolytes with ceramic solid electrolytes "
            "enhances battery safety and energy density. We characterize ionic "
            "conductivity in garnet-type Li7La3Zr2O12 pellets and demonstrate "
            "stable cycling performance at room temperature."
        ),
        "topic": "renewable_energy",
    },
    {
        "id": "D28",
        "title": "Techno-economic Analysis of Green Hydrogen Production",
        "abstract": (
            "Electrolysis powered by surplus renewable electricity produces "
            "green hydrogen as a versatile energy carrier. We model levelized "
            "hydrogen costs for proton-exchange membrane and alkaline electrolyser "
            "configurations at varying capacity factors."
        ),
        "topic": "renewable_energy",
    },
    {
        "id": "D29",
        "title": "Tidal Stream Turbine Performance in High-Flow Channels",
        "abstract": (
            "Tidal energy harvesting exploits predictable marine currents. We "
            "deploy horizontal-axis tidal turbines in a strait and measure power "
            "curves across spring-neap tidal cycles, comparing field data with "
            "blade-element momentum theory predictions."
        ),
        "topic": "renewable_energy",
    },
    {
        "id": "D30",
        "title": "Agrivoltaic Systems: Crop Yield under Elevated Solar Panels",
        "abstract": (
            "Co-locating photovoltaic arrays with agricultural land addresses "
            "competing demands for space. We assess lettuce and wheat yields "
            "beneath bifacial panel rows and quantify the dual benefit of energy "
            "generation and reduced irrigation demand."
        ),
        "topic": "renewable_energy",
    },
    {
        "id": "D31",
        "title": "Concentrated Solar Power with Molten Salt Thermal Storage",
        "abstract": (
            "Parabolic trough plants store excess thermal energy in molten "
            "nitrate salts for dispatchable electricity generation. We simulate "
            "annual plant performance across arid climate zones and evaluate the "
            "cost-effectiveness of extended storage durations."
        ),
        "topic": "renewable_energy",
    },
    {
        "id": "D32",
        "title": "Microgrid Control Strategies for Rural Electrification",
        "abstract": (
            "Hybrid microgrids combining solar panels, small wind turbines, and "
            "battery banks electrify remote villages. We design a predictive "
            "energy management controller that balances load demand, renewable "
            "variability, and battery state of charge to minimize diesel backup."
        ),
        "topic": "renewable_energy",
    },
    # ── Astronomy (D33–D40) ─────────────────────────────────────────────
    {
        "id": "D33",
        "title": "Exoplanet Atmosphere Characterization with JWST Transit Spectra",
        "abstract": (
            "The James Webb Space Telescope enables high-resolution transmission "
            "spectroscopy of exoplanet atmospheres. We detect water vapour, "
            "carbon dioxide, and methane absorption features in the spectrum of "
            "a warm Neptune orbiting a nearby M-dwarf star."
        ),
        "topic": "astronomy",
    },
    {
        "id": "D34",
        "title": "Fast Radio Burst Localization and Host Galaxy Identification",
        "abstract": (
            "Millisecond-duration radio transients originate from cosmological "
            "distances. We use interferometric baselines to sub-arcsecond "
            "localize repeating fast radio bursts and identify their host "
            "galaxies, constraining progenitor models through environment "
            "demographics."
        ),
        "topic": "astronomy",
    },
    {
        "id": "D35",
        "title": "Gravitational Wave Parameter Estimation for Binary Mergers",
        "abstract": (
            "LIGO and Virgo detections of compact binary coalescences provide "
            "constraints on neutron star equations of state. We apply nested "
            "sampling algorithms to infer component masses, spins, and tidal "
            "deformabilities from observed gravitational wave strain data."
        ),
        "topic": "astronomy",
    },
    {
        "id": "D36",
        "title": "Dark Matter Halo Profiles from Weak Gravitational Lensing",
        "abstract": (
            "Weak lensing shear catalogues allow statistical measurement of dark "
            "matter distribution around galaxy clusters. We stack lensing signals "
            "from thousands of clusters and fit Navarro-Frenk-White profiles to "
            "constrain halo mass-concentration relations."
        ),
        "topic": "astronomy",
    },
    {
        "id": "D37",
        "title": "Star Formation Rates in High-Redshift Submillimetre Galaxies",
        "abstract": (
            "Dusty star-forming galaxies at redshift z > 2 exhibit prodigious "
            "infrared luminosities. We combine ALMA continuum observations with "
            "spectral energy distribution fitting to derive star formation rates "
            "and dust masses in a flux-limited sample."
        ),
        "topic": "astronomy",
    },
    {
        "id": "D38",
        "title": "Supernova Remnant Shocks and Cosmic Ray Acceleration",
        "abstract": (
            "Diffusive shock acceleration in supernova remnants is the primary "
            "mechanism for galactic cosmic ray production. We map synchrotron "
            "X-ray filaments in Cassiopeia A and model maximum particle energies "
            "attainable at the forward shock."
        ),
        "topic": "astronomy",
    },
    {
        "id": "D39",
        "title": "Pulsar Timing Arrays and Nanohertz Gravitational Waves",
        "abstract": (
            "Millisecond pulsars serve as precise clocks for detecting low-"
            "frequency gravitational wave backgrounds. We analyse a fifteen-year "
            "timing dataset from an international pulsar timing array and report "
            "evidence for a stochastic signal consistent with supermassive black "
            "hole binary inspirals."
        ),
        "topic": "astronomy",
    },
    {
        "id": "D40",
        "title": "Asteroid Spectral Taxonomy and Near-Earth Object Hazard",
        "abstract": (
            "Spectroscopic classification of asteroids informs composition and "
            "impact hazard assessment. We obtain visible and near-infrared "
            "reflectance spectra of newly discovered near-Earth objects and "
            "assign taxonomic types to prioritize planetary defence missions."
        ),
        "topic": "astronomy",
    },
]

# ---------------------------------------------------------------------------
# Evaluation queries (2 per topic, 10 total)
# ---------------------------------------------------------------------------
EVAL_QUERIES: List[Dict[str, str]] = [
    # Machine Learning
    {
        "query": "deep learning architectures for image recognition",
        "relevant_topic": "machine_learning",
    },
    {
        "query": "interpretable ensemble methods for prediction",
        "relevant_topic": "machine_learning",
    },
    # Climate Science
    {
        "query": "global warming effects on polar ice and permafrost",
        "relevant_topic": "climate_science",
    },
    {
        "query": "rainfall patterns and extreme weather projections",
        "relevant_topic": "climate_science",
    },
    # Biomedical
    {
        "query": "gene editing techniques and genomic mutations",
        "relevant_topic": "biomedical",
    },
    {
        "query": "drug discovery and therapeutic molecular screening",
        "relevant_topic": "biomedical",
    },
    # Renewable Energy
    {
        "query": "solar photovoltaic cell efficiency improvements",
        "relevant_topic": "renewable_energy",
    },
    {
        "query": "wind and tidal power generation optimization",
        "relevant_topic": "renewable_energy",
    },
    # Astronomy
    {
        "query": "detection of gravitational waves from compact objects",
        "relevant_topic": "astronomy",
    },
    {
        "query": "exoplanet atmospheres and stellar spectroscopy",
        "relevant_topic": "astronomy",
    },
]

# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------

def ensure_nltk_resources() -> None:
    """Download any missing NLTK resources."""
    import nltk  # noqa: delayed import

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
                raise RuntimeError(
                    f"Failed to download NLTK resource '{name}': {exc}"
                ) from exc


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

def preprocess(text: str) -> List[str]:
    """Lowercase, tokenize, remove non-alphabetic tokens and stopwords, lemmatize."""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text.lower())
    return [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in stop_words
    ]


# ---------------------------------------------------------------------------
# Retrieval Models
# ---------------------------------------------------------------------------

def boolean_retrieval(
    query_tokens: List[str],
    doc_token_lists: List[List[str]],
    doc_ids: List[str],
) -> List[Tuple[str, float]]:
    """Boolean term-matching retrieval with match-count scoring."""
    query_set = set(query_tokens)
    results: List[Tuple[str, float]] = []

    for idx, dtokens in enumerate(doc_token_lists):
        doc_set = set(dtokens)
        matches = query_set & doc_set
        if not matches:
            continue
        score = len(matches)
        if query_set <= doc_set:
            score += len(query_set)  # bonus for full match
        results.append((doc_ids[idx], float(score)))

    # Sort descending by score, then ascending by doc ID for determinism
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


def tfidf_retrieval(
    query_text: str,
    processed_corpus: List[str],
) -> List[Tuple[int, float]]:
    """TF-IDF cosine similarity retrieval. Returns (doc_index, score) pairs."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(processed_corpus)
    query_vec = vectorizer.transform([query_text])
    scores = cosine_similarity(query_vec, doc_matrix).flatten()

    ranked = sorted(
        enumerate(scores), key=lambda x: (-x[1], x[0])
    )
    return ranked


def bm25_retrieval(
    query_tokens: List[str],
    doc_token_lists: List[List[str]],
) -> List[Tuple[int, float]]:
    """BM25 probabilistic retrieval using NumPy."""
    n_docs = len(doc_token_lists)
    doc_lens = np.array([len(d) for d in doc_token_lists], dtype=np.float64)
    avgdl = doc_lens.mean()

    # Build vocabulary and document frequency
    vocab: Dict[str, int] = {}
    df: Dict[str, int] = {}
    for dtokens in doc_token_lists:
        seen = set()
        for tok in dtokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            if tok not in seen:
                df[tok] = df.get(tok, 0) + 1
                seen.add(tok)

    # IDF
    idf: Dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    # Term frequencies per document
    tf_matrix: List[Dict[str, int]] = []
    for dtokens in doc_token_lists:
        tf: Dict[str, int] = {}
        for tok in dtokens:
            tf[tok] = tf.get(tok, 0) + 1
        tf_matrix.append(tf)

    # Score each document
    scores = np.zeros(n_docs, dtype=np.float64)
    for term in query_tokens:
        if term not in idf:
            continue
        term_idf = idf[term]
        for i in range(n_docs):
            tf_val = tf_matrix[i].get(term, 0)
            if tf_val == 0:
                continue
            numerator = tf_val * (BM25_K1 + 1)
            denominator = tf_val + BM25_K1 * (
                1 - BM25_B + BM25_B * doc_lens[i] / avgdl
            )
            scores[i] += term_idf * numerator / denominator

    ranked = sorted(enumerate(scores.tolist()), key=lambda x: (-x[1], x[0]))
    return ranked


def lsa_retrieval(
    query_text: str,
    processed_corpus: List[str],
) -> List[Tuple[int, float]]:
    """LSA retrieval via TruncatedSVD on TF-IDF matrix."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(processed_corpus)

    n_docs, n_features = doc_matrix.shape
    n_components = min(MAX_LSA_COMPONENTS, n_docs - 1, n_features - 1)

    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    doc_lsa = svd.fit_transform(doc_matrix)

    query_vec = vectorizer.transform([query_text])
    query_lsa = svd.transform(query_vec)

    scores = cosine_similarity(query_lsa, doc_lsa).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))
    return ranked


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Precision at k."""
    top = retrieved_ids[:k]
    return sum(1 for d in top if d in relevant_ids) / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Recall at k."""
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    return sum(1 for d in top if d in relevant_ids) / len(relevant_ids)


def average_precision_at_k(retrieved_ids: List[str], relevant_ids: set, k: int) -> float:
    """Average Precision at k."""
    top = retrieved_ids[:k]
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(top, 1):
        if doc_id in relevant_ids:
            hits += 1
            sum_prec += hits / i
    if hits == 0:
        return 0.0
    return sum_prec / min(len(relevant_ids), k)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: set) -> float:
    """Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_relevant_ids(topic: str) -> set:
    """Return set of document IDs belonging to a topic."""
    return {d["id"] for d in DOCUMENTS if d["topic"] == topic}


def build_ranked_ids_for_model(
    method_name: str,
    query_tokens: List[str],
    query_processed_text: str,
    doc_token_lists: List[List[str]],
    doc_ids: List[str],
    processed_corpus: List[str],
) -> List[str]:
    """Run one retrieval model and return ranked document IDs."""
    if method_name == "Boolean":
        pairs = boolean_retrieval(query_tokens, doc_token_lists, doc_ids)
        return [p[0] for p in pairs]
    elif method_name == "TF-IDF":
        pairs = tfidf_retrieval(query_processed_text, processed_corpus)
        return [doc_ids[i] for i, _ in pairs]
    elif method_name == "BM25":
        pairs = bm25_retrieval(query_tokens, doc_token_lists)
        return [doc_ids[i] for i, _ in pairs]
    elif method_name == "LSA":
        pairs = lsa_retrieval(query_processed_text, processed_corpus)
        return [doc_ids[i] for i, _ in pairs]
    return []


def get_scores_for_model(
    method_name: str,
    query_tokens: List[str],
    query_processed_text: str,
    doc_token_lists: List[List[str]],
    doc_ids: List[str],
    processed_corpus: List[str],
) -> Dict[str, float]:
    """Run one retrieval model and return {doc_id: score}."""
    if method_name == "Boolean":
        pairs = boolean_retrieval(query_tokens, doc_token_lists, doc_ids)
        return {did: sc for did, sc in pairs}
    elif method_name == "TF-IDF":
        pairs = tfidf_retrieval(query_processed_text, processed_corpus)
        return {doc_ids[i]: sc for i, sc in pairs}
    elif method_name == "BM25":
        pairs = bm25_retrieval(query_tokens, doc_token_lists)
        return {doc_ids[i]: sc for i, sc in pairs}
    elif method_name == "LSA":
        pairs = lsa_retrieval(query_processed_text, processed_corpus)
        return {doc_ids[i]: sc for i, sc in pairs}
    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 04 – Information Retrieval System"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Custom scientific query for retrieval demonstration.",
    )
    args = parser.parse_args()

    # --- NLTK setup ---
    ensure_nltk_resources()

    # --- Prepare data ---
    doc_ids = [d["id"] for d in DOCUMENTS]
    doc_texts = [d["title"] + " " + d["abstract"] for d in DOCUMENTS]
    doc_id_to_idx = {d["id"]: i for i, d in enumerate(DOCUMENTS)}

    doc_token_lists = [preprocess(t) for t in doc_texts]
    processed_corpus = [" ".join(tokens) for tokens in doc_token_lists]

    method_names = ["Boolean", "TF-IDF", "BM25", "LSA"]

    # --- Evaluation ---
    method_metrics: Dict[str, Dict[str, List[float]]] = {
        m: {"p": [], "r": [], "ap": [], "rr": []}
        for m in method_names
    }

    for eq in EVAL_QUERIES:
        q_tokens = preprocess(eq["query"])
        q_text = " ".join(q_tokens)
        relevant = get_relevant_ids(eq["relevant_topic"])

        for m in method_names:
            ranked = build_ranked_ids_for_model(
                m, q_tokens, q_text, doc_token_lists, doc_ids, processed_corpus
            )
            method_metrics[m]["p"].append(precision_at_k(ranked, relevant, TOP_K))
            method_metrics[m]["r"].append(recall_at_k(ranked, relevant, TOP_K))
            method_metrics[m]["ap"].append(average_precision_at_k(ranked, relevant, TOP_K))
            method_metrics[m]["rr"].append(reciprocal_rank(ranked, relevant))

    # Compute means
    eval_table: List[Tuple[str, float, float, float, float]] = []
    for m in method_names:
        mp = float(np.mean(method_metrics[m]["p"]))
        mr = float(np.mean(method_metrics[m]["r"]))
        ma = float(np.mean(method_metrics[m]["ap"]))
        mrr = float(np.mean(method_metrics[m]["rr"]))
        eval_table.append((m, mp, mr, ma, mrr))

    # Select best method
    eval_table_sorted = sorted(eval_table, key=lambda x: (-x[3], -x[4], -x[1]))
    best_method = eval_table_sorted[0][0]

    # --- Custom query ---
    custom_query = args.query
    cq_tokens = preprocess(custom_query)
    cq_text = " ".join(cq_tokens)

    custom_results: Dict[str, List[Tuple[str, float]]] = {}
    for m in method_names:
        scores = get_scores_for_model(
            m, cq_tokens, cq_text, doc_token_lists, doc_ids, processed_corpus
        )
        ranked = build_ranked_ids_for_model(
            m, cq_tokens, cq_text, doc_token_lists, doc_ids, processed_corpus
        )
        custom_results[m] = [(did, scores.get(did, 0.0)) for did in ranked[:TOP_K]]

    # --- Build output report ---
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.txt"

    lines: List[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("=" * 72)
    add("EXPERIMENT 04 – INFORMATION RETRIEVAL SYSTEM")
    add("=" * 72)
    add()
    add(f"Student Name    : {STUDENT_NAME}")
    add(f"Register Number : {REGISTER_NUMBER}")
    add()
    add("Title:")
    add(EXPERIMENT_TITLE)
    add()
    add("Aim:")
    add(AIM)
    add()
    add("-" * 72)
    add("DATASET SUMMARY")
    add("-" * 72)
    add(f"Total documents : {len(DOCUMENTS)}")
    topics = sorted(set(d["topic"] for d in DOCUMENTS))
    for t in topics:
        count = sum(1 for d in DOCUMENTS if d["topic"] == t)
        add(f"  {t:20s}: {count} documents")
    add()

    add("-" * 72)
    add("RETRIEVAL MODELS")
    add("-" * 72)
    add("A. Boolean Term-Matching Retrieval")
    add("   Scores documents by the count of matching query terms.")
    add("   Awards a bonus when all query terms appear in the document.")
    add()
    add("B. TF-IDF Vector-Space Retrieval")
    add("   Fits a TF-IDF vectorizer on the corpus, transforms queries,")
    add("   and ranks documents by cosine similarity.")
    add()
    add("C. BM25 Probabilistic Retrieval")
    add(f"   Okapi BM25 with k1={BM25_K1}, b={BM25_B}.")
    add("   Computes term-level IDF and length-normalized term frequency.")
    add()
    add("D. Latent Semantic Analysis (LSA)")
    add("   Applies TruncatedSVD to the TF-IDF matrix and ranks by")
    add("   cosine similarity in the reduced latent space.")
    add()

    add("-" * 72)
    add("EVALUATION")
    add("-" * 72)
    add(f"Evaluation queries: {len(EVAL_QUERIES)}")
    add(f"Top-K             : {TOP_K}")
    add()
    header = f"{'Method':<12} {'Prec@5':>8} {'Rec@5':>8} {'MAP@5':>8} {'MRR':>8}"
    add(header)
    add("-" * len(header))
    for m, mp, mr, ma, mrr in eval_table:
        add(f"{m:<12} {mp:>8.4f} {mr:>8.4f} {ma:>8.4f} {mrr:>8.4f}")
    add()
    add(f"Best retrieval model: {best_method}")
    add(f"  (Selected by highest MAP@5, then MRR, then Mean Precision@5)")
    add()

    add("-" * 72)
    add("CUSTOM QUERY RESULTS")
    add("-" * 72)
    add(f"Query: {custom_query}")
    add(f"Processed tokens: {cq_tokens}")
    add()
    for m in method_names:
        add(f"  [{m}] Top-{TOP_K} Results:")
        for rank, (did, score) in enumerate(custom_results[m], 1):
            doc = DOCUMENTS[doc_id_to_idx[did]]
            add(
                f"    {rank}. {did} | {doc['title']}"
            )
            add(
                f"       Topic: {doc['topic']}  |  Score: {score:.4f}"
            )
        add()

    add("-" * 72)
    add("RESULT")
    add("-" * 72)
    add(
        "An information retrieval system was implemented using Boolean, TF-IDF, "
        "BM25, and LSA models. The models were evaluated on 10 queries across "
        "5 scientific topics. "
        f"{best_method} achieved the highest MAP@5 and was selected as the "
        "best retrieval model."
    )
    add("=" * 72)

    report = "\n".join(lines) + "\n"
    output_path.write_text(report, encoding="utf-8")

    # --- Terminal summary ---
    print("=" * 60)
    print("Experiment 04 – Information Retrieval System")
    print("=" * 60)
    print(f"Student  : {STUDENT_NAME} ({REGISTER_NUMBER})")
    print(f"Dataset  : {len(DOCUMENTS)} documents, {len(topics)} topics")
    print()
    print(header)
    print("-" * len(header))
    for m, mp, mr, ma, mrr in eval_table:
        print(f"{m:<12} {mp:>8.4f} {mr:>8.4f} {ma:>8.4f} {mrr:>8.4f}")
    print()
    print(f"Best model: {best_method}")
    print()
    print(f"Custom query: {custom_query}")
    print(f"Processed tokens: {cq_tokens}")
    print()
    print(f"[{best_method}] Top-{TOP_K} Results:")
    for rank, (did, score) in enumerate(custom_results[best_method], 1):
        doc = DOCUMENTS[doc_id_to_idx[did]]
        print(f"  {rank}. {did} | {doc['title']}")
        print(f"     Topic: {doc['topic']}  |  Score: {score:.4f}")
    print()
    print(f"Output saved: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
