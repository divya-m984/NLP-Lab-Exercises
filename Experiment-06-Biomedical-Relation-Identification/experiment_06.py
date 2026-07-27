"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 06
Title           : Investigate Different Approaches for Relation Identification in
                  Biomedical Texts and Evaluate Their Precision and Recall
"""

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SPLIT_RATIO = 0.8

RELATION_LABELS = ["TREATS", "CAUSES", "INHIBITS", "ASSOCIATED_WITH", "NO_RELATION"]

EXPERIMENT_TITLE = (
    "Investigate Different Approaches for Relation Identification in "
    "Biomedical Texts and Evaluate Their Precision and Recall"
)

AIM = (
    "To identify relations between biomedical entities using a rule-based "
    "approach and evaluate the performance using Precision, Recall, and "
    "F1-Score."
)

DEFAULT_TEXT = "Metformin controls blood glucose in patients with type 2 diabetes."
DEFAULT_ENTITY_1 = "Metformin"
DEFAULT_ENTITY_2 = "type 2 diabetes"

# ---------------------------------------------------------------------------
# Embedded biomedical dataset – 75 sentences, 15 per relation
# ---------------------------------------------------------------------------
DATASET: List[Dict[str, str]] = [
    # ── TREATS (15) ──────────────────────────────────────────────────────
    {"sentence": "Amoxicillin treats bacterial pneumonia by inhibiting cell wall synthesis.",
     "entity_1": "Amoxicillin", "entity_2": "bacterial pneumonia", "relation": "TREATS"},
    {"sentence": "Lisinopril is prescribed to manage chronic hypertension in elderly patients.",
     "entity_1": "Lisinopril", "entity_2": "chronic hypertension", "relation": "TREATS"},
    {"sentence": "Methotrexate reduces joint inflammation in patients with rheumatoid arthritis.",
     "entity_1": "Methotrexate", "entity_2": "rheumatoid arthritis", "relation": "TREATS"},
    {"sentence": "Insulin therapy improves glycaemic control in type 1 diabetes mellitus.",
     "entity_1": "Insulin therapy", "entity_2": "type 1 diabetes mellitus", "relation": "TREATS"},
    {"sentence": "Salbutamol relieves acute bronchospasm during asthma exacerbations.",
     "entity_1": "Salbutamol", "entity_2": "bronchospasm", "relation": "TREATS"},
    {"sentence": "Omeprazole controls gastric acid secretion in gastroesophageal reflux disease.",
     "entity_1": "Omeprazole", "entity_2": "gastroesophageal reflux disease", "relation": "TREATS"},
    {"sentence": "Doxycycline is a therapeutic agent for Lyme disease caused by Borrelia.",
     "entity_1": "Doxycycline", "entity_2": "Lyme disease", "relation": "TREATS"},
    {"sentence": "Levothyroxine manages symptoms of hypothyroidism effectively.",
     "entity_1": "Levothyroxine", "entity_2": "hypothyroidism", "relation": "TREATS"},
    {"sentence": "Fluoxetine improves mood and cognitive function in major depressive disorder.",
     "entity_1": "Fluoxetine", "entity_2": "major depressive disorder", "relation": "TREATS"},
    {"sentence": "Atorvastatin reduces low-density lipoprotein levels in hypercholesterolemia.",
     "entity_1": "Atorvastatin", "entity_2": "hypercholesterolemia", "relation": "TREATS"},
    {"sentence": "Oseltamivir is used to treat influenza A and B infections.",
     "entity_1": "Oseltamivir", "entity_2": "influenza A", "relation": "TREATS"},
    {"sentence": "Ciprofloxacin provides effective therapy for urinary tract infections.",
     "entity_1": "Ciprofloxacin", "entity_2": "urinary tract infections", "relation": "TREATS"},
    {"sentence": "Montelukast controls chronic asthma symptoms through leukotriene receptor antagonism.",
     "entity_1": "Montelukast", "entity_2": "chronic asthma", "relation": "TREATS"},
    {"sentence": "Acyclovir reduces viral load and treats herpes simplex encephalitis.",
     "entity_1": "Acyclovir", "entity_2": "herpes simplex encephalitis", "relation": "TREATS"},
    {"sentence": "Ibuprofen relieves pain and fever in patients with mild osteoarthritis.",
     "entity_1": "Ibuprofen", "entity_2": "osteoarthritis", "relation": "TREATS"},

    # ── CAUSES (15) ──────────────────────────────────────────────────────
    {"sentence": "Prolonged corticosteroid use causes adrenal insufficiency in some patients.",
     "entity_1": "corticosteroid", "entity_2": "adrenal insufficiency", "relation": "CAUSES"},
    {"sentence": "Cigarette smoking induces chronic obstructive pulmonary disease over decades.",
     "entity_1": "Cigarette smoking", "entity_2": "chronic obstructive pulmonary disease", "relation": "CAUSES"},
    {"sentence": "Helicobacter pylori infection triggers peptic ulcer formation in the gastric mucosa.",
     "entity_1": "Helicobacter pylori", "entity_2": "peptic ulcer", "relation": "CAUSES"},
    {"sentence": "Excessive alcohol consumption leads to hepatic cirrhosis.",
     "entity_1": "alcohol consumption", "entity_2": "hepatic cirrhosis", "relation": "CAUSES"},
    {"sentence": "Ionizing radiation produces DNA double-strand breaks in lymphocytes.",
     "entity_1": "Ionizing radiation", "entity_2": "DNA double-strand breaks", "relation": "CAUSES"},
    {"sentence": "Asbestos exposure results in malignant mesothelioma after prolonged contact.",
     "entity_1": "Asbestos exposure", "entity_2": "malignant mesothelioma", "relation": "CAUSES"},
    {"sentence": "Staphylococcus aureus causes skin and soft tissue infections in immunocompromised hosts.",
     "entity_1": "Staphylococcus aureus", "entity_2": "soft tissue infections", "relation": "CAUSES"},
    {"sentence": "High dietary sodium intake induces elevated blood pressure in susceptible individuals.",
     "entity_1": "dietary sodium", "entity_2": "elevated blood pressure", "relation": "CAUSES"},
    {"sentence": "Mutations in the BRCA1 gene lead to increased breast cancer susceptibility.",
     "entity_1": "BRCA1 gene", "entity_2": "breast cancer", "relation": "CAUSES"},
    {"sentence": "Chronic hyperglycaemia produces peripheral neuropathy in diabetic patients.",
     "entity_1": "hyperglycaemia", "entity_2": "peripheral neuropathy", "relation": "CAUSES"},
    {"sentence": "Mycobacterium tuberculosis triggers granuloma formation in lung tissue.",
     "entity_1": "Mycobacterium tuberculosis", "entity_2": "granuloma formation", "relation": "CAUSES"},
    {"sentence": "Lead poisoning induces cognitive impairment in young children.",
     "entity_1": "Lead poisoning", "entity_2": "cognitive impairment", "relation": "CAUSES"},
    {"sentence": "Streptococcus pyogenes causes rheumatic fever following pharyngeal infection.",
     "entity_1": "Streptococcus pyogenes", "entity_2": "rheumatic fever", "relation": "CAUSES"},
    {"sentence": "Uncontrolled diabetes results in diabetic retinopathy affecting the retinal vasculature.",
     "entity_1": "Uncontrolled diabetes", "entity_2": "diabetic retinopathy", "relation": "CAUSES"},
    {"sentence": "Chronic stress triggers cortisol dysregulation in the hypothalamic-pituitary axis.",
     "entity_1": "Chronic stress", "entity_2": "cortisol dysregulation", "relation": "CAUSES"},

    # ── INHIBITS (15) ────────────────────────────────────────────────────
    {"sentence": "Aspirin inhibits cyclooxygenase-2 enzyme activity in inflamed tissue.",
     "entity_1": "Aspirin", "entity_2": "cyclooxygenase-2", "relation": "INHIBITS"},
    {"sentence": "Tamoxifen blocks estrogen receptor signalling in breast carcinoma cells.",
     "entity_1": "Tamoxifen", "entity_2": "estrogen receptor", "relation": "INHIBITS"},
    {"sentence": "Metformin suppresses hepatic gluconeogenesis through AMPK activation.",
     "entity_1": "Metformin", "entity_2": "hepatic gluconeogenesis", "relation": "INHIBITS"},
    {"sentence": "Imatinib prevents BCR-ABL tyrosine kinase phosphorylation in leukaemia cells.",
     "entity_1": "Imatinib", "entity_2": "BCR-ABL tyrosine kinase", "relation": "INHIBITS"},
    {"sentence": "Warfarin suppresses vitamin K-dependent clotting factor synthesis in the liver.",
     "entity_1": "Warfarin", "entity_2": "clotting factor synthesis", "relation": "INHIBITS"},
    {"sentence": "Allopurinol inhibits xanthine oxidase and reduces uric acid production.",
     "entity_1": "Allopurinol", "entity_2": "xanthine oxidase", "relation": "INHIBITS"},
    {"sentence": "Sirolimus blocks mTOR signalling and reduces T-cell proliferation.",
     "entity_1": "Sirolimus", "entity_2": "mTOR signalling", "relation": "INHIBITS"},
    {"sentence": "Captopril inhibits angiotensin-converting enzyme in the renin-angiotensin system.",
     "entity_1": "Captopril", "entity_2": "angiotensin-converting enzyme", "relation": "INHIBITS"},
    {"sentence": "Finasteride suppresses 5-alpha reductase activity in prostatic tissue.",
     "entity_1": "Finasteride", "entity_2": "5-alpha reductase", "relation": "INHIBITS"},
    {"sentence": "Clopidogrel blocks the P2Y12 receptor on platelet membranes.",
     "entity_1": "Clopidogrel", "entity_2": "P2Y12 receptor", "relation": "INHIBITS"},
    {"sentence": "Erlotinib inhibits epidermal growth factor receptor in non-small-cell lung carcinoma.",
     "entity_1": "Erlotinib", "entity_2": "epidermal growth factor receptor", "relation": "INHIBITS"},
    {"sentence": "Statins reduce HMG-CoA reductase activity and downregulate cholesterol biosynthesis.",
     "entity_1": "Statins", "entity_2": "HMG-CoA reductase", "relation": "INHIBITS"},
    {"sentence": "Omalizumab prevents IgE from binding to mast cell receptors.",
     "entity_1": "Omalizumab", "entity_2": "IgE", "relation": "INHIBITS"},
    {"sentence": "Acarbose inhibits alpha-glucosidase in the small intestinal brush border.",
     "entity_1": "Acarbose", "entity_2": "alpha-glucosidase", "relation": "INHIBITS"},
    {"sentence": "Bevacizumab suppresses vascular endothelial growth factor in tumour angiogenesis.",
     "entity_1": "Bevacizumab", "entity_2": "vascular endothelial growth factor", "relation": "INHIBITS"},

    # ── ASSOCIATED_WITH (15) ─────────────────────────────────────────────
    {"sentence": "Obesity is associated with an increased risk of cardiovascular disease.",
     "entity_1": "Obesity", "entity_2": "cardiovascular disease", "relation": "ASSOCIATED_WITH"},
    {"sentence": "The HLA-B27 antigen is linked to ankylosing spondylitis susceptibility.",
     "entity_1": "HLA-B27", "entity_2": "ankylosing spondylitis", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Elevated C-reactive protein levels are correlated with systemic inflammation.",
     "entity_1": "C-reactive protein", "entity_2": "systemic inflammation", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Vitamin D deficiency is related to an increased incidence of osteoporosis.",
     "entity_1": "Vitamin D deficiency", "entity_2": "osteoporosis", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Sedentary lifestyle is a risk factor for metabolic syndrome.",
     "entity_1": "Sedentary lifestyle", "entity_2": "metabolic syndrome", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Low serum folate is connected to elevated homocysteine levels.",
     "entity_1": "serum folate", "entity_2": "homocysteine", "relation": "ASSOCIATED_WITH"},
    {"sentence": "APOE-e4 allele is associated with late-onset Alzheimer disease.",
     "entity_1": "APOE-e4 allele", "entity_2": "Alzheimer disease", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Chronic sleep deprivation is linked to impaired glucose tolerance.",
     "entity_1": "sleep deprivation", "entity_2": "glucose tolerance", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Gut microbiome dysbiosis is correlated with irritable bowel syndrome severity.",
     "entity_1": "microbiome dysbiosis", "entity_2": "irritable bowel syndrome", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Air pollution exposure is related to increased respiratory morbidity.",
     "entity_1": "Air pollution", "entity_2": "respiratory morbidity", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Elevated troponin I is associated with acute myocardial infarction.",
     "entity_1": "troponin I", "entity_2": "acute myocardial infarction", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Iron deficiency is linked to restless leg syndrome in clinical studies.",
     "entity_1": "Iron deficiency", "entity_2": "restless leg syndrome", "relation": "ASSOCIATED_WITH"},
    {"sentence": "High body mass index is a known risk factor for type 2 diabetes mellitus.",
     "entity_1": "body mass index", "entity_2": "type 2 diabetes mellitus", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Polymorphisms in the TNF-alpha gene are connected to rheumatoid arthritis risk.",
     "entity_1": "TNF-alpha gene", "entity_2": "rheumatoid arthritis", "relation": "ASSOCIATED_WITH"},
    {"sentence": "Elevated LDL cholesterol is correlated with atherosclerotic plaque progression.",
     "entity_1": "LDL cholesterol", "entity_2": "atherosclerotic plaque", "relation": "ASSOCIATED_WITH"},

    # ── NO_RELATION (15) ─────────────────────────────────────────────────
    {"sentence": "Aspirin and melatonin were both stored in the pharmacy refrigerator.",
     "entity_1": "Aspirin", "entity_2": "melatonin", "relation": "NO_RELATION"},
    {"sentence": "The platelet count was measured before the glucose tolerance test was performed.",
     "entity_1": "platelet count", "entity_2": "glucose tolerance test", "relation": "NO_RELATION"},
    {"sentence": "Penicillin was discovered in 1928 while hepatitis B vaccine became available in 1981.",
     "entity_1": "Penicillin", "entity_2": "hepatitis B vaccine", "relation": "NO_RELATION"},
    {"sentence": "Hemoglobin levels were recorded alongside serum creatinine in the laboratory report.",
     "entity_1": "Hemoglobin", "entity_2": "serum creatinine", "relation": "NO_RELATION"},
    {"sentence": "The patient received paracetamol but the diagnosis was unrelated to migraine.",
     "entity_1": "paracetamol", "entity_2": "migraine", "relation": "NO_RELATION"},
    {"sentence": "MRI scans showed no connection between the spleen enlargement and calcium levels.",
     "entity_1": "spleen enlargement", "entity_2": "calcium levels", "relation": "NO_RELATION"},
    {"sentence": "Chlorhexidine was used as an antiseptic while the biopsy targeted the thyroid gland.",
     "entity_1": "Chlorhexidine", "entity_2": "thyroid gland", "relation": "NO_RELATION"},
    {"sentence": "The study measured albumin concentration independently of fibrinogen levels.",
     "entity_1": "albumin", "entity_2": "fibrinogen", "relation": "NO_RELATION"},
    {"sentence": "Both erythromycin and calcium carbonate were listed in the formulary without interaction data.",
     "entity_1": "erythromycin", "entity_2": "calcium carbonate", "relation": "NO_RELATION"},
    {"sentence": "Serotonin receptors were examined in the same tissue sample as collagen fibres.",
     "entity_1": "Serotonin receptors", "entity_2": "collagen fibres", "relation": "NO_RELATION"},
    {"sentence": "The clinical trial enrolled participants for warfarin while a separate arm studied magnesium.",
     "entity_1": "warfarin", "entity_2": "magnesium", "relation": "NO_RELATION"},
    {"sentence": "Dopamine levels were quantified before the scheduled electrocardiogram procedure.",
     "entity_1": "Dopamine", "entity_2": "electrocardiogram", "relation": "NO_RELATION"},
    {"sentence": "Metformin was compared with placebo but the endpoint was platelet aggregation.",
     "entity_1": "Metformin", "entity_2": "platelet aggregation", "relation": "NO_RELATION"},
    {"sentence": "The lab processed samples for cortisol assay and separately for vitamin B12 measurement.",
     "entity_1": "cortisol", "entity_2": "vitamin B12", "relation": "NO_RELATION"},
    {"sentence": "Heparin vials and saline ampoules were placed on the same shelf for convenience.",
     "entity_1": "Heparin", "entity_2": "saline", "relation": "NO_RELATION"},
]

# ---------------------------------------------------------------------------
# Keyword groups for the Keyword Baseline
# ---------------------------------------------------------------------------
KEYWORD_GROUPS: Dict[str, List[str]] = {
    "TREATS": [
        "treats", "reduces", "relieves", "manages", "improves",
        "controls", "therapy", "therapeutic",
    ],
    "CAUSES": [
        "causes", "induces", "triggers", "produces", "leads", "results",
    ],
    "INHIBITS": [
        "inhibits", "blocks", "suppresses", "prevents",
        "reduces activity", "downregulates",
    ],
    "ASSOCIATED_WITH": [
        "associated", "linked", "correlated", "related",
        "connected", "risk factor",
    ],
}

# Precedence order: TREATS > CAUSES > INHIBITS > ASSOCIATED_WITH > NO_RELATION
KEYWORD_PRECEDENCE = ["TREATS", "CAUSES", "INHIBITS", "ASSOCIATED_WITH"]


# ---------------------------------------------------------------------------
# NLTK resource setup
# ---------------------------------------------------------------------------
def ensure_nltk_resources() -> None:
    """Download only missing NLTK resources with readable error handling."""
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
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> Tuple[List[str], str]:
    """Shared preprocessing: lowercase, tokenize, filter, remove stopwords, lemmatize."""
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens, " ".join(tokens)


# ---------------------------------------------------------------------------
# A. Keyword Baseline
# ---------------------------------------------------------------------------
def keyword_predict(sentence: str) -> str:
    """Predict relation using keyword matching with fixed precedence."""
    _, processed = preprocess(sentence)
    for label in KEYWORD_PRECEDENCE:
        for kw in KEYWORD_GROUPS[label]:
            if kw in processed:
                return label
    return "NO_RELATION"


# ---------------------------------------------------------------------------
# B. Regex / Pattern-Based Classifier
# ---------------------------------------------------------------------------
_REGEX_PATTERNS: List[Tuple[str, str]] = [
    # TREATS patterns
    (r"used\s+to\s+treat", "TREATS"),
    (r"therap(y|eutic)\s+(agent\s+)?for", "TREATS"),
    (r"(?:treats|treat)\b", "TREATS"),
    (r"\brelieves?\b", "TREATS"),
    (r"\bmanages?\b", "TREATS"),
    (r"\bimproves?\b", "TREATS"),
    (r"\bcontrols?\b", "TREATS"),
    (r"\breduces?\b.*\b(?:symptoms?|inflammation|levels?|load|pain|fever)\b", "TREATS"),
    (r"\bprescribed\s+to\b", "TREATS"),
    # CAUSES patterns
    (r"responsible\s+for", "CAUSES"),
    (r"caused\s+by", "CAUSES"),
    (r"\bcauses?\b", "CAUSES"),
    (r"\binduces?\b", "CAUSES"),
    (r"\btriggers?\b", "CAUSES"),
    (r"\bproduces?\b", "CAUSES"),
    (r"\bleads?\s+to\b", "CAUSES"),
    (r"\bresults?\s+in\b", "CAUSES"),
    # INHIBITS patterns
    (r"inhibits?\s+the\s+activity\s+of", "INHIBITS"),
    (r"\binhibits?\b", "INHIBITS"),
    (r"\bblocks?\b", "INHIBITS"),
    (r"\bsuppresses?\b", "INHIBITS"),
    (r"\bprevents?\b", "INHIBITS"),
    (r"\bdownregulates?\b", "INHIBITS"),
    (r"\breduces?\b.*\bactivity\b", "INHIBITS"),
    # ASSOCIATED_WITH patterns
    (r"associated\s+with", "ASSOCIATED_WITH"),
    (r"\blinked\s+to\b", "ASSOCIATED_WITH"),
    (r"\bcorrelated\s+with\b", "ASSOCIATED_WITH"),
    (r"\brelated\s+to\b", "ASSOCIATED_WITH"),
    (r"\bconnected\s+to\b", "ASSOCIATED_WITH"),
    (r"\brisk\s+factor\s+for\b", "ASSOCIATED_WITH"),
    # NO_RELATION patterns
    (r"has\s+no\s+effect\s+on", "NO_RELATION"),
    (r"was\s+compared\s+with", "NO_RELATION"),
    (r"\bno\s+connection\b", "NO_RELATION"),
    (r"\bunrelated\s+to\b", "NO_RELATION"),
    (r"\bindependently\s+of\b", "NO_RELATION"),
    (r"\bwithout\s+interaction\b", "NO_RELATION"),
    (r"\bseparately\b", "NO_RELATION"),
    (r"\bseparate\s+arm\b", "NO_RELATION"),
]


def regex_predict(sentence: str, entity_1: str, entity_2: str) -> str:
    """Predict relation using regex patterns on the sentence and entity context."""
    sent_lower = sentence.lower()
    e1_lower = entity_1.lower()
    e2_lower = entity_2.lower()

    # Find positions to look at text between entities
    e1_pos = sent_lower.find(e1_lower)
    e2_pos = sent_lower.find(e2_lower)

    if e1_pos >= 0 and e2_pos >= 0:
        start = min(e1_pos + len(e1_lower), e2_pos + len(e2_lower))
        end = max(e1_pos, e2_pos)
        between = sent_lower[start:end] if start < end else ""
    else:
        between = ""

    # Check NO_RELATION patterns first on full sentence
    for pattern, label in _REGEX_PATTERNS:
        if label == "NO_RELATION":
            if re.search(pattern, sent_lower):
                return "NO_RELATION"

    # Check remaining patterns on between-text first, then full sentence
    for text_to_search in [between, sent_lower]:
        for pattern, label in _REGEX_PATTERNS:
            if label == "NO_RELATION":
                continue
            if re.search(pattern, text_to_search):
                return label

    return "NO_RELATION"


# ---------------------------------------------------------------------------
# C & D. ML model helpers
# ---------------------------------------------------------------------------
def build_tfidf_nb(
    train_texts: List[str], train_labels: List[str]
) -> Tuple[TfidfVectorizer, MultinomialNB]:
    """Train a Multinomial Naive Bayes on TF-IDF features."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    model = MultinomialNB()
    model.fit(X_train, train_labels)
    return vectorizer, model


def build_tfidf_lr(
    train_texts: List[str], train_labels: List[str]
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Train a Logistic Regression on TF-IDF features."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
    )
    model.fit(X_train, train_labels)
    return vectorizer, model


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def stratified_split(
    data: List[Dict[str, str]], ratio: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split data into train/test with stratification by relation label."""
    rng = np.random.RandomState(seed)
    by_label: Dict[str, List[Dict[str, str]]] = {}
    for rec in data:
        by_label.setdefault(rec["relation"], []).append(rec)

    train, test = [], []
    for label in RELATION_LABELS:
        items = by_label.get(label, [])
        indices = rng.permutation(len(items))
        n_train = int(len(items) * ratio)
        for i in indices[:n_train]:
            train.append(items[i])
        for i in indices[n_train:]:
            test.append(items[i])
    return train, test


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate(
    y_true: List[str], y_pred: List[str]
) -> Dict[str, float]:
    """Compute accuracy, macro/weighted precision, recall, F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def per_relation_metrics(
    y_true: List[str], y_pred: List[str]
) -> Dict[str, Dict[str, float]]:
    """Per-relation precision, recall, F1, support."""
    result = {}
    for label in RELATION_LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[label] = {
            "precision": prec, "recall": rec, "f1": f1, "support": support
        }
    return result


def format_confusion_matrix(y_true: List[str], y_pred: List[str]) -> str:
    """Create a readable text confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=RELATION_LABELS)
    # Short labels for display
    short = ["TREATS", "CAUSES", "INHIBITS", "ASSOC_W", "NO_REL"]
    col_w = 9
    header = " " * 16 + "".join(s.rjust(col_w) for s in short)
    lines = [header]
    for i, label in enumerate(RELATION_LABELS):
        display = label[:15].ljust(16)
        row = "".join(str(cm[i][j]).rjust(col_w) for j in range(len(RELATION_LABELS)))
        lines.append(display + row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entity validation
# ---------------------------------------------------------------------------
def validate_entities(sentence: str, entity_1: str, entity_2: str) -> List[str]:
    """Check that both entities appear in the sentence (case-insensitive)."""
    errors = []
    sent_lower = sentence.lower()
    if entity_1.lower() not in sent_lower:
        errors.append(f"Entity 1 '{entity_1}' not found in the sentence.")
    if entity_2.lower() not in sent_lower:
        errors.append(f"Entity 2 '{entity_2}' not found in the sentence.")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 06: Biomedical Relation Identification"
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Biomedical sentence")
    parser.add_argument("--entity-one", default=DEFAULT_ENTITY_1, help="First entity")
    parser.add_argument("--entity-two", default=DEFAULT_ENTITY_2, help="Second entity")
    args = parser.parse_args()

    # NLTK setup
    ensure_nltk_resources()

    # ── Dataset split ─────────────────────────────────────────────────
    train_data, test_data = stratified_split(DATASET, SPLIT_RATIO, RANDOM_SEED)

    train_sentences = [r["sentence"] for r in train_data]
    train_labels = [r["relation"] for r in train_data]
    test_sentences = [r["sentence"] for r in test_data]
    test_labels = [r["relation"] for r in test_data]
    test_e1 = [r["entity_1"] for r in test_data]
    test_e2 = [r["entity_2"] for r in test_data]

    # Preprocess train/test for ML
    _, train_processed = zip(*[preprocess(s) for s in train_sentences])
    _, test_processed = zip(*[preprocess(s) for s in test_sentences])
    train_processed = list(train_processed)
    test_processed = list(test_processed)

    # Count helpers
    from collections import Counter
    total_counts = Counter(r["relation"] for r in DATASET)
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    # ── Train ML models ──────────────────────────────────────────────
    t0 = time.time()
    nb_vec, nb_model = build_tfidf_nb(train_processed, train_labels)
    nb_train_time = time.time() - t0

    t0 = time.time()
    lr_vec, lr_model = build_tfidf_lr(train_processed, train_labels)
    lr_train_time = time.time() - t0

    # ── Predictions ──────────────────────────────────────────────────
    # A. Keyword
    t0 = time.time()
    kw_preds = [keyword_predict(s) for s in test_sentences]
    kw_time = time.time() - t0

    # B. Regex
    t0 = time.time()
    regex_preds = [
        regex_predict(s, e1, e2)
        for s, e1, e2 in zip(test_sentences, test_e1, test_e2)
    ]
    regex_time = time.time() - t0

    # C. Naive Bayes
    t0 = time.time()
    nb_preds = nb_model.predict(nb_vec.transform(test_processed)).tolist()
    nb_inf_time = time.time() - t0

    # D. Logistic Regression
    t0 = time.time()
    lr_preds = lr_model.predict(lr_vec.transform(test_processed)).tolist()
    lr_inf_time = time.time() - t0

    # ── Evaluation ───────────────────────────────────────────────────
    results = {
        "Keyword Baseline": {**evaluate(test_labels, kw_preds), "inf_time": kw_time},
        "Regex Patterns": {**evaluate(test_labels, regex_preds), "inf_time": regex_time},
        "Naive Bayes": {
            **evaluate(test_labels, nb_preds),
            "inf_time": nb_inf_time,
            "train_time": nb_train_time,
        },
        "Logistic Regression": {
            **evaluate(test_labels, lr_preds),
            "inf_time": lr_inf_time,
            "train_time": lr_train_time,
        },
    }

    # Select best method
    ranking = sorted(
        results.items(),
        key=lambda x: (x[1]["macro_f1"], x[1]["macro_recall"], x[1]["accuracy"]),
        reverse=True,
    )
    best_name = ranking[0][0]

    best_preds_map = {
        "Keyword Baseline": kw_preds,
        "Regex Patterns": regex_preds,
        "Naive Bayes": nb_preds,
        "Logistic Regression": lr_preds,
    }
    best_preds = best_preds_map[best_name]

    # Per-relation metrics for best
    best_per_rel = per_relation_metrics(test_labels, best_preds)
    cm_text = format_confusion_matrix(test_labels, best_preds)

    # Error analysis
    errors_list = []
    for i, (t, p) in enumerate(zip(test_labels, best_preds)):
        if t != p and len(errors_list) < 5:
            errors_list.append({
                "sentence": test_data[i]["sentence"],
                "entity_1": test_data[i]["entity_1"],
                "entity_2": test_data[i]["entity_2"],
                "true": t,
                "predicted": p,
            })

    # ── Custom sentence analysis ─────────────────────────────────────
    custom_text = args.text
    custom_e1 = args.entity_one
    custom_e2 = args.entity_two
    entity_errors = validate_entities(custom_text, custom_e1, custom_e2)

    custom_tokens, custom_processed_str = preprocess(custom_text)

    custom_predictions: Dict[str, str] = {}
    nb_proba = None
    lr_proba = None

    if not entity_errors:
        custom_predictions["Keyword Baseline"] = keyword_predict(custom_text)
        custom_predictions["Regex Patterns"] = regex_predict(
            custom_text, custom_e1, custom_e2
        )

        X_custom = nb_vec.transform([custom_processed_str])
        custom_predictions["Naive Bayes"] = nb_model.predict(X_custom)[0]
        nb_proba = dict(zip(nb_model.classes_, nb_model.predict_proba(X_custom)[0]))

        X_custom_lr = lr_vec.transform([custom_processed_str])
        custom_predictions["Logistic Regression"] = lr_model.predict(X_custom_lr)[0]
        lr_proba = dict(
            zip(lr_model.classes_, lr_model.predict_proba(X_custom_lr)[0])
        )

        final_relation = custom_predictions.get(best_name, "N/A")
    else:
        final_relation = "N/A (entity validation failed)"

    # ── Build report ─────────────────────────────────────────────────
    report_lines: List[str] = []
    w = report_lines.append

    w("=" * 72)
    w("EXPERIMENT 06 — BIOMEDICAL RELATION IDENTIFICATION")
    w("=" * 72)
    w("")
    w("Student Name    : Divya M")
    w("Register Number : 24AD0074")
    w("")
    w(f"Title: {EXPERIMENT_TITLE}")
    w("")
    w(f"Aim: {AIM}")
    w("")
    w("Note: Multiple alternative algorithms are compared in this experiment.")
    w("      The original title and aim are preserved while evaluating a")
    w("      keyword baseline, regex patterns, Naive Bayes and Logistic")
    w("      Regression on the same biomedical relation-identification task.")
    w("")

    # Dataset summary
    w("-" * 72)
    w("DATASET SUMMARY")
    w("-" * 72)
    w(f"Total sentences     : {len(DATASET)}")
    w(f"Training sentences  : {len(train_data)}")
    w(f"Testing sentences   : {len(test_data)}")
    w("")
    w("Relation counts (Total / Train / Test):")
    for label in RELATION_LABELS:
        w(f"  {label:20s}  {total_counts[label]:3d}  /  "
          f"{train_counts[label]:3d}  /  {test_counts[label]:3d}")
    w("")

    # Approach descriptions
    w("-" * 72)
    w("APPROACH DESCRIPTIONS")
    w("-" * 72)
    w("")
    w("A. Keyword Baseline")
    w("   Deterministic keyword matching with fixed precedence:")
    w("   TREATS > CAUSES > INHIBITS > ASSOCIATED_WITH > NO_RELATION")
    w("   Uses predefined keyword groups per relation type.")
    w("")
    w("B. Regex and Pattern-Based Classifier")
    w("   Regular expression patterns examining entity order, words between")
    w("   entities, passive constructions, and multi-word phrases.")
    w("   Returns one of five relation labels based on pattern matches.")
    w("")
    w("C. Multinomial Naive Bayes")
    w("   TF-IDF vectorization with unigrams and bigrams (ngram_range=(1,2)).")
    w("   Trained on preprocessed training sentences.")
    w("")
    w("D. Logistic Regression")
    w("   TF-IDF vectorization with unigrams and bigrams.")
    w("   max_iter=1000, class_weight='balanced', random_state=42.")
    w("")

    # Evaluation table
    w("-" * 72)
    w("EVALUATION TABLE")
    w("-" * 72)
    header = (
        f"{'Method':<22s} {'Acc':>6s} {'M-Prec':>7s} {'M-Rec':>6s} "
        f"{'M-F1':>6s} {'W-F1':>6s} {'T-Time':>8s} {'I-Time':>8s}"
    )
    w(header)
    w("-" * len(header))
    for name, m in results.items():
        tt = f"{m.get('train_time', 0):.4f}" if "train_time" in m else "   N/A"
        w(
            f"{name:<22s} {m['accuracy']:6.3f} {m['macro_precision']:7.3f} "
            f"{m['macro_recall']:6.3f} {m['macro_f1']:6.3f} "
            f"{m['weighted_f1']:6.3f} {tt:>8s} {m['inf_time']:8.4f}"
        )
    w("")
    w(f"Best-performing method: {best_name}")
    w("  (Selected by: highest macro F1 > macro recall > accuracy)")
    w("")

    # Per-relation metrics
    w("-" * 72)
    w(f"PER-RELATION METRICS — {best_name}")
    w("-" * 72)
    pr_header = f"{'Relation':<22s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}"
    w(pr_header)
    w("-" * len(pr_header))
    for label in RELATION_LABELS:
        m = best_per_rel[label]
        w(
            f"{label:<22s} {m['precision']:10.3f} {m['recall']:8.3f} "
            f"{m['f1']:8.3f} {m['support']:8d}"
        )
    w("")

    # Confusion matrix
    w("-" * 72)
    w(f"CONFUSION MATRIX — {best_name}")
    w("-" * 72)
    w("Rows: True label | Columns: Predicted label")
    w("")
    w(cm_text)
    w("")

    # Error analysis
    w("-" * 72)
    w("ERROR ANALYSIS")
    w("-" * 72)
    if not errors_list:
        w("No misclassified examples were found.")
    else:
        w(f"Showing up to 5 misclassified test examples ({len(errors_list)} found):")
        w("")
        for idx, err in enumerate(errors_list, 1):
            w(f"  {idx}. Sentence  : {err['sentence']}")
            w(f"     Entity 1  : {err['entity_1']}")
            w(f"     Entity 2  : {err['entity_2']}")
            w(f"     True      : {err['true']}")
            w(f"     Predicted : {err['predicted']}")
            w("")

    # Custom sentence analysis
    w("-" * 72)
    w("CUSTOM BIOMEDICAL SENTENCE ANALYSIS")
    w("-" * 72)
    w(f"Sentence  : {custom_text}")
    w(f"Entity 1  : {custom_e1}")
    w(f"Entity 2  : {custom_e2}")
    w(f"Tokens    : {custom_tokens}")
    w("")

    if entity_errors:
        for err in entity_errors:
            w(f"ERROR: {err}")
    else:
        for mname, pred in custom_predictions.items():
            w(f"  {mname:<22s} => {pred}")
        w("")
        w(f"Final prediction ({best_name}): {final_relation}")
        w("")
        if nb_proba:
            w("  Naive Bayes probabilities:")
            for label in RELATION_LABELS:
                w(f"    {label:<20s} {nb_proba.get(label, 0.0):.4f}")
        if lr_proba:
            w("  Logistic Regression probabilities:")
            for label in RELATION_LABELS:
                w(f"    {label:<20s} {lr_proba.get(label, 0.0):.4f}")
    w("")

    # Result statement
    w("-" * 72)
    w("RESULT")
    w("-" * 72)
    w(f"The experiment compared four approaches for biomedical relation")
    w(f"identification: Keyword Baseline, Regex Patterns, Multinomial Naive")
    w(f"Bayes and Logistic Regression. The best-performing method was")
    w(f"{best_name} with a macro F1-score of "
      f"{results[best_name]['macro_f1']:.3f}.")
    w("=" * 72)

    report = "\n".join(report_lines) + "\n"

    # ── Write output file ────────────────────────────────────────────
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.txt"
    output_file.write_text(report, encoding="utf-8")

    # ── Terminal summary ─────────────────────────────────────────────
    print("=" * 60)
    print("EXPERIMENT 06 — Biomedical Relation Identification")
    print("=" * 60)
    print(f"Student : Divya M | Reg No : 24AD0074")
    print(f"Dataset : {len(DATASET)} sentences | "
          f"Train: {len(train_data)} | Test: {len(test_data)}")
    print()
    print(f"{'Method':<22s} {'Acc':>5s} {'M-F1':>5s} {'W-F1':>5s}")
    print("-" * 42)
    for name, m in results.items():
        print(f"{name:<22s} {m['accuracy']:5.3f} {m['macro_f1']:5.3f} "
              f"{m['weighted_f1']:5.3f}")
    print()
    print(f"Best method: {best_name} "
          f"(Macro F1 = {results[best_name]['macro_f1']:.3f})")
    print()
    print(f"Custom sentence : {custom_text}")
    print(f"Entities        : {custom_e1} | {custom_e2}")
    if entity_errors:
        for err in entity_errors:
            print(f"  ERROR: {err}")
    else:
        for mname, pred in custom_predictions.items():
            print(f"  {mname:<22s} => {pred}")
        print(f"  Final ({best_name}): {final_relation}")
    print()
    print(f"Output saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
