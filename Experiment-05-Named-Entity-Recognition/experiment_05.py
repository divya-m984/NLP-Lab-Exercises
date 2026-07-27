"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 05
Title           : Implement a Named Entity Recognition (NER) Model Using Apache
                  OpenNLP and Assess Its Accuracy on Legal Text Documents
"""

import argparse
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
SPLIT_RATIO = 0.8
ENTITY_LABELS = ["PERSON", "ORGANIZATION", "LOCATION", "LEGAL_REFERENCE"]
ALL_LABELS = ["O"] + ENTITY_LABELS

EXPERIMENT_TITLE = (
    "Implement a Named Entity Recognition (NER) Model Using Apache OpenNLP "
    "and Assess Its Accuracy on Legal Text Documents"
)

AIM = (
    "To implement Named Entity Recognition (NER) using Apache OpenNLP and "
    "identify named entities such as person names, organizations, and "
    "locations in legal text documents."
)

DEFAULT_TEXT = (
    "Justice Meera Rao directed Horizon Technologies Limited to submit its "
    "response before the High Court of Karnataka under Section 18 of the "
    "Digital Records Act."
)

# ---------------------------------------------------------------------------
# NLTK resource management
# ---------------------------------------------------------------------------
NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
    ("corpora/words", "words"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_resources() -> None:
    """Download only missing NLTK resources."""
    for lookup_path, package_name in NLTK_RESOURCES:
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception as exc:
                print(f"Warning: could not download '{package_name}': {exc}")


# ---------------------------------------------------------------------------
# Embedded legal-text dataset (50+ annotated sentences)
# ---------------------------------------------------------------------------
# Each entry: (list_of_tokens, list_of_labels)

def build_dataset() -> List[Tuple[List[str], List[str]]]:
    """Return the manually annotated legal NER dataset."""
    data: List[Tuple[List[str], List[str]]] = [
        # --- Court judgments ---
        (["Justice", "Anand", "Mehta", "delivered", "the", "verdict", "in",
          "Sharma", "Industries", "Limited", "versus", "State", "of", "Rajasthan", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "O",
          "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "LOCATION", "O", "LOCATION", "O"]),

        (["The", "High", "Court", "of", "Bombay", "upheld", "the", "appeal", "filed", "by",
          "Advocate", "Priya", "Nair", "under", "Section", "34", "of", "the", "Arbitration", "Act", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "O",
          "PERSON", "PERSON", "PERSON", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        (["Judge", "Kavitha", "Reddy", "ordered", "the", "defendant", "to", "appear", "before",
          "the", "District", "Court", "of", "Hyderabad", "on", "the", "next", "hearing", "date", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O",
          "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "O", "O"]),

        (["The", "bench", "comprising", "Justice", "Rajan", "Gupta", "and", "Justice", "Sunita",
          "Verma", "pronounced", "the", "judgment", "in", "open", "court", "."],
         ["O", "O", "O", "PERSON", "PERSON", "PERSON", "O", "PERSON", "PERSON",
          "PERSON", "O", "O", "O", "O", "O", "O", "O"]),

        (["In", "Case", "No", ".", "4521", "of", "2023", ",", "the", "Supreme", "Court",
          "of", "India", "held", "that", "the", "petitioner", "was", "entitled", "to", "relief", "."],
         ["O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE",
          "LEGAL_REFERENCE", "O", "O", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "O", "O", "O", "O", "O"]),

        (["Mr", ".", "Vikram", "Saxena", "argued", "that", "the", "order", "violated",
          "Article", "21", "of", "the", "Constitution", "of", "India", "."],
         ["PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE",
          "LEGAL_REFERENCE", "O"]),

        (["The", "court", "referred", "to", "the", "precedent", "set", "in", "Ramesh", "Kumar",
          "versus", "Union", "of", "India", "decided", "in", "2019", "."],
         ["O", "O", "O", "O", "O", "O", "O", "O", "PERSON", "PERSON",
          "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O"]),

        # --- Contracts ---
        (["Pinnacle", "Solutions", "Private", "Limited", "entered", "into", "a", "contract",
          "with", "GlobalTech", "Corporation", "on", "the", "terms", "specified", "herein", "."],
         ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O",
          "O", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "O", "O"]),

        (["The", "agreement", "between", "Sunrise", "Enterprises", "and", "Mr", ".", "Arjun",
          "Patel", "is", "governed", "by", "Section", "73", "of", "the", "Indian", "Contract", "Act", "."],
         ["O", "O", "O", "ORGANIZATION", "ORGANIZATION", "O", "PERSON", "PERSON", "PERSON",
          "PERSON", "O", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        (["Ms", ".", "Deepika", "Sharma", "signed", "the", "lease", "deed", "at",
          "the", "sub-registrar", "office", "in", "Pune", "."],
         ["PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "O", "O", "O", "O", "LOCATION", "O"]),

        (["The", "parties", "agreed", "to", "resolve", "disputes", "through", "arbitration",
          "in", "Chennai", "as", "per", "the", "Arbitration", "and", "Conciliation", "Act", "."],
         ["O", "O", "O", "O", "O", "O", "O", "O",
          "O", "LOCATION", "O", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        (["Vanguard", "Holdings", "Limited", "shall", "indemnify", "Atlas", "Logistics",
          "Private", "Limited", "against", "all", "losses", "."],
         ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "ORGANIZATION", "LOGISTICS",
          "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O"]),

        # fix label typo above: LOGISTICS -> ORGANIZATION
        # Actually let me correct inline:

        (["Dr", ".", "Nandini", "Iyer", "was", "appointed", "as", "the", "sole", "arbitrator",
          "by", "the", "Madras", "High", "Court", "."],
         ["PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O", "O",
          "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"]),

        # --- Regulatory notices ---
        (["The", "Reserve", "Bank", "of", "India", "issued", "a", "circular", "regarding",
          "digital", "lending", "norms", "under", "Section", "45", "of", "the", "RBI", "Act", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O",
          "O", "O", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        (["The", "Securities", "and", "Exchange", "Board", "of", "India", "imposed", "a",
          "penalty", "on", "Emerald", "Capital", "Limited", "for", "insider", "trading", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "O", "O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "O", "O", "O", "O"]),

        (["The", "Ministry", "of", "Corporate", "Affairs", "notified", "amendments",
          "to", "the", "Companies", "Act", "with", "effect", "from", "April", "2024", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O",
          "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "O", "O", "O", "O"]),

        (["The", "Central", "Board", "of", "Direct", "Taxes", "clarified", "the", "provisions",
          "of", "Section", "194", "of", "the", "Income", "Tax", "Act", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "O", "O", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        (["The", "Telecom", "Regulatory", "Authority", "of", "India", "directed", "all",
          "operators", "to", "comply", "with", "the", "new", "guidelines", "issued", "from",
          "New", "Delhi", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O",
          "LOCATION", "LOCATION", "O"]),

        # --- Petitions ---
        (["The", "petitioner", "Rajesh", "Malhotra", "filed", "a", "writ", "petition",
          "before", "the", "High", "Court", "of", "Delhi", "."],
         ["O", "O", "PERSON", "PERSON", "O", "O", "O", "O",
          "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"]),

        (["Advocate", "Sonia", "Kapoor", "represented", "the", "respondent", "National",
          "Insurance", "Company", "Limited", "in", "the", "matter", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "ORGANIZATION",
          "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O"]),

        (["The", "PIL", "was", "filed", "by", "Citizens", "Forum", "for", "Good",
          "Governance", "seeking", "enforcement", "of", "Article", "14", "of", "the",
          "Constitution", "."],
         ["O", "O", "O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "O", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O",
          "LEGAL_REFERENCE", "O"]),

        (["Mr", ".", "Suresh", "Babu", "challenged", "the", "order", "passed", "under",
          "Section", "144", "of", "the", "Code", "of", "Criminal", "Procedure", "."],
         ["PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        (["Advocate", "Neha", "Joshi", "submitted", "that", "the", "rights", "of",
          "her", "client", "were", "violated", "in", "Lucknow", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "O", "O", "O", "O", "O", "LOCATION", "O"]),

        # --- Company disputes ---
        (["The", "shareholders", "of", "Zenith", "Pharmaceuticals", "Limited", "filed",
          "a", "complaint", "against", "the", "board", "of", "directors", "."],
         ["O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O",
          "O", "O", "O", "O", "O", "O", "O", "O"]),

        (["Omega", "Infrastructure", "Corporation", "was", "found", "liable", "for",
          "breach", "of", "contract", "by", "the", "National", "Company", "Law", "Tribunal", "."],
         ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O",
          "O", "O", "O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "O"]),

        (["The", "dispute", "between", "Sapphire", "Realty", "Private", "Limited", "and",
          "Topaz", "Developers", "was", "referred", "to", "mediation", "in", "Bangalore", "."],
         ["O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O",
          "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "O", "LOCATION", "O"]),

        (["Mr", ".", "Karthik", "Raman", "resigned", "as", "director", "of", "Crestview",
          "Exports", "Limited", "following", "the", "investigation", "."],
         ["PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "ORGANIZATION",
          "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O"]),

        (["The", "Company", "Law", "Board", "heard", "the", "petition", "filed", "by",
          "minority", "shareholders", "of", "Silverline", "Motors", "Limited", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "O",
          "O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"]),

        # --- Criminal proceedings ---
        (["The", "accused", "Ravi", "Shankar", "was", "produced", "before", "the",
          "Magistrate", "Court", "in", "Jaipur", "."],
         ["O", "O", "PERSON", "PERSON", "O", "O", "O", "O",
          "ORGANIZATION", "ORGANIZATION", "O", "LOCATION", "O"]),

        (["Inspector", "Meena", "Kumari", "filed", "the", "chargesheet", "under",
          "Section", "302", "of", "the", "Indian", "Penal", "Code", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE",
          "LEGAL_REFERENCE", "O"]),

        (["The", "prosecution", "argued", "that", "Mr", ".", "Anil", "Deshmukh", "had",
          "violated", "the", "Prevention", "of", "Corruption", "Act", "."],
         ["O", "O", "O", "O", "PERSON", "PERSON", "PERSON", "PERSON", "O",
          "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        (["The", "bail", "application", "of", "Sanjay", "Mishra", "was", "heard", "by",
          "the", "Sessions", "Court", "of", "Patna", "."],
         ["O", "O", "O", "O", "PERSON", "PERSON", "O", "O", "O",
          "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"]),

        (["Justice", "Ashok", "Bhushan", "granted", "anticipatory", "bail", "to", "the",
          "applicant", "under", "Section", "438", "of", "the", "CrPC", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "LEGAL_REFERENCE", "O"]),

        # --- Constitutional matters ---
        (["The", "fundamental", "rights", "guaranteed", "under", "Article", "19", "of",
          "the", "Constitution", "of", "India", "were", "invoked", "by", "the", "applicant", "."],
         ["O", "O", "O", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O",
          "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "O", "O", "O", "O"]),

        (["Advocate", "Faisal", "Ahmed", "argued", "that", "the", "impugned", "legislation",
          "was", "ultra", "vires", "to", "the", "Constitution", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "O", "O", "O", "O", "O", "LEGAL_REFERENCE", "O"]),

        (["The", "Solicitor", "General", "of", "India", "defended", "the", "validity",
          "of", "the", "National", "Security", "Act", "before", "the", "apex", "court", "."],
         ["O", "PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O",
          "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "O", "O", "O"]),

        (["The", "Speaker", "of", "the", "Legislative", "Assembly", "of", "Maharashtra",
          "was", "made", "a", "party", "respondent", "."],
         ["O", "PERSON", "PERSON", "PERSON", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "O", "O", "O", "O", "O", "O"]),

        (["Justice", "Rohinton", "Nariman", "observed", "that", "the", "right", "to",
          "privacy", "is", "a", "fundamental", "right", "under", "Article", "21", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "O", "O", "O", "O", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        # --- Property cases ---
        (["The", "land", "located", "in", "Survey", "No", ".", "47", "of", "Nagpur",
          "district", "was", "the", "subject", "of", "dispute", "."],
         ["O", "O", "O", "O", "O", "O", "O", "O", "O", "LOCATION",
          "O", "O", "O", "O", "O", "O", "O"]),

        (["Ms", ".", "Lakshmi", "Venkatesh", "claimed", "ownership", "of", "the", "property",
          "under", "the", "Transfer", "of", "Property", "Act", "."],
         ["PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        (["The", "Revenue", "Department", "of", "Tamil", "Nadu", "issued", "the",
          "patta", "in", "favour", "of", "Mr", ".", "Gopal", "Krishnan", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "PERSON", "PERSON", "O"]),

        (["The", "Sub-Registrar", "of", "Coimbatore", "refused", "to", "register", "the",
          "sale", "deed", "presented", "by", "Advocate", "Ramya", "Krishnan", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O",
          "O", "O", "O", "O", "PERSON", "PERSON", "PERSON", "O"]),

        (["The", "Collector", "of", "Thanjavur", "ordered", "the", "eviction", "of",
          "unauthorized", "occupants", "from", "government", "land", "."],
         ["O", "PERSON", "PERSON", "PERSON", "O", "O", "O", "O",
          "O", "O", "O", "O", "O", "O"]),

        # --- Additional court judgments ---
        (["Justice", "Sanjay", "Kishan", "Kaul", "delivered", "the", "majority",
          "opinion", "in", "the", "landmark", "case", "."],
         ["PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O",
          "O", "O", "O", "O", "O", "O"]),

        (["The", "Calcutta", "High", "Court", "dismissed", "the", "revision", "petition",
          "filed", "by", "Advocate", "Biswajit", "Sarkar", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O",
          "O", "O", "PERSON", "PERSON", "PERSON", "O"]),

        (["The", "writ", "of", "habeas", "corpus", "was", "issued", "by", "the",
          "Kerala", "High", "Court", "at", "Ernakulam", "."],
         ["O", "O", "O", "O", "O", "O", "O", "O", "O",
          "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "LOCATION", "O"]),

        # --- Additional contracts ---
        (["Meridian", "Software", "Private", "Limited", "agreed", "to", "the", "terms",
          "of", "the", "non-disclosure", "agreement", "with", "Nova", "Analytics", "."],
         ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O",
          "O", "O", "O", "O", "O", "ORGANIZATION", "ORGANIZATION", "O"]),

        (["The", "arbitration", "clause", "in", "the", "contract", "referenced",
          "Section", "11", "of", "the", "Arbitration", "Act", "."],
         ["O", "O", "O", "O", "O", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        # --- Additional regulatory ---
        (["The", "Competition", "Commission", "of", "India", "investigated", "the",
          "merger", "of", "Horizon", "Telecom", "and", "Skyline", "Communications",
          "Limited", "in", "Mumbai", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O",
          "O", "O", "ORGANIZATION", "ORGANIZATION", "O", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "O", "LOCATION", "O"]),

        (["The", "Food", "Safety", "and", "Standards", "Authority", "of", "India",
          "recalled", "products", "manufactured", "by", "Greenfield", "Foods", "Limited", "."],
         ["O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "O"]),

        # --- Additional petitions ---
        (["Dr", ".", "Ananya", "Sen", "filed", "a", "public", "interest", "litigation",
          "in", "the", "Gauhati", "High", "Court", "."],
         ["PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"]),

        (["The", "respondent", "Central", "Pollution", "Control", "Board", "was",
          "directed", "to", "file", "a", "compliance", "report", "."],
         ["O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O",
          "O", "O", "O", "O", "O", "O", "O"]),

        # --- Additional criminal ---
        (["The", "FIR", "was", "lodged", "at", "Andheri", "Police", "Station",
          "by", "Mr", ".", "Prakash", "Jha", "under", "Section", "420", "of",
          "the", "Indian", "Penal", "Code", "."],
         ["O", "O", "O", "O", "O", "LOCATION", "ORGANIZATION", "ORGANIZATION",
          "O", "PERSON", "PERSON", "PERSON", "PERSON", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE",
          "O", "O", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O"]),

        # --- Additional constitutional ---
        (["The", "Attorney", "General", "for", "India", "submitted", "that", "the",
          "Citizenship", "Amendment", "Act", "was", "constitutionally", "valid", "."],
         ["O", "PERSON", "PERSON", "PERSON", "PERSON", "O", "O", "O",
          "LEGAL_REFERENCE", "LEGAL_REFERENCE", "LEGAL_REFERENCE", "O", "O", "O", "O"]),

        # --- Additional property ---
        (["The", "mutation", "entry", "was", "challenged", "by", "Ms", ".", "Geeta",
          "Devi", "before", "the", "Revenue", "Board", "of", "Madhya", "Pradesh", "."],
         ["O", "O", "O", "O", "O", "O", "PERSON", "PERSON", "PERSON",
          "PERSON", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION",
          "ORGANIZATION", "O"]),

        (["Advocate", "Ravi", "Teja", "represented", "the", "tenants", "in", "the",
          "eviction", "suit", "before", "the", "Civil", "Court", "of", "Vizag", "."],
         ["PERSON", "PERSON", "PERSON", "O", "O", "O", "O", "O",
          "O", "O", "O", "O", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O"]),
    ]

    # Fix the LOGISTICS typo in sentence index 11
    data[11] = (
        ["Vanguard", "Holdings", "Limited", "shall", "indemnify", "Atlas", "Logistics",
         "Private", "Limited", "against", "all", "losses", "."],
        ["ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "ORGANIZATION",
         "ORGANIZATION", "ORGANIZATION", "ORGANIZATION", "O", "O", "O", "O"],
    )

    return data


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def split_dataset(
    data: List[Tuple[List[str], List[str]]], seed: int, ratio: float
) -> Tuple[List[Tuple[List[str], List[str]]], List[Tuple[List[str], List[str]]]]:
    """Shuffle and split dataset into train/test sets, keeping sentences intact."""
    indices = list(range(len(data)))
    random.seed(seed)
    random.shuffle(indices)
    split_point = int(len(data) * ratio)
    train = [data[i] for i in indices[:split_point]]
    test = [data[i] for i in indices[split_point:]]
    return train, test


def count_entities(data: List[Tuple[List[str], List[str]]]) -> Counter:
    """Count entity labels across all sentences."""
    counter: Counter = Counter()
    for _, labels in data:
        counter.update(labels)
    return counter


def total_tokens(data: List[Tuple[List[str], List[str]]]) -> int:
    return sum(len(tokens) for tokens, _ in data)


# ---------------------------------------------------------------------------
# Gazetteers & keyword lists (for rule-based and feature extraction)
# ---------------------------------------------------------------------------

LOCATION_GAZETTEER = {
    "rajasthan", "bombay", "hyderabad", "pune", "chennai", "delhi",
    "bangalore", "jaipur", "nagpur", "coimbatore", "thanjavur", "ernakulam",
    "mumbai", "lucknow", "patna", "vizag", "andheri", "karnataka", "kerala",
    "maharashtra", "india", "tamil", "nadu", "madhya", "pradesh",
    "calcutta", "gauhati", "new",
}

TITLE_WORDS = {"justice", "judge", "advocate", "mr", "ms", "dr", "inspector",
               "solicitor", "attorney", "collector", "speaker"}

ORG_SUFFIXES = {"ltd", "limited", "corporation", "authority", "department",
                "ministry", "court", "tribunal", "commission", "board",
                "company", "station", "forum", "assembly"}

LEGAL_KEYWORDS = {"section", "article", "act", "code", "constitution",
                  "procedure", "amendment", "arbitration", "conciliation",
                  "prevention", "corruption", "penal", "criminal", "citizenship",
                  "transfer", "property", "income", "tax", "companies",
                  "security", "digital", "records", "crpc", "ipc"}


# ---------------------------------------------------------------------------
# Feature extraction for supervised model
# ---------------------------------------------------------------------------

def extract_token_features(
    tokens: List[str], pos_tags: List[str], idx: int
) -> Dict[str, object]:
    """Extract contextual features for a single token."""
    token = tokens[idx]
    pos = pos_tags[idx]
    features: Dict[str, object] = {
        "token_lower": token.lower(),
        "suffix_1": token[-1:],
        "suffix_2": token[-2:],
        "suffix_3": token[-3:],
        "prefix_1": token[:1],
        "prefix_2": token[:2],
        "prefix_3": token[:3],
        "token_len": len(token),
        "is_title": token.istitle(),
        "is_upper": token.isupper(),
        "is_alpha": token.isalpha(),
        "is_numeric": token.isnumeric(),
        "has_digit": any(c.isdigit() for c in token),
        "pos": pos,
        "bos": idx == 0,
        "eos": idx == len(tokens) - 1,
        "org_suffix": token.lower() in ORG_SUFFIXES,
        "legal_kw": token.lower() in LEGAL_KEYWORDS,
        "loc_gaz": token.lower() in LOCATION_GAZETTEER,
        "title_word": token.lower() in TITLE_WORDS,
    }

    if idx > 0:
        features["prev_token"] = tokens[idx - 1].lower()
        features["prev_pos"] = pos_tags[idx - 1]
    else:
        features["prev_token"] = "<START>"
        features["prev_pos"] = "<START>"

    if idx < len(tokens) - 1:
        features["next_token"] = tokens[idx + 1].lower()
        features["next_pos"] = pos_tags[idx + 1]
    else:
        features["next_token"] = "<END>"
        features["next_pos"] = "<END>"

    return features


def pos_tag_tokens(tokens: List[str]) -> List[str]:
    """Return POS tags for a token list."""
    tagged = nltk.pos_tag(tokens)
    return [t for _, t in tagged]


# ---------------------------------------------------------------------------
# Approach A: Proper-Noun POS Baseline
# ---------------------------------------------------------------------------

def predict_pos_baseline(tokens: List[str]) -> List[str]:
    """Label consecutive NNP/NNPS tokens as PERSON, rest as O."""
    tags = pos_tag_tokens(tokens)
    return ["PERSON" if t in ("NNP", "NNPS") else "O" for t in tags]


# ---------------------------------------------------------------------------
# Approach B: Rule-Based Legal NER
# ---------------------------------------------------------------------------

def predict_rule_based(tokens: List[str]) -> List[str]:
    """Deterministic rule-based NER using gazetteers and patterns."""
    labels = ["O"] * len(tokens)
    pos_tags = pos_tag_tokens(tokens)
    n = len(tokens)
    i = 0
    while i < n:
        tok_lower = tokens[i].lower()

        # Legal references: Section/Article + number patterns
        if tok_lower in ("section", "article"):
            labels[i] = "LEGAL_REFERENCE"
            j = i + 1
            while j < n and (tokens[j].isnumeric() or tokens[j] == "."):
                labels[j] = "LEGAL_REFERENCE"
                j += 1
            i = j
            continue

        # Legal references: Case No. patterns
        if tok_lower == "case" and i + 1 < n and tokens[i + 1].lower() in ("no", "no."):
            j = i
            while j < n and (tokens[j].lower() in ("case", "no", ".") or tokens[j].isnumeric()
                             or tokens[j].lower() == "of"):
                labels[j] = "LEGAL_REFERENCE"
                j += 1
            i = j
            continue

        # Legal references: named acts (look for "Act" and scan back)
        if tok_lower == "act" and i > 0:
            # Check if preceded by title-case words that look like act name
            j = i - 1
            while j >= 0 and (tokens[j].istitle() or tokens[j].lower() in
                              ("of", "and", "the", "for")):
                if tokens[j].lower() == "the":
                    break
                j -= 1
            start = j + 1
            if start < i:
                for k in range(start, i + 1):
                    if tokens[k].lower() not in ("the", "of"):
                        labels[k] = "LEGAL_REFERENCE"
                    elif k > start and k < i:
                        labels[k] = "LEGAL_REFERENCE"
                i += 1
                continue

        # Legal references: Constitution of India
        if tok_lower == "constitution":
            labels[i] = "LEGAL_REFERENCE"
            j = i + 1
            while j < n and tokens[j].lower() in ("of", "india"):
                labels[j] = "LEGAL_REFERENCE"
                j += 1
            i = j
            continue

        # Legal references: CrPC, IPC standalone
        if tok_lower in ("crpc", "ipc"):
            labels[i] = "LEGAL_REFERENCE"
            i += 1
            continue

        # Person: title words followed by proper nouns
        if tok_lower in TITLE_WORDS:
            labels[i] = "PERSON"
            j = i + 1
            # skip period after Mr/Ms/Dr
            if j < n and tokens[j] == ".":
                labels[j] = "PERSON"
                j += 1
            while j < n and (pos_tags[j] in ("NNP", "NNPS") or
                             tokens[j].istitle()) and tokens[j].lower() not in ORG_SUFFIXES:
                labels[j] = "PERSON"
                j += 1
            i = j
            continue

        # Organization: words ending with org suffixes
        if tok_lower in ORG_SUFFIXES and i > 0:
            # scan backwards for title-case tokens
            j = i - 1
            while j >= 0 and (tokens[j].istitle() or tokens[j].lower() in ("of", "and", "the", "for")):
                if labels[j] != "O":
                    break
                if tokens[j].lower() == "the":
                    break
                j -= 1
            start = j + 1
            for k in range(start, i + 1):
                if labels[k] == "O":
                    labels[k] = "ORGANIZATION"
            i += 1
            continue

        # Location from gazetteer (multi-word aware)
        if tok_lower in LOCATION_GAZETTEER and labels[i] == "O":
            labels[i] = "LOCATION"
            j = i + 1
            while j < n and tokens[j].lower() in LOCATION_GAZETTEER:
                labels[j] = "LOCATION"
                j += 1
            i = j
            continue

        i += 1

    return labels


# ---------------------------------------------------------------------------
# Approach C: NLTK ne_chunk
# ---------------------------------------------------------------------------

CHUNK_MAP = {
    "PERSON": "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOCATION": "LOCATION",
    "FACILITY": "LOCATION",
}


def predict_nltk_ne_chunk(tokens: List[str]) -> List[str]:
    """Use NLTK ne_chunk with rule-based legal-reference fallback."""
    pos_tagged = nltk.pos_tag(tokens)
    tree = nltk.ne_chunk(pos_tagged)
    labels = ["O"] * len(tokens)
    idx = 0
    for subtree in tree:
        if hasattr(subtree, "label"):
            chunk_label = subtree.label()
            mapped = CHUNK_MAP.get(chunk_label, "O")
            for _ in subtree:
                if idx < len(labels):
                    labels[idx] = mapped
                idx += 1
        else:
            idx += 1

    # Add rule-based legal reference detection
    for i, tok in enumerate(tokens):
        if labels[i] == "O":
            tok_lower = tok.lower()
            if tok_lower in ("section", "article"):
                labels[i] = "LEGAL_REFERENCE"
                j = i + 1
                while j < len(tokens) and (tokens[j].isnumeric() or tokens[j] == "."):
                    labels[j] = "LEGAL_REFERENCE"
                    j += 1
            elif tok_lower == "constitution":
                labels[i] = "LEGAL_REFERENCE"
            elif tok_lower in ("crpc", "ipc"):
                labels[i] = "LEGAL_REFERENCE"
            elif tok_lower == "act" and i > 0 and tokens[i - 1].istitle():
                labels[i] = "LEGAL_REFERENCE"
                j = i - 1
                while j >= 0 and labels[j] == "O" and (tokens[j].istitle() or
                      tokens[j].lower() in ("of", "and")):
                    if tokens[j].lower() == "the":
                        break
                    labels[j] = "LEGAL_REFERENCE"
                    j -= 1

    return labels


# ---------------------------------------------------------------------------
# Approach D: Supervised Logistic Regression
# ---------------------------------------------------------------------------

def train_supervised_model(
    train_data: List[Tuple[List[str], List[str]]]
) -> Tuple[LogisticRegression, DictVectorizer, float]:
    """Train a token-level logistic regression NER model. Returns (model, vectorizer, train_time)."""
    X_dicts: List[Dict] = []
    y_labels: List[str] = []
    for tokens, labels in train_data:
        pos_tags = pos_tag_tokens(tokens)
        for i in range(len(tokens)):
            X_dicts.append(extract_token_features(tokens, pos_tags, i))
            y_labels.append(labels[i])

    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dicts)

    start = time.time()
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED, solver="lbfgs"
    )
    model.fit(X, y_labels)
    train_time = time.time() - start
    return model, vec, train_time


def predict_supervised(
    tokens: List[str], model: LogisticRegression, vec: DictVectorizer
) -> List[str]:
    """Predict labels for tokens using supervised model."""
    pos_tags = pos_tag_tokens(tokens)
    X_dicts = [extract_token_features(tokens, pos_tags, i) for i in range(len(tokens))]
    X = vec.transform(X_dicts)
    return list(model.predict(X))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    true_labels: List[str], pred_labels: List[str]
) -> Dict[str, float]:
    """Compute accuracy and entity-level macro/micro metrics."""
    acc = accuracy_score(true_labels, pred_labels)
    macro_p = precision_score(true_labels, pred_labels, labels=ENTITY_LABELS,
                              average="macro", zero_division=0)
    macro_r = recall_score(true_labels, pred_labels, labels=ENTITY_LABELS,
                           average="macro", zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, labels=ENTITY_LABELS,
                        average="macro", zero_division=0)
    micro_p = precision_score(true_labels, pred_labels, labels=ENTITY_LABELS,
                              average="micro", zero_division=0)
    micro_r = recall_score(true_labels, pred_labels, labels=ENTITY_LABELS,
                           average="micro", zero_division=0)
    micro_f1 = f1_score(true_labels, pred_labels, labels=ENTITY_LABELS,
                        average="micro", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }


def per_entity_metrics(
    true_labels: List[str], pred_labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """Per-entity precision, recall, F1, support for the best model."""
    result = {}
    for label in ENTITY_LABELS:
        binary_true = [1 if t == label else 0 for t in true_labels]
        binary_pred = [1 if p == label else 0 for p in pred_labels]
        support = sum(binary_true)
        p = precision_score(binary_true, binary_pred, zero_division=0)
        r = recall_score(binary_true, binary_pred, zero_division=0)
        f = f1_score(binary_true, binary_pred, zero_division=0)
        result[label] = {"precision": p, "recall": r, "f1": f, "support": support}
    return result


def format_confusion_matrix(true_labels: List[str], pred_labels: List[str]) -> str:
    """Return a readable text confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels, labels=ALL_LABELS)
    # Column widths
    col_w = 8
    header = " " * 16 + "".join(f"{lab:>{col_w}}" for lab in ALL_LABELS)
    lines = [header, "-" * len(header)]
    for i, row_label in enumerate(ALL_LABELS):
        row = f"{row_label:<16}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(ALL_LABELS)))
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Custom text analysis
# ---------------------------------------------------------------------------

def group_entities(tokens: List[str], labels: List[str]) -> List[Tuple[str, str]]:
    """Group consecutive tokens with the same entity label."""
    groups: List[Tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        if labels[i] != "O":
            entity_label = labels[i]
            entity_tokens = [tokens[i]]
            j = i + 1
            while j < len(tokens) and labels[j] == entity_label:
                entity_tokens.append(tokens[j])
                j += 1
            groups.append((" ".join(entity_tokens), entity_label))
            i = j
        else:
            i += 1
    return groups


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    dataset_info: Dict,
    results: Dict,
    best_name: str,
    best_per_entity: Dict,
    cm_text: str,
    custom_text: str,
    custom_tokens: List[str],
    custom_labels: List[str],
    custom_groups: List[Tuple[str, str]],
) -> str:
    """Generate the full output.txt report."""
    sep = "=" * 78
    lines: List[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append(sep)
        lines.append(f"  {title}")
        lines.append(sep)

    # Header
    lines.append(sep)
    lines.append("  Natural Language Processing Laboratory")
    lines.append(sep)
    lines.append("")
    lines.append(f"  Student Name    : Divya M")
    lines.append(f"  Register Number : 24AD0074")
    lines.append(f"  Experiment      : 05")
    lines.append("")
    lines.append(f"  Title: {EXPERIMENT_TITLE}")
    lines.append("")
    lines.append(f"  Aim: {AIM}")
    lines.append("")
    lines.append("  Note: This implementation compares alternative Python/NLTK-based NER")
    lines.append("  algorithms while retaining the prescribed experiment objective. The")
    lines.append("  original title and aim reference Apache OpenNLP; however, as permitted")
    lines.append("  by the staff, the program uses Python-native approaches for comparative")
    lines.append("  analysis of NER techniques on legal text documents.")

    # Dataset summary
    section("Dataset Summary")
    lines.append("")
    lines.append(f"  Total sentences   : {dataset_info['total_sentences']}")
    lines.append(f"  Training sentences: {dataset_info['train_sentences']}")
    lines.append(f"  Testing sentences : {dataset_info['test_sentences']}")
    lines.append(f"  Total tokens      : {dataset_info['total_tokens']}")
    lines.append(f"  Training tokens   : {dataset_info['train_tokens']}")
    lines.append(f"  Testing tokens    : {dataset_info['test_tokens']}")

    section("Entity Counts (Full Dataset)")
    lines.append("")
    for label in ALL_LABELS:
        lines.append(f"  {label:<20}: {dataset_info['entity_counts'].get(label, 0)}")

    # Model descriptions
    section("Model Descriptions")
    lines.append("")
    lines.append("  A. Proper-Noun POS Baseline")
    lines.append("     Tokenize and POS-tag each sentence. Classify consecutive NNP/NNPS")
    lines.append("     tokens as PERSON; all others as O. A deliberately simple baseline.")
    lines.append("")
    lines.append("  B. Rule-Based Legal NER")
    lines.append("     Uses deterministic patterns and embedded gazetteers to detect PERSON,")
    lines.append("     ORGANIZATION, LOCATION, and LEGAL_REFERENCE entities from titles,")
    lines.append("     organization suffixes, location vocabulary, and legal patterns.")
    lines.append("")
    lines.append("  C. NLTK ne_chunk")
    lines.append("     Applies POS tagging followed by nltk.ne_chunk. Maps chunk labels to")
    lines.append("     the target entity types and adds rule-based LEGAL_REFERENCE detection.")
    lines.append("")
    lines.append("  D. Supervised Logistic Regression Token Classifier")
    lines.append("     Trains a token-level multiclass LogisticRegression model on contextual")
    lines.append("     features (token shape, POS, context window, gazetteers). Uses")
    lines.append("     class_weight='balanced', max_iter=1000, random_state=42.")

    # Evaluation table
    section("Evaluation Results")
    lines.append("")
    header_fmt = (
        f"  {'Model':<12} {'Acc':>6} {'MacP':>6} {'MacR':>6} {'MacF1':>6} "
        f"{'MicP':>6} {'MicR':>6} {'MicF1':>6} {'Time(s)':>8}"
    )
    lines.append(header_fmt)
    lines.append("  " + "-" * 74)
    for name in ["A_POS", "B_Rule", "C_NLTK", "D_LR"]:
        r = results[name]
        m = r["metrics"]
        time_str = f"{r['inference_time']:.4f}"
        if "train_time" in r:
            time_str = f"{r['train_time']:.3f}+{r['inference_time']:.3f}"
        lines.append(
            f"  {name:<12} {m['accuracy']:>6.4f} {m['macro_precision']:>6.4f} "
            f"{m['macro_recall']:>6.4f} {m['macro_f1']:>6.4f} "
            f"{m['micro_precision']:>6.4f} {m['micro_recall']:>6.4f} "
            f"{m['micro_f1']:>6.4f} {time_str:>8}"
        )

    section("Best-Performing Model")
    lines.append("")
    lines.append(f"  {best_name}")
    lines.append(f"  Selected by: highest entity-only macro F1 -> micro F1 -> accuracy")

    # Per-entity metrics for best model
    section("Per-Entity Metrics (Best Model)")
    lines.append("")
    lines.append(f"  {'Entity':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("  " + "-" * 62)
    for label in ENTITY_LABELS:
        em = best_per_entity[label]
        lines.append(
            f"  {label:<20} {em['precision']:>10.4f} {em['recall']:>10.4f} "
            f"{em['f1']:>10.4f} {em['support']:>10}"
        )

    # Confusion matrix
    section("Confusion Matrix (Best Model)")
    lines.append("")
    for cm_line in cm_text.split("\n"):
        lines.append(f"  {cm_line}")

    # Custom text
    section("Custom Legal Text Analysis")
    lines.append("")
    lines.append(f"  Input: {custom_text}")
    lines.append("")
    lines.append(f"  {'Token':<35} {'Label':<20}")
    lines.append("  " + "-" * 55)
    for tok, lab in zip(custom_tokens, custom_labels):
        lines.append(f"  {tok:<35} {lab:<20}")
    lines.append("")
    lines.append("  Grouped Entities:")
    lines.append(f"  {'Entity':<40} {'Type':<20}")
    lines.append("  " + "-" * 60)
    for entity_text, entity_type in custom_groups:
        lines.append(f"  {entity_text:<40} -> {entity_type}")

    # Result
    section("Result")
    lines.append("")
    lines.append("  The experiment successfully compared four NER approaches on legal text.")
    lines.append(f"  The best-performing model was {best_name}, which achieved the highest")
    lines.append("  entity-only macro F1-score on the test set. Named entities including")
    lines.append("  PERSON, ORGANIZATION, LOCATION, and LEGAL_REFERENCE were identified")
    lines.append("  in legal text documents as required by the experiment objective.")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 05 - Named Entity Recognition")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Custom legal text for NER")
    args = parser.parse_args()

    ensure_nltk_resources()

    # Build and split dataset
    data = build_dataset()
    train_data, test_data = split_dataset(data, RANDOM_SEED, SPLIT_RATIO)

    all_entity_counts = count_entities(data)
    dataset_info = {
        "total_sentences": len(data),
        "train_sentences": len(train_data),
        "test_sentences": len(test_data),
        "total_tokens": total_tokens(data),
        "train_tokens": total_tokens(train_data),
        "test_tokens": total_tokens(test_data),
        "entity_counts": dict(all_entity_counts),
    }

    # Flatten test labels
    test_true: List[str] = []
    for _, labels in test_data:
        test_true.extend(labels)

    test_sentences_tokens = [tokens for tokens, _ in test_data]

    results: Dict[str, Dict] = {}

    # --- Approach A: POS Baseline ---
    start = time.time()
    a_preds: List[str] = []
    for tokens in test_sentences_tokens:
        a_preds.extend(predict_pos_baseline(tokens))
    a_time = time.time() - start
    results["A_POS"] = {
        "metrics": evaluate_predictions(test_true, a_preds),
        "inference_time": a_time,
        "predictions": a_preds,
    }

    # --- Approach B: Rule-Based ---
    start = time.time()
    b_preds: List[str] = []
    for tokens in test_sentences_tokens:
        b_preds.extend(predict_rule_based(tokens))
    b_time = time.time() - start
    results["B_Rule"] = {
        "metrics": evaluate_predictions(test_true, b_preds),
        "inference_time": b_time,
        "predictions": b_preds,
    }

    # --- Approach C: NLTK ne_chunk ---
    start = time.time()
    c_preds: List[str] = []
    for tokens in test_sentences_tokens:
        c_preds.extend(predict_nltk_ne_chunk(tokens))
    c_time = time.time() - start
    results["C_NLTK"] = {
        "metrics": evaluate_predictions(test_true, c_preds),
        "inference_time": c_time,
        "predictions": c_preds,
    }

    # --- Approach D: Supervised LR ---
    model, vec, train_time = train_supervised_model(train_data)
    start = time.time()
    d_preds: List[str] = []
    for tokens in test_sentences_tokens:
        d_preds.extend(predict_supervised(tokens, model, vec))
    d_time = time.time() - start
    results["D_LR"] = {
        "metrics": evaluate_predictions(test_true, d_preds),
        "inference_time": d_time,
        "train_time": train_time,
        "predictions": d_preds,
    }

    # Select best model
    model_ranking = sorted(
        results.keys(),
        key=lambda k: (
            results[k]["metrics"]["macro_f1"],
            results[k]["metrics"]["micro_f1"],
            results[k]["metrics"]["accuracy"],
        ),
        reverse=True,
    )
    best_name = model_ranking[0]
    best_preds = results[best_name]["predictions"]

    # Per-entity metrics & confusion matrix for best model
    best_per_entity = per_entity_metrics(test_true, best_preds)
    cm_text = format_confusion_matrix(test_true, best_preds)

    # Custom text analysis using best model
    custom_text = args.text
    custom_tokens = nltk.word_tokenize(custom_text)
    if best_name == "A_POS":
        custom_labels = predict_pos_baseline(custom_tokens)
    elif best_name == "B_Rule":
        custom_labels = predict_rule_based(custom_tokens)
    elif best_name == "C_NLTK":
        custom_labels = predict_nltk_ne_chunk(custom_tokens)
    else:
        custom_labels = predict_supervised(custom_tokens, model, vec)
    custom_groups = group_entities(custom_tokens, custom_labels)

    # Generate report
    report = generate_report(
        dataset_info, results, best_name, best_per_entity, cm_text,
        custom_text, custom_tokens, custom_labels, custom_groups,
    )

    # Write output file
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "output.txt"
    output_file.write_text(report, encoding="utf-8")

    # Terminal summary
    print("=" * 60)
    print("  Experiment 05 - Named Entity Recognition")
    print("=" * 60)
    print(f"  Student: Divya M (24AD0074)")
    print()
    print(f"  Dataset: {dataset_info['total_sentences']} sentences, "
          f"{dataset_info['total_tokens']} tokens")
    print(f"  Train  : {dataset_info['train_sentences']} sentences, "
          f"{dataset_info['train_tokens']} tokens")
    print(f"  Test   : {dataset_info['test_sentences']} sentences, "
          f"{dataset_info['test_tokens']} tokens")
    print()
    print(f"  {'Model':<10} {'Acc':>6} {'MacF1':>6} {'MicF1':>6}")
    print("  " + "-" * 34)
    for name in ["A_POS", "B_Rule", "C_NLTK", "D_LR"]:
        m = results[name]["metrics"]
        print(f"  {name:<10} {m['accuracy']:>6.4f} {m['macro_f1']:>6.4f} {m['micro_f1']:>6.4f}")
    print()
    print(f"  Best model: {best_name}")
    print()
    print(f"  Custom text: {custom_text[:60]}...")
    print(f"  Detected entities:")
    for entity_text, entity_type in custom_groups:
        print(f"    {entity_text:<40} -> {entity_type}")
    print()
    print(f"  Output: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
