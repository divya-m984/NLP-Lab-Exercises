# NLP Lab Exercises

**Student Name:** Divya M
**Register Number:** 24AD0074

---

## Description

This repository contains laboratory exercises for the Natural Language Processing course. Each experiment investigates a core NLP problem using Python, exploring multiple algorithms and comparing their effectiveness rather than following a single prescribed approach.

---

## Experiments

| No. | Title |
|-----|-------|
| 01 | Implement Tokenization and Compare the Effectiveness of Stemming Versus Lemmatization in Improving Text Preprocessing for Sentiment Analysis |
| 02 | Develop a Part-of-Speech (POS) Tagging System Using NLTK and Evaluate Its Accuracy on a Corpus of News Articles |
| 03 | Explore Various Text Similarity Metrics, Including WordNet-Based Similarity, for Clustering News Headlines into Topics |
| 04 | Build an Information Retrieval System Using Classical and Nonclassical Models and Compare Their Performance on a Dataset of Scientific Papers |
| 05 | Implement a Named Entity Recognition (NER) Model Using Apache OpenNLP and Assess Its Accuracy on Legal Text Documents |
| 06 | Investigate Different Approaches for Relation Identification in Biomedical Texts and Evaluate Their Precision and Recall |
| 07 | Construct a Language Model Using N-Gram Models and Compare Its Performance with a Hidden Markov Model (HMM) on a Corpus of Tweets |
| 08 | Apply Topic Modeling Techniques to Extract Themes from a Collection of Customer Reviews and Visualize the Results Using t-SNE |
| 09 | Develop a Rule-Based Classifier to Categorize Legal Documents into Different Types and Measure Its Accuracy Against a Maximum Entropy Classifier |
| 10 | Utilize Word and Phrase-Based Clustering Algorithms to Identify Patterns in Social Media Conversations and Analyze Their Implications for Marketing Strategies |

---

## Repository Structure

```
NLP-Lab-Exercises/
├── README.md
├── requirements.txt
├── .gitignore
├── Experiment-01-Tokenization-Stemming-Lemmatization/
│   ├── experiment_01.py
│   ├── README.md
│   └── output/
├── Experiment-02-POS-Tagging/
│   ├── experiment_02.py
│   ├── README.md
│   └── output/
├── Experiment-03-Text-Similarity-Headline-Clustering/
│   ├── experiment_03.py
│   ├── README.md
│   └── output/
├── Experiment-04-Information-Retrieval/
│   ├── experiment_04.py
│   ├── README.md
│   └── output/
├── Experiment-05-Named-Entity-Recognition/
│   ├── experiment_05.py
│   ├── README.md
│   └── output/
├── Experiment-06-Biomedical-Relation-Identification/
│   ├── experiment_06.py
│   ├── README.md
│   └── output/
├── Experiment-07-NGram-HMM-Language-Models/
│   ├── experiment_07.py
│   ├── README.md
│   └── output/
├── Experiment-08-Topic-Modeling-TSNE/
│   ├── experiment_08.py
│   ├── README.md
│   └── output/
├── Experiment-09-Legal-Document-Classification/
│   ├── experiment_09.py
│   ├── README.md
│   └── output/
└── Experiment-10-Social-Media-Clustering/
    ├── experiment_10.py
    ├── README.md
    └── output/
```

---

## Setup

**1. Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Download required NLTK data (run once):**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Execution

Each experiment is a self-contained Python script. Run from the experiment directory:

```bash
cd Experiment-01-Tokenization-Stemming-Lemmatization
python experiment_01.py
```

Or from the repository root:

```bash
python Experiment-01-Tokenization-Stemming-Lemmatization/experiment_01.py
```

---

## Output Evidence

Each experiment's `output/` directory will contain:

- `output.txt` — printed results captured from the terminal run
- Terminal screenshots showing program execution and results
- Plots (`.png`) where visualization is part of the experiment (e.g., t-SNE scatter plots, accuracy bar charts)
