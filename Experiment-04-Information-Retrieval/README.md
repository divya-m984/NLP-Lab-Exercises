# Experiment 04 — Information Retrieval

## Experiment Title

Build an Information Retrieval System Using Classical and Nonclassical Models and Compare Their Performance on a Dataset of Scientific Papers

---

## Student Details

| Field           | Details    |
|-----------------|------------|
| Student Name    | Divya M    |
| Register Number | 24AD0074   |

---

## Aim

To implement an information retrieval system using TF-IDF and LSA
techniques and retrieve relevant documents based on a user query.

---

## Algorithms Explored

1. **Boolean Term-Matching Retrieval** — A classical retrieval model that represents documents and queries as sets of terms. A document is considered relevant if it contains at least one query term. Documents are ranked by the count of distinct query terms present.

2. **TF-IDF Vector-Space Retrieval** — Documents and queries are represented as vectors of TF-IDF weights. Relevance is measured by the cosine similarity between the query vector and each document vector, enabling term-frequency and inverse-document-frequency weighting.

3. **BM25 Probabilistic Retrieval** — A probabilistic ranking function (Okapi BM25) that scores documents based on query-term frequency, document length normalisation, and inverse document frequency. It extends simple TF-IDF by incorporating saturation and length-normalisation parameters (k1 and b).

4. **Latent Semantic Analysis (LSA) using TruncatedSVD** — A nonclassical model that applies Truncated Singular Value Decomposition to the TF-IDF matrix, projecting documents and queries into a lower-dimensional latent semantic space. This captures hidden relationships between terms and documents, improving retrieval when vocabulary differs between query and relevant documents.

---

## Methodology

1. **Preprocessing** — Each document abstract and query is lowercased, tokenised, and stripped of English stop words and non-alphabetic tokens using NLTK. The cleaned tokens form the basis for all retrieval models.

2. **Document Indexing** — A TF-IDF matrix is built over the 40-document corpus using scikit-learn's `TfidfVectorizer`. For Boolean retrieval, an inverted index of term presence is maintained. For BM25, term frequencies and document lengths are computed separately. For LSA, `TruncatedSVD` is applied to the TF-IDF matrix to produce a reduced-rank representation.

3. **Query Processing** — Each query undergoes the same preprocessing pipeline. It is then transformed into the appropriate representation for each model: a set of terms (Boolean), a TF-IDF vector (TF-IDF), raw term frequencies (BM25), or a projected latent vector (LSA).

4. **Ranking** — Each model scores every document against the query and returns the top-k (k = 5) documents in descending order of relevance score.

5. **Evaluation** — 10 built-in evaluation queries, each with a known relevant topic, are used to compute Precision@5, Recall@5, Average Precision@5, and Reciprocal Rank per query. These are then aggregated into Mean Precision@5, Mean Recall@5, MAP@5, and Mean Reciprocal Rank (MRR) across all queries.

---

## Dataset or Sample Input

The experiment uses **40 scientific-paper abstracts** embedded directly in `experiment_04.py`. The corpus is balanced across five topics with **8 documents each**:

| Topic              | Documents |
|--------------------|-----------|
| Machine Learning   | 8         |
| Climate Science    | 8         |
| Biomedical         | 8         |
| Renewable Energy   | 8         |
| Astronomy          | 8         |

No external dataset or data directory is required; the corpus and all evaluation queries are self-contained within the script.

**Default custom query:**

```
Deep neural networks for detecting disease from medical images
```

---

## Requirements

```
nltk
numpy
scikit-learn
```

---

## Execution

Run with default settings (processes 10 evaluation queries and the default custom query):

```bash
python Experiment-04-Information-Retrieval/experiment_04.py
```

Run with a custom query:

```bash
python Experiment-04-Information-Retrieval/experiment_04.py \
  --query "Solar energy storage for reliable electricity grids"
```

---

## Evaluation Criteria

| Metric              | Description                                                                                         |
|----------------------|-----------------------------------------------------------------------------------------------------|
| Precision@5          | Fraction of the top-5 retrieved documents that are relevant to the query                           |
| Recall@5             | Fraction of all relevant documents in the corpus that appear in the top-5 results                  |
| Average Precision@5  | Average of precision values computed at each relevant document position within the top-5 results   |
| MAP@5                | Mean Average Precision at 5 — the mean of Average Precision@5 across all 10 evaluation queries    |
| Reciprocal Rank      | The reciprocal of the rank position of the first relevant document retrieved for a query           |
| Mean Reciprocal Rank | The mean of Reciprocal Rank values across all 10 evaluation queries                                |

---

## Output Files

Results are saved to the `output/` directory:

- `output/output.txt` — ranked retrieval results and evaluation metrics
- `output/terminal-output.png` — terminal screenshot of the program run

---

## Observations

| Method   | Precision@5 | Recall@5 | MAP@5  | MRR    |
|----------|-------------|----------|--------|--------|
| Boolean  | 0.6200      | 0.3875   | 0.5920 | 1.0000 |
| TF-IDF   | 0.6200      | 0.3875   | 0.5900 | 1.0000 |
| BM25     | 0.6200      | 0.3875   | 0.5900 | 1.0000 |
| LSA      | 0.7400      | 0.4625   | 0.7090 | 1.0000 |

- LSA obtained the highest Precision@5 (0.7400), Recall@5 (0.4625), and MAP@5 (0.7090) among all four retrieval models.
- All four methods achieved an MRR of 1.0000, meaning each evaluation query retrieved at least one relevant document in the first rank.
- Boolean retrieval achieved a slightly higher MAP@5 (0.5920) than TF-IDF and BM25 (both 0.5900) on this dataset.
- TF-IDF and BM25 produced identical aggregate evaluation scores across all metrics.
- LSA benefited from latent semantic relationships captured by TruncatedSVD, enabling it to retrieve more relevant documents even when there were vocabulary differences between queries and documents.
- The default custom query combines machine-learning and biomedical vocabulary, so results from multiple topics are reasonable.
- Results are specific to the embedded corpus, queries, preprocessing, and relevance definition used in this experiment.

---

## Result

LSA was the best-performing retrieval model according to MAP@5, with a score of 0.7090. It also achieved the highest Precision@5 (0.7400) and Recall@5 (0.4625). Performance may differ on larger or real-world scientific collections.
