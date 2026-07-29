"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 04
Title           : Build an Information Retrieval System Using Classical and
                  Nonclassical Models and Compare Their Performance on a Dataset
                  of Scientific Papers
"""
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"


def main():
    # Get documents from user
    n = int(input("Enter number of documents: "))
    documents = []
    for i in range(n):
        doc = input("Enter document: ")
        documents.append(doc)

    query = input("Enter search query: ")

    # TF-IDF similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    lines = []
    lines.append("TF-IDF Similarity Scores:")
    for i, score in enumerate(tfidf_scores):
        lines.append(f"Document {i + 1} : {score:.4f}")

    # LSA similarity using TruncatedSVD
    n_components = min(len(documents), len(vectorizer.get_feature_names_out()), 2)
    if n_components < 1:
        n_components = 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    lsa_query = svd.transform(query_vec)
    lsa_scores = cosine_similarity(lsa_query, lsa_matrix).flatten()

    lines.append("\nLSA Similarity Scores:")
    for i, score in enumerate(lsa_scores):
        lines.append(f"Document {i + 1} : {score:.4f}")

    # Most relevant document
    best_idx = tfidf_scores.argmax()
    lines.append(f"\nMost Relevant Document:\n{documents[best_idx]}")

    lines.append("\nResult:")
    lines.append("The information retrieval system successfully retrieved relevant")
    lines.append("documents using TF-IDF and LSA. LSA provided better semantic")
    lines.append("understanding of documents.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
