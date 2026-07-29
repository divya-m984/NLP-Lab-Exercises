"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 03
Title           : Explore Various Text Similarity Metrics, Including WordNet-Based
                  Similarity, for Clustering News Headlines into Topics
"""
from pathlib import Path
import nltk
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"


def ensure_nltk():
    for res in ["punkt_tab", "wordnet", "omw-1.4"]:
        nltk.download(res, quiet=True)


def main():
    ensure_nltk()
    from nltk.corpus import wordnet

    # Get headlines from user
    n = int(input("Enter number of headlines: "))
    if n < 2:
        print("Error: At least two headlines are required for clustering.")
        return

    headlines = []
    for i in range(n):
        h = input("Enter headline: ")
        headlines.append(h)

    # Cosine similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(headlines)
    cos_sim = cosine_similarity(tfidf_matrix)

    lines = []
    lines.append("Cosine Similarity Matrix:")
    for row in cos_sim:
        lines.append(str([round(v, 4) for v in row]))

    # K-Means clustering with 2 clusters
    km = KMeans(n_clusters=2, random_state=0, n_init=10)
    clusters = km.fit_predict(tfidf_matrix)

    lines.append("\nHeadline Clusters:")
    for i, h in enumerate(headlines):
        lines.append(f"{h} -> Cluster {clusters[i]}")

    # WordNet similarity
    word1 = input("Enter first word: ")
    word2 = input("Enter second word: ")

    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)

    if synsets1 and synsets2:
        sim = synsets1[0].path_similarity(synsets2[0])
        if sim is not None:
            lines.append(f"\nWordNet Similarity: {sim:.4f}")
        else:
            lines.append("\nSimilarity not found")
    else:
        lines.append("\nSimilarity not found")

    lines.append("\nResult:")
    lines.append("Text similarity and WordNet similarity were successfully calculated.")
    lines.append("Similar news headlines were grouped together using clustering")
    lines.append("techniques.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
