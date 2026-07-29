"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 10
Title           : Utilize Word and Phrase-Based Clustering Algorithms to Identify Patterns
                  in Social Media Conversations and Analyze Their Implications for
                  Marketing Strategies
"""
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"


def main():
    # Get posts from user
    n = int(input("Enter number of posts: "))
    if n < 2:
        print("Error: At least two posts are required for clustering.")
        return

    posts = []
    for i in range(n):
        post = input("Enter post: ")
        posts.append(post)

    k = int(input("Enter number of clusters: "))
    if k < 1 or k > n:
        print(f"Error: Number of clusters must be between 1 and {n}.")
        return

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(posts)

    if len(vectorizer.get_feature_names_out()) == 0:
        print("Error: No meaningful words found after removing stop words.")
        return

    # K-Means clustering
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)

    lines = []
    lines.append("Cluster Results:\n")
    for i, post in enumerate(posts):
        lines.append(f"Post: {post}")
        lines.append(f"Cluster: {labels[i]}")
        lines.append("")

    # Important keywords per cluster
    feature_names = vectorizer.get_feature_names_out()
    lines.append("Important Keywords:\n")
    for cid in range(k):
        mask = labels == cid
        if not mask.any():
            continue
        mean_vec = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
        top_idx = mean_vec.argsort()[-5:][::-1]
        top_terms = [feature_names[i] for i in top_idx if mean_vec[i] > 0]
        lines.append(f"Cluster {cid}")
        lines.append(" ".join(top_terms))
        lines.append("")

    lines.append("Marketing Insight:")
    lines.append("Similar customer opinions are grouped together.")
    lines.append("Clusters help identify product trends and issues.")

    lines.append("\nResult:")
    lines.append("Social media posts were successfully clustered using TF-IDF and")
    lines.append("K-Means. The clusters revealed customer interests, trends, and marketing")
    lines.append("opportunities.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
