"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 08
Title           : Apply Topic Modeling Techniques to Extract Themes from a Collection
                  of Customer Reviews and Visualize the Results Using t-SNE
"""
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"


def main():
    # Get reviews from user
    n = int(input("Enter number of reviews: "))
    if n < 3:
        print("Error: At least three reviews are required for t-SNE.")
        return

    reviews = []
    for i in range(n):
        review = input("Enter review: ")
        reviews.append(review)

    # Topic modeling with LDA
    vectorizer = CountVectorizer(stop_words="english")
    dtm = vectorizer.fit_transform(reviews)
    lda = LatentDirichletAllocation(n_components=2, random_state=42)
    doc_topics = lda.fit_transform(dtm)

    # Display topics
    feature_names = vectorizer.get_feature_names_out()
    lines = []
    lines.append("Topics:\n")
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-5:][::-1]]
        lines.append(f"Topic {idx + 1}")
        lines.append(" ".join(top_words))
        lines.append("")

    # t-SNE visualization
    perplexity = min(5, n - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(doc_topics)

    lines.append("t-SNE Coordinates:")
    for i in range(n):
        lines.append(f"Review {i + 1} : [{coords[i][0]:.4f} {coords[i][1]:.4f}]")

    lines.append("\nResult:")
    lines.append("Topic modeling successfully extracted important themes from customer")
    lines.append("reviews. t-SNE visualization helped represent review clusters and topic")
    lines.append("distributions.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")

    # Show scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c="blue", edgecolors="black")
    for i in range(n):
        plt.annotate(f"R{i + 1}", (coords[i][0], coords[i][1]),
                     textcoords="offset points", xytext=(5, 5))
    plt.title("t-SNE Visualization of Customer Reviews")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()


if __name__ == "__main__":
    main()
