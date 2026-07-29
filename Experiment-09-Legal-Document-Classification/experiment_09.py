"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 09
Title           : Develop a Rule-Based Classifier to Categorize Legal Documents into
                  Different Types and Measure Its Accuracy Against a Maximum Entropy
                  Classifier
"""
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"


def rule_classify(text):
    """Rule-based classification of legal documents."""
    lower = text.lower()
    if "contract" in lower:
        return "contract"
    elif "judgment" in lower:
        return "judgment"
    else:
        return "agreement"


def main():
    # Get documents from user
    n = int(input("Enter number of documents: "))
    documents = []
    categories = []
    for i in range(n):
        doc = input("Enter document: ")
        cat = input("Enter category: ")
        documents.append(doc)
        categories.append(cat)

    # Rule-based classification
    rb_predictions = [rule_classify(doc) for doc in documents]
    rb_accuracy = accuracy_score(categories, rb_predictions)

    lines = []
    lines.append(f"Rule-Based Accuracy: {rb_accuracy:.4f}")

    # Maximum Entropy (Logistic Regression)
    unique_cats = set(categories)
    if len(unique_cats) < 2:
        lines.append("Maximum Entropy Accuracy: Cannot train with fewer than 2 unique categories.")
    else:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(documents)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, categories)
        me_predictions = model.predict(X)
        me_accuracy = accuracy_score(categories, me_predictions)
        lines.append(f"Maximum Entropy Accuracy: {me_accuracy:.4f}")

    lines.append("\nResult:")
    lines.append("Legal documents were successfully classified using Rule-Based and")
    lines.append("Maximum Entropy classifiers. The classification accuracies were")
    lines.append("calculated and compared.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
