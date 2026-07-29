"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 06
Title           : Investigate Different Approaches for Relation Identification in
                  Biomedical Texts and Evaluate Their Precision and Recall
"""
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"

# Keywords indicating a biomedical relation
RELATION_KEYWORDS = ["treats", "reduces", "controls", "helps"]


def ensure_nltk():
    for res in ["punkt_tab"]:
        nltk.download(res, quiet=True)


def main():
    ensure_nltk()

    sentence = input("Enter biomedical sentence: ")
    actual = int(input("Actual Relation (1/0): "))

    # Tokenize and predict
    tokens = word_tokenize(sentence.lower())
    predicted = 1 if any(kw in tokens for kw in RELATION_KEYWORDS) else 0

    # Metrics using sklearn (single sample)
    precision = precision_score([actual], [predicted], zero_division=0)
    recall = recall_score([actual], [predicted], zero_division=0)
    f1 = f1_score([actual], [predicted], zero_division=0)

    lines = []
    lines.append(f"Tokens:\n{tokens}")
    lines.append(f"\nPredicted Relation: {predicted}")
    lines.append(f"Precision: {precision:.4f}")
    lines.append(f"Recall: {recall:.4f}")
    lines.append(f"F1-Score: {f1:.4f}")

    lines.append("\nResult:")
    lines.append("Biomedical relations were successfully identified using a rule-based")
    lines.append("approach. The system evaluated performance using Precision, Recall, and")
    lines.append("F1-Score.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
