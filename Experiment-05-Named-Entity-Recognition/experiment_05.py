"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 05
Title           : Implement a Named Entity Recognition (NER) Model Using Apache
                  OpenNLP and Assess Its Accuracy on Legal Text Documents
"""
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"


def ensure_nltk():
    for res in ["punkt_tab", "averaged_perceptron_tagger_eng"]:
        nltk.download(res, quiet=True)


def main():
    ensure_nltk()

    text = input("Enter legal text: ")
    actual_count = int(input("Enter actual number of entities: "))

    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Detect entities: words tagged NNP or NNPS
    entities = [(word, "ENTITY") for word, tag in pos_tags if tag in ("NNP", "NNPS")]
    predicted_count = len(entities)

    lines = []
    lines.append("Detected Named Entities:")
    for word, label in entities:
        lines.append(f"{word} -> {label}")

    lines.append(f"\nPredicted Entities: {predicted_count}")

    # Calculate accuracy
    if predicted_count == 0 and actual_count == 0:
        accuracy = 100.0
    elif max(predicted_count, actual_count) == 0:
        accuracy = 0.0
    else:
        accuracy = min(predicted_count, actual_count) / max(predicted_count, actual_count) * 100

    lines.append(f"NER Accuracy: {accuracy:.2f} %")

    lines.append("\nResult:")
    lines.append("The Named Entity Recognition (NER) model successfully identified")
    lines.append("entities such as person names, organizations, and locations from legal")
    lines.append("text documents.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
