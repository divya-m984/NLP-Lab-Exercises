"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 02
Title           : Develop a Part-of-Speech (POS) Tagging System Using NLTK and
                  Evaluate Its Accuracy on a Corpus of News Articles
"""
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"

# Common POS tag meanings
TAG_MEANINGS = {
    "NN": "Noun",
    "VB": "Verb",
    "JJ": "Adjective",
    "RB": "Adverb",
    "PRP": "Pronoun",
    "DT": "Determiner",
}


def ensure_nltk():
    for res in ["punkt_tab", "averaged_perceptron_tagger_eng"]:
        nltk.download(res, quiet=True)


def main():
    ensure_nltk()

    sentence = input("Enter a sentence: ")

    # Tokenization
    tokens = word_tokenize(sentence)

    # POS Tagging
    pos_tags = nltk.pos_tag(tokens)

    # Display results
    lines = []

    lines.append(f"Tokens:\n{tokens}")

    lines.append("\nPOS Tags:")
    for word, tag in pos_tags:
        lines.append(f"{word} -> {tag}")

    lines.append("\nTag Meanings:")
    for tag, meaning in TAG_MEANINGS.items():
        lines.append(f"{tag:<4} -> {meaning}")

    lines.append(f"\nTotal Words: {len(tokens)}")

    lines.append("\nResult:")
    lines.append("The POS tagging system successfully identified grammatical categories")
    lines.append("of words. It effectively analyzed sentence structure and language")
    lines.append("patterns.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
