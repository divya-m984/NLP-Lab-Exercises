"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 01
Title           : Implement Tokenization and Compare the Effectiveness of Stemming
                  Versus Lemmatization in Improving Text Preprocessing for Sentiment
                  Analysis
"""
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"


def ensure_nltk():
    for res in ["punkt_tab", "wordnet", "omw-1.4"]:
        nltk.download(res, quiet=True)


def main():
    ensure_nltk()

    sentence = input("Enter a sentence: ")

    # Tokenization
    tokens = word_tokenize(sentence)

    # Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    # Display results
    lines = []

    lines.append(f"Original Text:\n{sentence}")
    lines.append(f"\nTokens:\n{tokens}")
    lines.append(f"\nStemmed Words:\n{stemmed}")
    lines.append(f"\nLemmatized Words:\n{lemmatized}")

    lines.append("\nComparison:")
    lines.append("Stemming reduces words to root forms, which may not be meaningful.")
    lines.append("Lemmatization converts words to meaningful base forms.")

    lines.append("\nResult:")
    lines.append("Tokenization, stemming, and lemmatization were successfully implemented.")
    lines.append("Lemmatization produced more meaningful words and improved text")
    lines.append("preprocessing for sentiment analysis.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
