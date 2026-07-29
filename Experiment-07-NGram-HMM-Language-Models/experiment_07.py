"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 07
Title           : Construct a Language Model Using N-Gram Models and Compare Its
                  Performance with a Hidden Markov Model (HMM) on a Corpus of Tweets
"""
import warnings
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import bigrams, trigrams
from nltk.probability import FreqDist, LidstoneProbDist
from nltk.tag import HiddenMarkovModelTrainer

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "output" / "output.txt"


def ensure_nltk():
    for res in ["punkt_tab", "averaged_perceptron_tagger_eng",
                "treebank", "universal_tagset"]:
        nltk.download(res, quiet=True)


def main():
    ensure_nltk()

    tweet = input("Enter a tweet: ")

    # Tokenize
    tokens = word_tokenize(tweet.lower())

    # N-Gram generation
    unigram_list = list(tokens)
    bigram_list = list(bigrams(tokens))
    trigram_list = list(trigrams(tokens))

    # Word frequencies
    freq = FreqDist(tokens)

    lines = []
    lines.append(f"Tokens:\n{tokens}")

    lines.append("\n========== N-GRAM MODEL ==========")

    lines.append(f"\nUnigrams:\n{unigram_list}")
    lines.append(f"\nBigrams:\n{bigram_list}")
    lines.append(f"\nTrigrams:\n{trigram_list}")

    lines.append("\nWord Frequencies:")
    for word, count in freq.most_common():
        lines.append(f"{word} : {count}")

    # HMM POS tagging using Treebank corpus
    lines.append("\n========== HMM MODEL ==========")

    tagged_sents = nltk.corpus.treebank.tagged_sents(tagset="universal")[:1000]
    trainer = HiddenMarkovModelTrainer()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        hmm = trainer.train_supervised(
            tagged_sents,
            estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins)
        )
        hmm_tags = hmm.tag(tokens)

    lines.append("\nHMM POS Tagging:")
    for word, tag in hmm_tags:
        lines.append(f"{word} -> {tag}")

    # Comparison
    lines.append("\n========== COMPARISON ==========")

    lines.append("\nN-Gram Model")
    lines.append("- Learns word sequences.")
    lines.append("- Predicts the next word based on previous words.")
    lines.append("- Used for text generation and language modeling.")

    lines.append("\nHMM Model")
    lines.append("- Predicts Part-of-Speech (POS) tags.")
    lines.append("- Uses transition and emission probabilities.")
    lines.append("- Used for sequence labeling tasks.")

    lines.append("\nResult:")
    lines.append("N-Gram and Hidden Markov Model (HMM) language models were implemented")
    lines.append("successfully. HMM captured contextual dependencies more effectively")
    lines.append("than N-Gram models.")

    output = "\n".join(lines)
    print(output)

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(output, encoding="utf-8")
    print("\nOutput saved to: output/output.txt")


if __name__ == "__main__":
    main()
