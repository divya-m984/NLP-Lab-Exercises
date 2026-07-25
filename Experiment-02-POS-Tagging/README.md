# Experiment 02 — Part-of-Speech Tagging

## Experiment Title

Develop a Part-of-Speech (POS) Tagging System Using NLTK and Evaluate Its Accuracy on a Corpus of News Articles

---

## Student Details

| Field           | Details    |
|-----------------|------------|
| Student Name    | Divya M    |
| Register Number | 24AD0074   |

---

## Aim

To implement Part-of-Speech tagging using NLTK and identify the
grammatical categories of words in a sentence.

---

## Algorithms Explored

1. **Default Tagger** — assigns the most common tag (NOUN) to every token as a baseline.
2. **Unigram Tagger** — trained on word-tag frequencies from the training corpus, with the Default Tagger as backoff.
3. **Bigram Tagger** — uses the previous word's tag as context, with the Unigram Tagger as backoff.
4. **Trigram Tagger** — uses the two preceding tags as context, with the Bigram Tagger as backoff.
5. **Averaged Perceptron Tagger** — NLTK's pre-trained POS tagger (`nltk.pos_tag`), evaluated without additional training.

All taggers use the **Universal POS tagset** for consistency.

---

## Methodology

1. Load tagged sentences from the Brown corpus (category `news`) with the Universal tagset.
2. Shuffle sentences reproducibly (seed 42) and split into 80% training / 20% testing, keeping full sentences intact.
3. Train the Default, Unigram, Bigram, and Trigram taggers on the training split using a backoff chain (Default -> Unigram -> Bigram -> Trigram).
4. Evaluate the pre-trained Averaged Perceptron Tagger on the same test set without additional training.
5. Flatten actual and predicted tags at the token level and compute: accuracy, macro precision, macro recall, macro F1-score, training time, and tagging time.
6. Select the best tagger by macro F1-score and compute per-tag precision, recall, F1, support, and a confusion matrix.
7. Tag a custom sentence with the best-performing tagger and display readable grammatical categories.

---

## Dataset or Sample Input

- **Corpus**: NLTK Brown Corpus, `news` category
- **Tagset**: Universal POS tagset (NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, CONJ, PRT, `.`, X)
- **Total sentences**: 4623
- **Training sentences**: 3698 (80%)
- **Testing sentences**: 925 (20%)
- **Training tokens**: 80760
- **Testing tokens**: 19794
- **Random seed**: 42
- **Default sentence**: `The government announced a new economic policy after Parliament approved the bill.`

---

## Requirements

```
nltk
numpy
scikit-learn
matplotlib
```

---

## Execution

From the repository root:

```bash
python Experiment-02-POS-Tagging/experiment_02.py
```

With custom text:

```bash
python Experiment-02-POS-Tagging/experiment_02.py \
  --text "Researchers developed a highly efficient language model."
```

---

## Evaluation Criteria

All five taggers are evaluated on the same test sentences using:

- **Accuracy** — fraction of correctly tagged tokens
- **Macro Precision** — unweighted mean of per-tag precision
- **Macro Recall** — unweighted mean of per-tag recall
- **Macro F1-Score** — unweighted mean of per-tag F1 (used to select the best tagger)
- **Training Time** — seconds to train (0 for the pre-trained perceptron tagger)
- **Tagging Time** — seconds to tag the entire test set

For the best-performing tagger, a per-tag breakdown and confusion matrix are also generated.

---

## Output Files

All outputs are saved to the `output/` directory:

| File                                        | Description                                              |
|---------------------------------------------|----------------------------------------------------------|
| [`output.txt`](output/output.txt)           | Full report: student details, dataset info, evaluation table, per-tag metrics, custom sentence analysis, and result |
| [`comparison.csv`](output/comparison.csv)   | Five-tagger comparison (accuracy, precision, recall, F1, times) |
| [`per_tag_metrics.csv`](output/per_tag_metrics.csv) | Per-tag precision, recall, F1, and support for the best tagger |
| [`accuracy_plot.png`](output/accuracy_plot.png)     | Bar chart comparing macro F1-score across all five taggers |
| [`confusion_matrix.png`](output/confusion_matrix.png) | Confusion matrix heatmap for the best-performing tagger  |
| [`terminal-output.png`](output/terminal-output.png)   | Screenshot of the terminal output during execution       |

---

## Evaluation Results

| Method                     | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|----------------------------|----------|-----------------|--------------|----------|
| Default Tagger             | 0.3025   | 0.0252          | 0.0833       | 0.0387   |
| Unigram Tagger             | 0.9271   | 0.9285          | 0.8579       | 0.8815   |
| Bigram Tagger              | 0.9310   | 0.9448          | 0.8570       | 0.8853   |
| Trigram Tagger             | 0.9298   | 0.9408          | 0.8518       | 0.8812   |
| Averaged Perceptron Tagger | 0.9324   | 0.8628          | 0.8453       | 0.8440   |

---

## Observations

- The **Bigram Tagger** achieved the highest macro F1-score (0.8853), making it the best overall approach in this evaluation.
- The **Averaged Perceptron Tagger** achieved the highest raw accuracy (0.9324), but its macro F1-score (0.8440) was lower. Since macro F1 gives equal importance to each POS category regardless of frequency, the Bigram Tagger was selected as the best-performing method.
- The Unigram, Bigram, and Trigram taggers all performed strongly because their backoff chains (Default -> Unigram -> Bigram -> Trigram) prevented unknown words from remaining untagged.
- The Bigram model improved over the Unigram model by using the previous tag as additional context.
- The Trigram Tagger did not improve over the Bigram Tagger, likely because longer contexts were sparser in the training corpus.
- The Default Tagger performed poorly (accuracy 0.3025, macro F1 0.0387) because it assigned NOUN to every token.
- Punctuation, conjunctions, determiners, and pronouns were tagged with very high F1-scores.
- The X category had low recall because it had only 14 test examples.
- Training and tagging times depend on the machine and execution environment.

---

## Result

The **Bigram Tagger** was the best-performing approach according to macro F1-score (0.8853) for this evaluation on the Brown corpus `news` category using the Universal POS tagset. It achieved an accuracy of 0.9310 on the test set of 925 sentences (19794 tokens).

This result is specific to this corpus, train/test split (80/20, seed 42), tagset, and evaluation setup. It should not be generalized to every POS-tagging task, as performance may differ on other corpora, domains, or tagsets.
