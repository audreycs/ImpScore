# ImpScore

This is the repository for paper "*ImpScore: A Learnable Metric For Quantifying The Implicitness Level of Language*"

## Metric Model
The trained metric `ImpScore` is available on HuggingFace. [to be done]

## Training Data
The training data consists of 112580 sentence pairs in form of (implicit sentence, explicit sentence). It is available in file `train_data.csv`. In the file, the first row is the header, and each following row consists of two sentence pairs:

positive pair: `(implicit sentence, explicit sentence)`, negative pair: `(implicit sentence, explicit sentence)`

The implicit sentence in the two pairs are the same, as described in the paper.
