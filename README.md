# ImpScore

This is the repository for paper [ImpScore: A Learnable Metric For Quantifying The Implicitness Level of Sentence](https://openreview.net/forum?id=gYWqxXE5RJ) accepted to ICLR 2025.

## Metric Model
### Usage 1: Download on Huggingface
A trained ImpScore metric is available for download on HuggingFace [[link](https://huggingface.co/audreyeleven/ImpScore)]. Instructions on how to use **ImpScore** is introduced there.

### Usage 2: Local Training
The code for training the metric is in the repository:
```plaintext
├── all_data.csv // training data
├── load_dataset.py
├── train.py  // the training main function
├── model.py // model implementation
└── utils.py 
```

Key packages required to run the code are listed below. Ensure that you have a GPU.
```plaintext
openai==1.34.0
tqdm=4.65.0
pandas=2.1.4
numpy=1.26.4
pytorch=2.3.0
transformers=4.36.2
datasets=2.19.1
numpy=1.26.4
sentence-transformers=3.0.1
```

To run the code, simply enter
```bash
$ python train.py
```

When training is finished, the code automatically generates figures for:
- training loss plot over epochs
- implicitness score distribution on test samples
- pragmatics and implicitness score distribution on test samples

**Feel free to modify or extend the data in `all_data.csv` file and train your own metric.**

#### Training Data

The training data consists of 112580 sentence pairs in form of (implicit sentence, explicit sentence). It is available in file `all_data.csv`. In the file, the first row is the header, and each following row consists of two sentence pairs:

positive pair: `(implicit sentence, explicit sentence)`, negative pair: `(implicit sentence, explicit sentence)`

The implicit sentence in the two pairs are the same, as described in the paper.
