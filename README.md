# ImpScore

This is the repository for ICLR 2025 paper [ImpScore: A Learnable Metric For Quantifying The Implicitness Level of Sentence](https://openreview.net/forum?id=gYWqxXE5RJ).

## Package To do list
- :white_check_mark: ~~Make ImpScore a package~~ (current version `0.1.1`).
- :black_square_button: Add functions to enable customized trainig of ImpScore.


## How To Use ImpScore
### Download through pip.
First, directly download it through `pip`.

```bash
$ pip install implicit-score
```

Then, in python code, import this package.
```python
import impscore
```

Use ImpScore to calculate the implicitness score of single English sentence:
```python
# this will download the default model version '0.1.1' into GPU
model = impscore.load_model(load_device="cuda")  # or "cpu"

# sentence list
sentences = ["I have to leave now. Talk to you later.",
             "I can't believe we've talked for so long."]

# calculate implicitness scores, prag_embs, and sem_embs. The returned variables are lists of results.
imp_scores, prag_embs, sem_embs = model.infer_single(sentences)

print(imp_scores)
# The outputs will be a tensor list ([0.6709, 1.0984])
# higher score indicates higher level of implicitness.
```

Use ImpScore to calculate the implicitness of English sentence pairs, so we can compute their pragmatic distance:
```python
model = impscore.load_model(load_device="cuda")

sentence_pairs = [
    ["I have to leave now. Talk to you later.", "I can't believe we've talked for so long."],
    ["You must find a new place and move out by the end of this month.",
     "Maybe exploring other housing options could benefit us both?"]
]

s1_list = [pair[0] for pair in sentence_pairs]  # list of the first sentence in pairs
s2_list = [pair[1] for pair in sentence_pairs]  # list of the second sentence in pairs

# imp_score1 is the implicitness score list for s1 sentences,
# imp_score2 is the implicitness score list for s2 sentences.
# prag_distance is the pragmatic distance list, where prag_distance[i] is the pragmatic distance between s1[i] and s2[i].
imp_score1, imp_score2, prag_distance = model.infer_pairs(s1_list, s2_list)

print(imp_score1, imp_score2, prag_distance)
# the outputs: tensor([0.6709, 0.9273]) tensor([1.0984, 1.3642]) tensor([0.6660, 0.7115])
```
<br>

## How To Train ImpScore
ImpScore is also open for customized training, where you can:
- change the model hyperparameter settings,
- use more training data,
- etc.

I plan to incorporate this feature into the future version of `implicit-score`. 
For now, you can download the source code and data in this repository for training. The data is the same with what we introduced in the paper.

The repository structure:
```plaintext
├── all_data.csv // training data
├── load_dataset.py
├── train.py  // the training main function
├── model.py // model implementation
└── utils.py 
```

> About The Training Data:
> The training data consists of 112,580 sentence pairs in form of (implicit sentence, explicit sentence). In the file, the first row is the header, and each following row consists of two sentence pairs:
> positive pair: `(implicit sentence, explicit sentence)`, negative pair: `(implicit sentence, explicit sentence)`
> The implicit sentence in the two pairs are the same, as described in the paper.

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

When training is completed, the code automatically generates plots for:
- training loss plot over epochs
- implicitness score distribution on test samples
- pragmatics and implicitness score distribution on test samples

**Feel free to modify or extend the data in `all_data.csv` file and train your own metric.**

