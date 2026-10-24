# Drug review sentiment classification with custom LSTM cells

Advanced NLP lab project (2026/27). We classify drug reviews (UCI Drugs.com dataset)
into 3 classes (rating <=4 negative, 5-6 neutral, >=7 positive) using LSTM cells
implemented from scratch in PyTorch (`nn.Linear` only, no `nn.LSTM`).

## Models
- `vanilla`: standard LSTM (baseline)
- `cifg`: coupled input-forget gate (i = 1 - f)
- `lstmp`: LSTM with projection (hidden 128, projection 64)
- `wmc`: working memory connections (implemented, not part of the final comparison)

## Structure
- `data.py`: loading, labels, stratified splits, vocabulary, EDA (`python data.py`)
- `cells.py`: the four cells and the classifier
- `train.py`: training script (cell, lr, optimizer, batch size, init, seed)
- `evaluate.py`: accuracy, macro F1, confusion matrix
- `results.csv`: all runs

## Reproduce
1. Put `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` in `data/`
2. `pip install -r requirements.txt`
3. `python train.py --cell cifg --lr 1e-3 --opt adam --bs 64 --epochs 8`

## Setup
Embedding 128, hidden 128, vocab 20k, max length 150, dropout 0.3, 8 epochs,
stratified train/val split (90/10) shared by all models, model selected on validation macro F1.

## Results (test set)

| Model | Cell params | Total params | Acc (mean of 2 seeds) | Macro F1 (seed 42 / seed 7) |
|---|---|---|---|---|
| vanilla | 131,584 | 2,691,971 | 0.846 | 0.754 / 0.757 |
| cifg | 98,688 | 2,659,075 | 0.849 | 0.761 / 0.760 |
| lstmp | 107,072 | 2,667,267 | 0.842 (1 seed) | 0.749 |

Hyperparameter study (vanilla, seed 42, macro F1): lr 1e-2 -> 0.553, RMSprop -> 0.746,
batch size 128 -> 0.729, Xavier init -> 0.753 (reference: 0.754).

CIFG is at least as good as the vanilla LSTM with about 25% fewer cell parameters; the gap
(about 0.005 F1) is small and should be read as "comparable", not as a clear win.

## Limitations
- Two seeds for vanilla and CIFG, one for LSTMP; hyperparameter runs use a single seed.
- Fixed 8 epochs: validation F1 was still rising for some runs (e.g. LSTMP, batch size 128).
- The neutral class (9% of the data) is the hardest one (recall around 60-66%).

## Data
Drug Review Dataset (Drugs.com), UCI Machine Learning Repository (Gräßer et al., 2018).