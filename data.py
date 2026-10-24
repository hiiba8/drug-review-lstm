import os
import re
import pandas as pd
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

SEED = 42


def to_label(r):  # 0 = négatif, 1 = neutre, 2 = positif
    return 0 if r <= 4 else (1 if r <= 6 else 2)


def tok(t):
    t = re.sub(r"&#039;|&quot;|<[^>]+>", " ", t.lower())
    return re.findall(r"[a-z']+", t)


def read_any(data_dir, name):
    for ext in ("tsv", "csv"):
        path = f"{data_dir}/{name}_raw.{ext}"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                sep = "\t" if "\t" in f.readline() else ","
            return pd.read_csv(path, sep=sep)
    raise FileNotFoundError(f"{name}_raw.(tsv|csv) introuvable dans {data_dir}/")


def load_splits(data_dir="data", sample=None, seed=SEED):
    train = read_any(data_dir, "drugsComTrain")
    test = read_any(data_dir, "drugsComTest")
    for df in (train, test):
        df["label"] = df["rating"].apply(to_label)
        df["review"] = df["review"].fillna("")
    train, val = train_test_split(
        train, test_size=0.1, stratify=train["label"], random_state=seed)
    if sample:  # sous-échantillon stratifié (pour les tests rapides)
        def sub(df, n):
            return train_test_split(df, train_size=n, stratify=df["label"],
                                    random_state=seed)[0]
        train, val, test = sub(train, sample), sub(val, sample // 5), sub(test, sample // 5)
    return train, val, test


class DS(Dataset):
    def __init__(self, df, stoi, max_len):
        def enc(t):
            ids = [stoi.get(w, 1) for w in tok(t)][:max_len]
            return ids + [0] * (max_len - len(ids))
        self.x = torch.tensor([enc(t) for t in df["review"]])
        self.y = torch.tensor(df["label"].values)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_loaders(data_dir="data", sample=None, max_len=200, max_vocab=20000,
                bs=64, seed=SEED):
    train, val, test = load_splits(data_dir, sample, seed)
    cnt = Counter(w for t in train["review"] for w in tok(t))
    itos = ["<pad>", "<unk>"] + [w for w, _ in cnt.most_common(max_vocab - 2)]
    stoi = {w: i for i, w in enumerate(itos)}
    mk = lambda df, sh: DataLoader(DS(df, stoi, max_len), bs, shuffle=sh)
    return mk(train, True), mk(val, False), mk(test, False), len(itos)


if __name__ == "__main__":  # petit EDA pour le rapport
    tr, va, te = load_splits()
    print("Tailles:", len(tr), len(va), len(te))
    print("Répartition train (%):\n", (tr["label"].value_counts(normalize=True) * 100).round(1))
    lens = tr["review"].apply(lambda t: len(tok(t)))
    print("Longueur moyenne / médiane / p95:", lens.mean().round(1), lens.median(), lens.quantile(.95))