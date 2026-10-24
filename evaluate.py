import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    tot, preds, ys = 0.0, [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot += loss_fn(out, y).item()
        preds += out.argmax(1).tolist()
        ys += y.tolist()
    return {"loss": tot / len(ys),
            "acc": accuracy_score(ys, preds),
            "f1": f1_score(ys, preds, average="macro"),
            "cm": confusion_matrix(ys, preds)}