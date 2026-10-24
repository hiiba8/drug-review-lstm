import argparse, csv, os, time
import torch
import torch.nn as nn
from data import get_loaders
from cells import build_model
from evaluate import evaluate

p = argparse.ArgumentParser()
p.add_argument("--cell", default="vanilla", choices=["vanilla", "cifg", "lstmp", "wmc"])
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--opt", default="adam", choices=["adam", "sgd", "rmsprop"])
p.add_argument("--bs", type=int, default=64)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--emb", type=int, default=128)
p.add_argument("--hid", type=int, default=128)
p.add_argument("--proj", type=int, default=64)
p.add_argument("--max_len", type=int, default=150)
p.add_argument("--init", default="default", choices=["default", "xavier", "orthogonal"])
p.add_argument("--sample", type=int, default=None)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

torch.manual_seed(a.seed)
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

train_dl, val_dl, test_dl, vocab = get_loaders(
    sample=a.sample, max_len=a.max_len, bs=a.bs, seed=a.seed)
model = build_model(a.cell, vocab, a.emb, a.hid, a.proj)

if a.init != "default":
    f = nn.init.xavier_uniform_ if a.init == "xavier" else nn.init.orthogonal_
    for m in model.cell.modules():
        if isinstance(m, nn.Linear):
            f(m.weight)
model.to(device)

n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
print(f"{a.cell}: {n_params:,} paramètres")

opts = {"adam": lambda ps: torch.optim.Adam(ps, lr=a.lr),
        "sgd": lambda ps: torch.optim.SGD(ps, lr=a.lr, momentum=0.9),
        "rmsprop": lambda ps: torch.optim.RMSprop(ps, lr=a.lr)}
opt = opts[a.opt](model.parameters())
counts = torch.bincount(train_dl.dataset.y, minlength=3).float()
weights = (counts.sum() / (3 * counts)).to(device)
loss_fn = nn.CrossEntropyLoss(weight=weights)

best_f1, best_state, t0 = -1, None, time.time()
for ep in range(a.epochs):
    model.train()
    tot, n = 0.0, 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(y)
        n += len(y)
    v = evaluate(model, val_dl, device)
    print(f"ep{ep+1} train_loss {tot/n:.3f} | val_loss {v['loss']:.3f} "
          f"val_acc {v['acc']:.3f} val_f1 {v['f1']:.3f}")
    if v["f1"] > best_f1:
        best_f1 = v["f1"]
        best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

model.load_state_dict(best_state)
t = evaluate(model, test_dl, device)
print(f"TEST acc {t['acc']:.3f} | F1 macro {t['f1']:.3f}\n{t['cm']}")

name = f"{a.cell}_lr{a.lr}_{a.opt}_bs{a.bs}_{a.init}" + (f"_s{a.seed}" if a.seed != 42 else "")
if not a.sample:  # on ne garde que les vrais runs
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state, f"checkpoints/{name}.pt")
row = {"run": name, "cell": a.cell, "lr": a.lr, "opt": a.opt, "bs": a.bs,
       "init": a.init, "sample": a.sample, "params": n_params,
       "val_f1": round(best_f1, 4), "test_acc": round(t["acc"], 4),
       "test_f1": round(t["f1"], 4), "minutes": round((time.time() - t0) / 60, 1)}
new = not os.path.exists("results.csv")
with open("results.csv", "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=row.keys())
    if new:
        w.writeheader()
    w.writerow(row)