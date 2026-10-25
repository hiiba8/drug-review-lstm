import json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = {"vanilla": "#4C72B0", "cifg": "#DD8452", "lstmp": "#55A868"}

# ---------- parse notebook logs ----------
nb = json.load(open("drug_review_experiments.ipynb"))
text = ""
for cell in nb["cells"]:
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            text += "".join(o["text"])

runs = []
for block in re.split(r"device: cuda\n", text)[1:]:
    m = re.match(r"(\w+): ([\d,]+) param", block)
    if not m:
        continue
    eps = [tuple(map(float, x)) for x in re.findall(
        r"train_loss ([\d.]+) \| val_loss ([\d.]+) val_acc ([\d.]+) val_f1 ([\d.]+)", block)]
    cm = re.search(r"TEST acc [\d.]+ \| F1 macro [\d.]+\n(\[\[.*?\]\])", block, re.S)
    cmat = np.array([[int(v) for v in row.split()] for row in
                     re.findall(r"\[+\s*([\d\s]+?)\]", cm.group(1))]) if cm else None
    runs.append({"cell": m.group(1), "epochs": eps, "cm": cmat})
runs = [r for r in runs if len(r["epochs"]) == 8]  # drop the 1-epoch smoke test

# order in notebook matches results.csv order
d = pd.read_csv("results.csv")
assert len(runs) == len(d), (len(runs), len(d))
for r, (_, row) in zip(runs, d.iterrows()):
    r["run"] = row["run"]

# ---------- Fig 3: architecture comparison ----------
cells = ["vanilla", "cifg", "lstmp"]
names = {"vanilla": "Vanilla LSTM", "cifg": "CIFG", "lstmp": "LSTMP"}
base = d[(d.lr == 0.001) & (d.opt == "adam") & (d.bs == 64) & (d.init == "default")]
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for i, c in enumerate(cells):
    vals = base[base.cell == c].test_f1.values
    ax.bar(i, vals.mean(), color=C[c], width=0.6)
    ax.scatter([i] * len(vals), vals, color="black", zorder=3, s=18)
    p = int(base[base.cell == c].params.iloc[0])
    ax.text(i, vals.mean() + 0.004, f"{vals.mean():.3f}\n(n={len(vals)})", ha="center", fontsize=9)
ax.set_xticks(range(3))
cp = {"vanilla": 131584, "cifg": 98688, "lstmp": 107072}
ax.set_xticklabels([f"{names[c]}\n{cp[c]:,} cell params" for c in cells])
ax.set_ylim(0.72, 0.775)
ax.set_ylabel("Test macro-F1")
ax.set_title("Architecture comparison (bars = mean, dots = seeds)")
plt.tight_layout(); plt.savefig("fig3_architectures.png", dpi=200); plt.close()

# ---------- Fig 4: hyperparameter study ----------
ref = d[d.run == "vanilla_lr0.001_adam_bs64_default"].test_f1.iloc[0]
hp = [("Reference\n(lr 1e-3, Adam,\nbs 64, default)", ref),
      ("lr = 1e-2", d[d.run == "vanilla_lr0.01_adam_bs64_default"].test_f1.iloc[0]),
      ("RMSprop", d[d.run == "vanilla_lr0.001_rmsprop_bs64_default"].test_f1.iloc[0]),
      ("Batch 128", d[d.run == "vanilla_lr0.001_adam_bs128_default"].test_f1.iloc[0]),
      ("Xavier init", d[d.run == "vanilla_lr0.001_adam_bs64_xavier"].test_f1.iloc[0])]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
cols = ["#4C72B0"] + ["#C44E52" if v < ref - 0.01 else "#8C8C8C" for _, v in hp[1:]]
ax.bar(range(len(hp)), [v for _, v in hp], color=cols, width=0.6)
ax.axhline(ref, ls="--", color="black", lw=1, zorder=0)
for i, (_, v) in enumerate(hp):
    delta = "" if i == 0 else f"\n({v - ref:+.3f})"
    ax.text(i, v - 0.008, f"{v:.3f}{delta}", ha="center", va="top", fontsize=9, color="white", fontweight="bold")
ax.set_xticks(range(len(hp))); ax.set_xticklabels([n for n, _ in hp], fontsize=9)
ax.set_ylim(0.5, 0.82); ax.set_ylabel("Test macro-F1")
ax.set_title("Hyperparameter study (vanilla LSTM, seed 42, one change at a time)")
plt.tight_layout(); plt.savefig("fig4_hyperparameters.png", dpi=200); plt.close()

# ---------- Fig 5: confusion matrix (best model: CIFG seed 42) ----------
best = next(r for r in runs if r["run"] == "cifg_lr0.001_adam_bs64_default")
cm = best["cm"]; cmn = cm / cm.sum(1, keepdims=True)
labels = ["Negative", "Neutral", "Positive"]
fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]:,})", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_yticks(range(3)); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("CIFG confusion matrix (row-normalised)")
ax.spines[:].set_visible(True)
plt.tight_layout(); plt.savefig("fig5_confusion_cifg.png", dpi=200); plt.close()

# ---------- Fig 6: training curves ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
for c in cells:
    r = next(r for r in runs if r["run"] == f"{c}_lr0.001_adam_bs64_default")
    e = np.arange(1, 9); arr = np.array(r["epochs"])
    a1.plot(e, arr[:, 0], "--", color=C[c], label=f"{names[c]} train")
    a1.plot(e, arr[:, 1], "-", color=C[c], label=f"{names[c]} val")
    a2.plot(e, arr[:, 3], "-o", color=C[c], ms=3, label=names[c])
a1.set_xlabel("Epoch"); a1.set_ylabel("Cross-entropy loss"); a1.set_title("Loss (train: class-weighted, val: unweighted)", fontsize=9)
a1.legend(fontsize=7, ncol=2)
a2.set_xlabel("Epoch"); a2.set_ylabel("Validation macro-F1"); a2.set_title("Validation macro-F1")
a2.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig6_training_curves.png", dpi=200); plt.close()

# ---------- Tables ----------
cell_params = {"vanilla": 131584, "cifg": 98688, "lstmp": 107072}
rows = []
for c in cells:
    s = base[base.cell == c]
    rows.append({"Model": names[c], "Cell params": f"{cell_params[c]:,}",
                 "Total params": f"{int(s.params.iloc[0]):,}",
                 "Seeds": len(s),
                 "Test acc (mean)": f"{s.test_acc.mean():.3f}",
                 "Test macro-F1 (mean)": f"{s.test_f1.mean():.3f}",
                 "Minutes/run": f"{s.minutes.mean():.1f}"})
t1 = pd.DataFrame(rows)
t2 = pd.DataFrame([{"Change vs reference": n.replace("\n", " "),
                    "Val F1": f"{d[d.test_f1 == v].val_f1.iloc[0]:.3f}",
                    "Test F1": f"{v:.3f}",
                    "Δ Test F1": "—" if i == 0 else f"{v - ref:+.3f}"}
                   for i, (n, v) in enumerate(hp)])
# per-class recall for each architecture
rec = []
for c in cells:
    r = next(r for r in runs if r["run"] == f"{c}_lr0.001_adam_bs64_default")
    rr = np.diag(r["cm"]) / r["cm"].sum(1)
    rec.append({"Model": names[c], **{f"Recall {l}": f"{x:.3f}" for l, x in zip(labels, rr)}})
t3 = pd.DataFrame(rec)
with open("tables.md", "w") as f:
    f.write("## Table 1 – Architectures (lr 1e-3, Adam, batch 64)\n\n" + t1.to_markdown(index=False))
    f.write("\n\n## Table 2 – Hyperparameter study (vanilla, seed 42)\n\n" + t2.to_markdown(index=False))
    f.write("\n\n## Table 3 – Per-class test recall (seed 42)\n\n" + t3.to_markdown(index=False) + "\n")
print(open("tables.md").read())
