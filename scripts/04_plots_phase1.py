"""Step 4: comparison plots, built only from results/*.csv written by scripts/03_train_compare.py."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import RESULTS_DIR, FIGURES_DIR

comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
folds = pd.read_csv(RESULTS_DIR / "cv_fold_metrics.csv")
conf = pd.read_csv(RESULTS_DIR / "student_confusion_counts.csv")
names = comp.model.tolist()
colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"][:len(names)]

# 1. CV macro F1 per model (boxplot over the 5 folds)
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([folds[folds.model == n].macro_f1_arc for n in names], tick_labels=names, showmeans=True)
ax.set_ylabel("CV macro F1 (per fold)"); ax.set_title("GroupKFold (5 folds, by trip) - 80 train trips")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(FIGURES_DIR / "cv_macro_f1.png", dpi=110); plt.close()

# 2. student-test accuracy + macro F1, and log loss
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
x = np.arange(len(names)); w = 0.38
ax[0].bar(x - w / 2, comp.student_top1, w, label="Top-1 accuracy", color="#4c78a8")
ax[0].bar(x + w / 2, comp.student_macro_f1_arc, w, label="Macro F1 (arc level)", color="#f58518")
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=20); ax[0].legend(); ax[0].set_title("student_test (10 trips)")
ax[1].bar(x, comp.student_log_loss, color="#54a24b"); ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=20)
ax[1].set_title("student_test log loss (lower is better)")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "student_metrics.png", dpi=110); plt.close()

# 3. accuracy vs cost
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(comp.train_time_s, comp.student_top1, c=colors, s=80)
for n, xx, yy in zip(names, comp.train_time_s, comp.student_top1):
    ax.annotate(n, (xx, yy), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("training time (s, log)"); ax.set_ylabel("student_test Top-1"); ax.set_title("Accuracy vs training cost")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "accuracy_vs_time.png", dpi=110); plt.close()

# 4. confusion matrices (candidate level: top-1 = predicted road)
fig, axes = plt.subplots(2, 3, figsize=(11, 7))
for ax, (_, r) in zip(axes.ravel(), conf.iterrows()):
    cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["not chosen", "chosen"]); ax.set_yticklabels(["wrong road", "true road"])
    ax.set_title(r.model, fontsize=10)
plt.suptitle("student_test confusion matrices (candidate level, top-1 per GPS point = 'chosen')")
plt.tight_layout(); plt.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=100); plt.close()

# 5. MLP loss curve (only model with epoch-wise history)
p = RESULTS_DIR / "mlp_loss_curve.csv"
if p.exists():
    c = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(c.epoch, c.train_loss); ax[0].set_title("MLP training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(c.epoch, c.val_score); ax[1].set_title("MLP validation accuracy (internal 10% split)"); ax[1].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "mlp_loss_curve.png", dpi=110); plt.close()
print("plots saved")
