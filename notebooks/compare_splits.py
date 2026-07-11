from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train["split"] = "train"
test["split"] = "test"
combined = pd.concat([train, test], ignore_index=True)

numeric_cols = combined.select_dtypes(include="number").columns.drop(
    "target", errors="ignore"
)[:6]

n = len(numeric_cols)
ncols = 3
nrows = -(-n // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for ax, col in zip(axes, numeric_cols):
    for label, group in combined.groupby("split"):
        sns.kdeplot(data=group[col], label=label, ax=ax, fill=True, alpha=0.3)
    ax.set_title(col)
    ax.legend()

for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle("Train vs Test Feature Distributions", fontsize=14)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "train_test_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, label in zip(axes, ["train", "test"]):
    df = combined[combined["split"] == label]
    sns.histplot(df["target"], kde=True, ax=ax, bins=60)
    ax.set_title(
        f"{label} target (n={len(df)}, mean={df['target'].mean():.3f}, std={df['target'].std():.3f})"
    )
fig.tight_layout()
fig.savefig(PLOTS_DIR / "train_test_target.png", dpi=150, bbox_inches="tight")
plt.show()

print("=== Target summary ===")
print(f"Train: mean={train['target'].mean():.4f}, std={train['target'].std():.4f}")
print(f"Test:  mean={test['target'].mean():.4f}, std={test['target'].std():.4f}")
