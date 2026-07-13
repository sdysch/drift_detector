from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns

_OUT = Path("plots")
_OUT.mkdir(parents=True, exist_ok=True)

numeric = [
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_gaussian",
    "feature_lognormal",
    "feature_exponential",
]

# %% read in data
files = {
    "none": "data/raw/data.csv",
    "slow": "data/drift_slow/data.csv",
    "multi": "data/drift_multi/data.csv",
    "concept": "data/drift_concept/data.csv",
}

dfs = {}
for label, path in files.items():
    d = pd.read_csv(path)
    d["drift"] = label
    dfs[label] = d

df = pd.concat(dfs.values(), ignore_index=True)

# %% per-drift t-test against clean dataset
clean = dfs["none"]
print("\n" + "=" * 80)
print("Two-sample t-test")
print("=" * 80)
for label in ["slow", "multi", "concept"]:
    d = dfs[label]
    print(f"\n--- {label} vs none ---")
    for col in numeric + ["target"]:
        stat, pval = ttest_ind(clean[col], d[col], equal_var=False)
        m_c, s_c = clean[col].mean(), clean[col].std()
        m_d, s_d = d[col].mean(), d[col].std()
        flag = " <<<" if pval < 0.001 else ""
        print(
            f"  {col:22s}  "
            f"clean: {m_c:>8.4f} ± {s_c:>7.4f}  "
            f"{label}: {m_d:>8.4f} ± {s_d:>7.4f}  "
            f"t={stat:>8.3f}  p={pval:.2e}{flag}"
        )

# %% feature distribution comparison plots
for label in ["slow", "multi", "concept"]:
    combined = pd.concat(
        [dfs["none"].assign(source="clean"), dfs[label].assign(source=label)],
        ignore_index=True,
    )
    n_cols = 3
    n_rows = (len(numeric) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    for ax, col in zip(axes, numeric):
        sns.histplot(
            data=combined,
            x=col,
            hue="source",
            stat="density",
            common_norm=False,
            alpha=0.5,
            ax=ax,
        )
        ax.set_title(col)
    for ax in axes[len(numeric) :]:
        ax.set_visible(False)
    fig.suptitle(f"Feature distributions: clean vs {label}")
    fig.tight_layout()
    fig.savefig(_OUT / f"features_{label}_vs_clean.png")
    plt.close(fig)

# %% Compare targets
fig, ax = plt.subplots()
sns.histplot(
    data=df,
    x="target",
    hue="drift",
    stat="density",
    common_norm=False,
)
fig.savefig(_OUT / "targets_drift.png")
plt.close(fig)
