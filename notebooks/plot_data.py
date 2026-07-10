from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# %% Load data
df = pd.read_csv("data/raw/data.csv")
df.columns


# %% pairplots
def plot_pairplot(data: pd.DataFrame, figsize: tuple[int, int] = (10, 10)):
    g = sns.PairGrid(data)
    g.fig.set_size_inches(figsize)
    g.map(sns.scatterplot)
    g.fig.savefig(PLOTS_DIR / "pairplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    return g


plot_pairplot(
    df[
        [
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_gaussian",
            "feature_lognormal",
            "feature_exponential",
            "target",
        ]
    ]
)

# %% correlation heatmap
numeric_cols = df.select_dtypes(include="number")
corr = numeric_cols.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Feature Correlation Heatmap")
fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
fig.show()

# %% feature histograms
# feature_cols = [c for c in df.columns if c != "target"]
feature_cols = df.columns
n = len(feature_cols)
ncols = 3
nrows = -(-n // ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3))
axes = axes.flatten()

for ax, col in zip(axes, feature_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(col)

for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle("Feature Distributions", y=1.01)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "feature_histograms.png", dpi=150, bbox_inches="tight")
fig.show()
