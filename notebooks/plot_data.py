import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Load data
df = pd.read_csv("data/raw/data.csv")
df.columns

# pairplots
sns.pairplot(
    data=df,
)
plt.show()
