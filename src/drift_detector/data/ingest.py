"""Prepare raw wine data for DVC versioning.

Sklearn datasets are loaded from modules at runtime, not stored on disk.
This script dumps the data to CSV so DVC can track it as a versioned artifact.
"""

"""Prepare raw wine data for DVC versioning.

Sklearn datasets are loaded from modules at runtime, not stored on disk.
This script dumps the data to CSV so DVC can track it as a versioned artifact.
"""

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_wine


def main():
    data = load_wine(as_frame=True)

    df = data.frame

    Path('data/raw').mkdir(parents=True, exist_ok=True)

    df.to_csv('data/raw/wine.csv', index=False)


if __name__ == '__main__':
    main()
