"""
Generate a synthetic regression dataset for this project.
Use sklearn's make_regression function, add:
    * Non-linear relationships,
    * Categorical features
"""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression

logger = logging.getLogger(__name__)


def make_dataset(
    n_samples=10000,
    random_state=42,
):
    """Generate a synthetic regression dataset.

    Builds on sklearn's ``make_regression`` output by adding categorical
    features (``category``, ``type``) with target effects,
    non-linear terms.

    Args:
        n_samples: Number of rows to generate.
        random_state: Seed passed to NumPy RNG and sklearn.

    Returns:
        A ``pandas.DataFrame`` with feature and target columns.
    """
    rng = np.random.default_rng(random_state)

    logger.info("Generating %d synthetic samples (seed=%d)", n_samples, random_state)

    X, y = make_regression(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        noise=20,
        random_state=random_state,
    )

    df = pd.DataFrame(
        X,
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ],
    )

    # categorical features
    df["category"] = rng.choice(
        ["A", "B", "C", "D"],
        size=n_samples,
    )

    df["type"] = rng.choice(
        ["type_1", "type_2", "type_3"],
        size=n_samples,
        p=[0.5, 0.4, 0.1],
    )

    # add categorical effects to target
    category_effect = {
        "A": 20,
        "B": -10,
        "C": 5,
        "D": 30,
    }

    type_effect = {
        "type_1": -20,
        "type_2": 10,
        "type_3": 50,
    }

    y += df["category"].map(category_effect)
    y += df["type"].map(type_effect)

    # non-linear relationships
    y += df["feature_1"] ** 2 * 0.1
    y += 4.0 / df["feature_2"]

    df["target"] = y

    logger.info(
        "Dataset complete: %d rows, %d columns.",
        df.shape[0],
        df.shape[1],
    )

    return df


@click.command()
@click.option(
    "--n-samples", default=50_000, type=int, help="Number of samples to generate"
)
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
@click.option(
    "--output", default="data/raw/data.csv", type=click.Path(), help="Output CSV path"
)
def main(n_samples, seed, output):
    """CLI entry-point for generating the synthetic raw dataset."""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    make_dataset(n_samples=n_samples, random_state=seed).to_csv(
        output,
        index=False,
    )
    click.echo(f"Wrote {n_samples} rows to {output}")


if __name__ == "__main__":
    main()
