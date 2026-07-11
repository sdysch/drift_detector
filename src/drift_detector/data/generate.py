"""
Generate a synthetic regression dataset for this project.

Builds on sklearn's ``make_regression`` output by adding:
    * Categorical features with target effects,
    * Non-linear relationships,
    * Correlated noise features (gaussian, log-normal, exponential).
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
    non-linear terms, and correlated noise features.

    Args:
        n_samples: Number of rows to generate.
        random_state: Seed passed to NumPy RNG and sklearn.

    Returns:
        A ``pandas.DataFrame`` with feature and target columns.
    """
    rng = np.random.default_rng(random_state)

    logger.info("Generating %d synthetic samples (seed=%d)", n_samples, random_state)

    X, _ = make_regression(
        n_samples=n_samples,
        n_features=3,
        n_informative=3,
        noise=1,
        random_state=random_state,
    )

    df = pd.DataFrame(
        X,
        columns=[
            "feature_1",
            "feature_2",
            "feature_3",
        ],
    )

    # build target from explicit coefficients so correlations are controlled
    y = (
        200 * df["feature_1"]
        + 150 * df["feature_2"]
        - 100 * df["feature_3"]
        + rng.standard_normal(n_samples) * 50
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
        "A": 10,
        "B": -5,
        "C": 3,
        "D": 15,
    }

    y += df["category"].map(category_effect)

    # non-linear relationships
    y += df["feature_1"] ** 2 * 0.5
    y += 2.0 / df["feature_2"]

    # === correlated noise features ===
    # Each feature shares a signal with y, plus independent noise,
    # so Pearson r is meaningful but not exactly 1.
    signal = rng.standard_normal(n_samples)

    df["feature_gaussian"] = signal + rng.standard_normal(n_samples)
    y += 300 * signal

    neg_signal = -signal
    df["feature_lognormal"] = (
        rng.lognormal(mean=0.5, sigma=0.5, size=n_samples) + neg_signal
    )
    y += 220 * neg_signal

    df["feature_exponential"] = rng.exponential(scale=1.0, size=n_samples) + signal
    y += 200 * signal

    df["target"] = y

    # scale target to zero mean, unit variance
    df["target"] = (df["target"] - df["target"].mean()) / df["target"].std()

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
    logging.basicConfig(level=logging.INFO)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    make_dataset(n_samples=n_samples, random_state=seed).to_csv(
        output,
        index=False,
    )
    logger.info("Wrote %d rows to %s", n_samples, output)


if __name__ == "__main__":
    main()
