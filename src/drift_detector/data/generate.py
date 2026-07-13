"""Generate synthetic regression datasets with optional drift scenarios.

Builds on sklearn's ``make_regression`` output by adding:
    * Categorical features with target effects,
    * Non-linear relationships,
    * Correlated noise features (gaussian, log-normal, exponential).
"""

import logging
from pathlib import Path
from typing import Literal

import click
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

logger = logging.getLogger(__name__)

DriftType = Literal["slow_feature_1", "multi_feature", "concept"]


def _apply_slow_feature_1(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """feature_1 drifts upward over the second half of the dataset."""
    midpoint = df.shape[0] // 2
    ramp = rng.random(df.shape[0] - midpoint) * 5
    df = df.copy()
    df.loc[midpoint:, "feature_1"] += ramp
    return df


def _apply_multi_feature(
    df: pd.DataFrame, rng: np.random.Generator, n_samples: int
) -> pd.DataFrame:
    """feature_1 drifts up, feature_2 down, category distribution shifts."""
    midpoint = n_samples // 2
    half_2 = slice(midpoint, None)
    df = df.copy()
    d1 = rng.uniform(0, 5, n_samples - midpoint)
    d2 = rng.uniform(-3, 0, n_samples - midpoint)
    df.loc[half_2, "feature_1"] += d1
    df.loc[half_2, "feature_2"] += d2
    category_d = rng.choice(
        ["A", "C", "D"], size=n_samples - midpoint, p=[0.4, 0.3, 0.3]
    )
    df.loc[half_2, "category"] = list(category_d)
    return df


def _apply_concept(df: pd.DataFrame, y: pd.Series, n_samples: int) -> pd.Series:
    """feature_1's coefficient flips from +200 to -100 midway."""
    midpoint = n_samples // 2
    y = y.copy()
    old = 200 * df["feature_1"].iloc[midpoint:]
    new = -100 * df["feature_1"].iloc[midpoint:]
    y.iloc[midpoint:] += new - old
    return y


def make_dataset(
    n_samples: int = 10000,
    random_state: int = 42,
    drift: DriftType | None = None,
) -> pd.DataFrame:
    """Generate a synthetic regression dataset, optionally with a
    data-drift scenario in the second half of the rows.

    First half of the rows has the standard distribution without drift,
    second half exhibits the drift pattern (both halves are time-ordered).

    Args:
        n_samples: Number of rows to generate.
        random_state: Seed for NumPy and sklearn.
        drift: ``None`` for clean data; ``\"slow_feature_1\"``,
            ``\"multi_feature\"``, or ``\"concept\"`` for a drift scenario.

    Returns:
        A ``pandas.DataFrame``.
    """
    rng = np.random.default_rng(random_state)

    logger.info(
        "Generating %d synthetic samples (seed=%d, drift=%s)",
        n_samples,
        random_state,
        drift,
    )

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
    y = pd.Series(
        200 * df["feature_1"]
        + 150 * df["feature_2"]
        - 100 * df["feature_3"]
        + rng.standard_normal(n_samples) * 50,
        dtype="float64",
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
    category_effect = {"A": 10, "B": -5, "C": 3, "D": 15}
    y += df["category"].map(category_effect)

    # non-linear relationships
    y += df["feature_1"] ** 2 * 0.5
    y += df["feature_2"] ** 3

    # === correlated noise features ===
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

    # === optional drift scenarios ===
    if drift == "slow_feature_1":
        df = _apply_slow_feature_1(df, rng)
    elif drift == "multi_feature":
        df = _apply_multi_feature(df, rng, n_samples)
    elif drift == "concept":
        y = _apply_concept(df, y, n_samples)

    df["target"] = y
    df["target"] = (df["target"] - df["target"].mean()) / df["target"].std()
    df = df.reset_index(drop=True)

    logger.info(
        "Dataset complete: %d rows, %d columns.",
        df.shape[0],
        df.shape[1],
    )

    return df


def pick_drift(drift: str | None) -> DriftType | None:
    if not drift:
        return None
    valid = ["slow_feature_1", "multi_feature", "concept"]
    if drift not in valid:
        msg = f"Unknown drift '{drift}'. Choose from: {', '.join(valid)}"
        raise click.BadParameter(msg)
    return drift  # type: ignore[return-value]


@click.command()
@click.option(
    "--n-samples",
    default=50_000,
    type=int,
    help="Number of samples to generate.",
)
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
@click.option(
    "--output",
    default="data/raw/data.csv",
    type=click.Path(),
    help="Output CSV path.",
)
@click.option(
    "--drift",
    default=None,
    type=str,
    help="Drift scenario: slow_feature_1 | multi_feature | concept",
)
def main(n_samples, seed, drift, output):
    """Generate synthetic raw data, optionally with a
    data-drift scenario in the second half of the rows."""
    logging.basicConfig(level=logging.ERROR)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df = make_dataset(
        n_samples=n_samples,
        random_state=seed,
        drift=pick_drift(drift),
    )
    df.to_csv(output, index=False)
    logger.info("Wrote %d rows to %s", n_samples, output)


if __name__ == "__main__":
    main()
