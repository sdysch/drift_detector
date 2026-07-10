"""
Split the raw dataset into train and test sets.
"""

import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def split_dataset(
    df: pd.DataFrame,
    *,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets.

    Args:
        df: Source dataframe to split.
        target_column: Name of the target column (unused in the split itself,
            but kept for future stratification support).
        test_size: Fraction of rows held out for the test set.
        random_state: Seed for reproducibility.

    Returns:
        A ``(train_df, test_df)`` tuple.
    """
    logger.info(
        "Splitting %d rows (test_size=%.0f%%, seed=%d)",
        df.shape[0],
        test_size * 100,
        random_state,
    )

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    logger.info("Train: %d rows, Test: %d rows", train_df.shape[0], test_df.shape[0])

    return train_df, test_df


@click.command()
@click.option(
    "--input",
    "input_path",
    default="data/raw/data.csv",
    type=click.Path(exists=True),
    help="Path to the raw CSV",
)
@click.option(
    "--output-dir",
    default="data",
    type=click.Path(),
    help="Directory to write train.csv and test.csv",
)
@click.option("--test-size", default=0.2, type=float, help="Fraction for test set")
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility")
def main(input_path, output_dir, test_size, seed):
    """Split the raw dataset into train and test CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    train_df, test_df = split_dataset(
        df,
        target_column="target",
        test_size=test_size,
        random_state=seed,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    logger.info(
        "Wrote %d train rows to %s", train_df.shape[0], output_dir / "train.csv"
    )
    logger.info("Wrote %d test rows to %s", test_df.shape[0], output_dir / "test.csv")


if __name__ == "__main__":
    main()
