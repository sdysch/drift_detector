"""Tests for the preprocessing + model pipeline."""

import pandas as pd
import pytest

from drift_detector.models.pipeline import build_pipeline, create_preprocessor


@pytest.fixture()
def sample_data():
    """Tiny DataFrame matching the real dataset schema."""
    df = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0],
            "feature_2": [4.0, 5.0, 6.0, 7.0],
            "feature_3": [7.0, 8.0, 9.0, 10.0],
            "feature_gaussian": [0.1, 0.2, 0.3, 0.4],
            "feature_lognormal": [0.4, 0.5, 0.6, 0.7],
            "feature_exponential": [0.7, 0.8, 0.9, 1.0],
            "category": ["A", "B", "C", "D"],
            "type": ["type_1", "type_2", "type_3", "type_1"],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    numeric = [
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_gaussian",
        "feature_lognormal",
        "feature_exponential",
    ]
    categorical = ["category", "type"]
    return df, numeric, categorical


def test_preprocessor_passes_all_features(sample_data):
    """All input columns must appear in the preprocessor output."""
    df, numeric, categorical = sample_data
    X = df[numeric + categorical]

    preprocessor = create_preprocessor(numeric, categorical)
    preprocessor.fit(X)

    out_names = list(preprocessor.get_feature_names_out())
    preprocessor.transform(X)

    # One-hot: 4 categories + 3 types = 7
    # Passthrough: 6 numeric
    assert len(out_names) == 7 + 6

    for col in numeric:
        assert f"remainder__{col}" in out_names

    for col in categorical:
        assert any(out.startswith(f"categorical__{col}_") for out in out_names)


def test_preprocessor_output_shape(sample_data):
    """Output should have correct rows and columns."""
    df, numeric, categorical = sample_data
    X = df[numeric + categorical]

    preprocessor = create_preprocessor(numeric, categorical)
    out = preprocessor.fit_transform(X)

    assert out.shape == (4, 13)


def test_full_pipeline_fit_and_predict(sample_data):
    """Full pipeline should fit and produce predictions."""
    df, numeric, categorical = sample_data
    X = df[numeric + categorical]
    y = df["target"]

    pipeline = build_pipeline(
        "linear",
        {},
        numeric,
        categorical,
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert preds.shape == (4,)
