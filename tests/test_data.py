import pandas as pd
import pytest

from drift_detector.data.generate import make_dataset
from drift_detector.data.load import load_training_data
from drift_detector.data.split import split_dataset


class TestMakeDataset:
    def test_default_shape(self):
        df = make_dataset(n_samples=100, random_state=42)
        assert df.shape == (100, 9)

    def test_columns(self):
        df = make_dataset(n_samples=10)
        expected = [
            "feature_1",
            "feature_2",
            "feature_3",
            "category",
            "type",
            "feature_gaussian",
            "feature_lognormal",
            "feature_exponential",
            "target",
        ]
        assert list(df.columns) == expected

    def test_deterministic(self):
        df1 = make_dataset(n_samples=50, random_state=1)
        df2 = make_dataset(n_samples=50, random_state=1)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = make_dataset(n_samples=50, random_state=1)
        df2 = make_dataset(n_samples=50, random_state=2)
        assert not df1.equals(df2)


class TestSplitDataset:
    def test_train_test_shapes(self):
        df = pd.DataFrame(
            {"a": range(100), "b": range(100, 200), "target": range(200, 300)}
        )
        train, test = split_dataset(
            df, target_column="target", test_size=0.2, random_state=42
        )
        assert train.shape[0] == 80
        assert test.shape[0] == 20
        assert train.shape[1] == test.shape[1] == 3

    def test_reproducible(self):
        df = pd.DataFrame({"a": range(100), "target": range(100)})
        train1, test1 = split_dataset(
            df, target_column="target", test_size=0.3, random_state=7
        )
        train2, test2 = split_dataset(
            df, target_column="target", test_size=0.3, random_state=7
        )
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_sums_to_total(self):
        df = pd.DataFrame({"a": range(200), "target": range(200)})
        train, test = split_dataset(
            df, target_column="target", test_size=0.25, random_state=0
        )
        assert len(train) + len(test) == 200


class TestLoadTrainingData:
    def test_loads_correct_columns(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame(
            {
                "feature_1": [1, 2, 3],
                "feature_2": [4, 5, 6],
                "category": ["A", "B", "A"],
                "target": [10, 20, 30],
                "extra_col": [0, 0, 0],
            }
        )
        df.to_csv(csv_path, index=False)

        config = {
            "data": {"path": str(csv_path), "target": "target"},
            "features": {
                "numeric": ["feature_1", "feature_2"],
                "categorical": ["category"],
            },
        }
        X, y = load_training_data(config)
        assert X.shape == (3, 3)
        assert list(X.columns) == ["feature_1", "feature_2", "category"]
        assert list(y) == [10, 20, 30]

    def test_missing_feature_raises(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"feature_1": [1], "target": [10]}).to_csv(csv_path, index=False)
        config = {
            "data": {"path": str(csv_path), "target": "target"},
            "features": {"numeric": ["feature_1", "missing_col"], "categorical": []},
        }
        with pytest.raises(KeyError):
            load_training_data(config)
