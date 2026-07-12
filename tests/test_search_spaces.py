import pytest

from drift_detector.optimisation.search_spaces import (
    SearchSpaceConfig,
    SearchSpaceEntry,
    get_search_space,
)


class TestSearchSpaceEntry:
    def test_valid_float_entry(self):
        entry = SearchSpaceEntry(type="float", low=0.0, high=1.0)
        assert entry.type == "float"
        assert entry.log is False

    def test_valid_int_entry(self):
        entry = SearchSpaceEntry(type="int", low=1, high=10)
        assert entry.type == "int"

    def test_log_float_entry(self):
        entry = SearchSpaceEntry(type="float", low=0.001, high=1.0, log=True)
        assert entry.log is True

    def test_low_equal_to_high_raises(self):
        with pytest.raises(ValueError, match="low.*must be less than high"):
            SearchSpaceEntry(type="float", low=1.0, high=1.0)

    def test_low_greater_than_high_raises(self):
        with pytest.raises(ValueError, match="low.*must be less than high"):
            SearchSpaceEntry(type="int", low=10, high=1)

    def test_log_with_int_raises(self):
        with pytest.raises(ValueError, match="log scale is not supported"):
            SearchSpaceEntry(type="int", low=1, high=10, log=True)

    def test_invalid_type_raises(self):
        with pytest.raises(Exception):
            SearchSpaceEntry(type="str", low=0, high=1)


class TestSearchSpaceConfig:
    def test_valid_config(self):
        config = SearchSpaceConfig.model_validate(
            {
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "learning_rate": {
                    "type": "float",
                    "low": 0.001,
                    "high": 0.3,
                    "log": True,
                },
            }
        )
        assert "n_estimators" in config.root
        assert "learning_rate" in config.root
        assert config.root["learning_rate"].log is True

    def test_unknown_parameters_still_parse(self):
        config = SearchSpaceConfig.model_validate(
            {
                "alpha": {"type": "float", "low": 0, "high": 1},
                "extra_param": {"type": "float", "low": 0, "high": 1},
            }
        )
        assert "alpha" in config.root
        assert "extra_param" in config.root

    def test_nested_validation_catches_bad_bounds(self):
        with pytest.raises(ValueError, match="low.*must be less than high"):
            SearchSpaceConfig.model_validate(
                {
                    "alpha": {"type": "float", "low": 10, "high": 1},
                }
            )


class _FakeTrial:
    """Minimal optuna Trial stand-in for testing."""

    def __init__(self):
        self._values = {}

    def suggest_float(self, name, low, high, log=False):
        self._values[name] = (low + high) / 2
        return self._values[name]

    def suggest_int(self, name, low, high):
        self._values[name] = (low + high) // 2
        return self._values[name]


class TestGetSearchSpace:
    def test_returns_params(self):
        trial = _FakeTrial()
        config = {
            "alpha": {"type": "float", "low": 0.001, "high": 100, "log": True},
        }
        result = get_search_space("ridge", trial, config)
        assert result == {"alpha": 50.0005}

    def test_unknown_model_raises(self):
        trial = _FakeTrial()
        with pytest.raises(KeyError):
            get_search_space("unknown_model", trial, {})

    def test_invalid_config_raises(self):
        trial = _FakeTrial()
        with pytest.raises(Exception):
            get_search_space("ridge", trial, {"alpha": {"type": "float"}})
