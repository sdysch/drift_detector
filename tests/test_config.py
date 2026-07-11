import pytest
import yaml

from drift_detector.utils.config import (
    load_config,
    load_experiment_config,
    merge_configs,
)


@pytest.fixture
def tmp_config_dir(tmp_path):
    def write(name, data):
        path = tmp_path / name
        path.write_text(yaml.dump(data))
        return str(path)

    return write


class TestLoadConfig:
    def test_loads_yaml(self, tmp_config_dir):
        path = tmp_config_dir("train.yml", {"data": {"path": "data.csv"}})
        config = load_config(path)
        assert config == {"data": {"path": "data.csv"}}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yml")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.yml"
        path.write_text("")
        config = load_config(str(path))
        assert config is None


class TestMergeConfigs:
    def test_merges_two_dicts(self):
        result = merge_configs({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_later_overwrites_earlier(self):
        result = merge_configs({"a": 1, "b": 2}, {"b": 3})
        assert result == {"a": 1, "b": 3}

    def test_shallow_merge_nested(self):
        result = merge_configs({"data": {"path": "a"}}, {"data": {"target": "y"}})
        assert result == {"data": {"target": "y"}}

    def test_empty_configs(self):
        assert merge_configs() == {}

    def test_single_config(self):
        assert merge_configs({"a": 1}) == {"a": 1}


class TestLoadExperimentConfig:
    def test_merges_train_and_model(self, tmp_config_dir):
        train_p = tmp_config_dir("train.yml", {"data": {"path": "data.csv"}})
        model_p = tmp_config_dir("model.yml", {"model": {"name": "linear"}})
        result = load_experiment_config(train_p, model_p)
        assert result == {"data": {"path": "data.csv"}, "model": {"name": "linear"}}

    def test_includes_optuna(self, tmp_config_dir):
        train_p = tmp_config_dir("train.yml", {"a": 1})
        model_p = tmp_config_dir("model.yml", {"b": 2})
        optuna_p = tmp_config_dir("optuna.yml", {"c": 3})
        result = load_experiment_config(train_p, model_p, optuna_p)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_missing_train_raises(self, tmp_config_dir):
        model_p = tmp_config_dir("model.yml", {})
        with pytest.raises(FileNotFoundError):
            load_experiment_config("missing.yml", model_p)
