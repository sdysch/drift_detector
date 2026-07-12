import pytest
import yaml

from drift_detector.utils.config import Config, load_config


@pytest.fixture
def tmp_config_dir(tmp_path):
    def write(name, data):
        path = tmp_path / name
        path.write_text(yaml.dump(data))
        return str(path)

    return write


@pytest.fixture
def minimal_config_data():
    return {
        "data": {"path": "data/train.csv", "target": "target"},
        "model": {"name": "linear"},
    }


class TestLoadConfig:
    def test_loads_yaml(self, tmp_config_dir, minimal_config_data):
        path = tmp_config_dir("linear.yml", minimal_config_data)
        config = load_config(path)
        assert isinstance(config, Config)
        assert config.model.name == "linear"
        assert config.data.target == "target"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yml")

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.yml"
        path.write_text("")
        with pytest.raises(Exception):
            load_config(str(path))


class TestConfigDefaults:
    def test_tracking_defaults(self, minimal_config_data):
        config = Config.model_validate(minimal_config_data)
        assert config.tracking.uri == "mlruns"

    def test_random_state_default(self, minimal_config_data):
        config = Config.model_validate(minimal_config_data)
        assert config.random_state == 42

    def test_params_defaults_empty(self, minimal_config_data):
        config = Config.model_validate(minimal_config_data)
        assert config.params == {}

    def test_optional_sections_default_none(self, minimal_config_data):
        config = Config.model_validate(minimal_config_data)
        assert config.study is None
        assert config.optuna is None
        assert config.save_name_suffix is None
        assert config.metric is None


class TestConfigValidation:
    def test_invalid_model_name_raises(self):
        with pytest.raises(Exception):
            Config.model_validate(
                {
                    "data": {"path": "data/train.csv", "target": "y"},
                    "model": {"name": "invalid_model"},
                }
            )

    def test_missing_data_raises(self):
        with pytest.raises(Exception):
            Config.model_validate({"model": {"name": "linear"}})

    def test_nonexistent_data_path_raises(self):
        with pytest.raises(ValueError, match="data path does not exist"):
            Config.model_validate(
                {
                    "data": {"path": "nonexistent/data.csv", "target": "y"},
                    "model": {"name": "linear"},
                }
            )

    def test_missing_model_raises(self):
        with pytest.raises(Exception):
            Config.model_validate({"data": {"path": "data/train.csv", "target": "y"}})

    def test_all_model_names_accepted(self):
        for name in ["linear", "ridge", "random_forest", "xgboost"]:
            config = Config.model_validate(
                {
                    "data": {"path": "data/train.csv", "target": "y"},
                    "model": {"name": name},
                }
            )
            assert config.model.name == name


class TestResolvedMetric:
    def test_top_level_metric(self, minimal_config_data):
        minimal_config_data["metric"] = "mae"
        config = Config.model_validate(minimal_config_data)
        assert config.resolved_metric == "mae"

    def test_optuna_metric_fallback(self, minimal_config_data):
        minimal_config_data["optuna"] = {"metric": "r2", "n_trials": 10}
        config = Config.model_validate(minimal_config_data)
        assert config.resolved_metric == "r2"

    def test_default_metric(self, minimal_config_data):
        config = Config.model_validate(minimal_config_data)
        assert config.resolved_metric == "rmse"

    def test_top_level_overrides_optuna(self, minimal_config_data):
        minimal_config_data["metric"] = "mae"
        minimal_config_data["optuna"] = {"metric": "r2", "n_trials": 10}
        config = Config.model_validate(minimal_config_data)
        assert config.resolved_metric == "mae"


class TestConfigSerialization:
    def test_model_dump(self, minimal_config_data):
        config = Config.model_validate(minimal_config_data)
        d = config.model_dump()
        assert isinstance(d, dict)
        assert d["model"]["name"] == "linear"

    def test_model_dump_serializable(self, minimal_config_data):
        import json

        config = Config.model_validate(minimal_config_data)
        d = config.model_dump_serializable()
        serialized = json.dumps(d)
        assert "linear" in serialized
