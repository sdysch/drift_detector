import pytest
import yaml

from drift_detector.utils.config import load_config


@pytest.fixture
def tmp_config_dir(tmp_path):
    def write(name, data):
        path = tmp_path / name
        path.write_text(yaml.dump(data))
        return str(path)

    return write


class TestLoadConfig:
    def test_loads_yaml(self, tmp_config_dir):
        path = tmp_config_dir("linear.yml", {"model": {"name": "linear"}})
        config = load_config(path)
        assert config == {"model": {"name": "linear"}}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yml")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.yml"
        path.write_text("")
        config = load_config(str(path))
        assert config is None
