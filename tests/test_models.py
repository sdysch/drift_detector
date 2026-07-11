import pytest

from drift_detector.models.models import MODELS, ModelSpec, build_model


class TestModelSpec:
    def test_is_frozen(self):
        spec = ModelSpec("test", int)
        with pytest.raises(AttributeError):
            spec.name = "other"

    def test_registry_has_all_models(self):
        assert set(MODELS.keys()) == {"linear", "random_forest", "xgboost", "ridge"}

    def test_registry_models_are_estimators(self):
        from sklearn.base import BaseEstimator

        for spec in MODELS.values():
            assert issubclass(spec.constructor, BaseEstimator)


class TestBuildModel:
    def test_build_linear(self):
        model = build_model("linear", {})
        from sklearn.linear_model import LinearRegression

        assert isinstance(model, LinearRegression)

    def test_build_with_params(self):
        model = build_model("ridge", {"alpha": 2.5})
        assert model.alpha == 2.5

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent", {})

    def test_invalid_param_raises_on_build(self):
        with pytest.raises(TypeError):
            build_model("linear", {"nonexistent_param": 42})
