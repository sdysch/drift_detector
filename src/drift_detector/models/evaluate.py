import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from drift_detector.metrics import compute_metrics

sns.set_theme(style="whitegrid")
logger = logging.getLogger(__name__)


def _model_name_from_path(model_path):
    stem = Path(model_path).stem
    return stem.split("_v")[0].replace("_best", "")


def _safe_dir(base, name):
    d = Path(base) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_train_metrics(model_path):
    run_id_path = Path(model_path).with_suffix(".run_id.json")
    if not run_id_path.exists():
        return None
    with open(run_id_path) as f:
        info = json.load(f)
    run_id = info.get("mlflow_run_id")
    if not run_id:
        return None

    import mlflow

    client = mlflow.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception:
        logger.warning("Could not fetch MLflow run %s", run_id)
        return None

    train_metrics = dict(run.data.metrics)
    return train_metrics if train_metrics else None


def _scatter_actual_vs_predicted(y_true, y_pred, output_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=100)
    plt.close(fig)


def _residuals_histogram(y_true, y_pred, output_dir, train_residuals=None):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, kde=True, ax=ax, label="Test", alpha=0.6)
    if train_residuals is not None:
        sns.histplot(train_residuals, kde=True, ax=ax, label="Train", alpha=0.4)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_title(f"Residuals Distribution (test bias={np.mean(residuals):.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "residuals_histogram.png", dpi=100)
    plt.close(fig)


def _residuals_vs_predicted(y_true, y_pred, output_dir):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(output_dir / "residuals_vs_predicted.png", dpi=100)
    plt.close(fig)


def _error_by_feature(X, y_true, y_pred, output_dir, max_features=8):
    error = np.abs(y_true - y_pred)
    numeric_cols = X.select_dtypes(include="number").columns[:max_features]
    n_cols = min(3, len(numeric_cols))
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, numeric_cols):
        ax.scatter(X[col], error, alpha=0.3, s=5)
        ax.set_xlabel(col)
        ax.set_ylabel("|Error|")
    for ax in axes[len(numeric_cols) :]:
        ax.set_visible(False)
    fig.suptitle("Absolute Error by Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "error_by_feature.png", dpi=100)
    plt.close(fig)


def _print_comparison(train_metrics, test_metrics, model_name):
    keys = ["mae", "rmse", "r2", "mape", "wmape", "smape", "bias"]
    print()
    print("=" * 70)
    print(f"  Comparison Report: {model_name}")
    print("=" * 70)
    print(f"  {'Metric':>10s}  {'Train':>12s}  {'Test':>12s}  {'Δ%':>10s}")
    print("-" * 70)
    for key in keys:
        train_val = train_metrics.get(key, float("nan"))
        test_val = test_metrics.get(key, float("nan"))
        diff_pct = (
            (test_val - train_val) / abs(train_val) * 100 if train_val else float("inf")
        )
        flag = " <<<" if abs(diff_pct) > 20 else ""
        print(
            f"  {key:>10s}: {train_val:>12.4f}  {test_val:>12.4f}  {diff_pct:>+9.1f}%{flag}"
        )
    print(
        f"  {'n_samples':>10s}: {train_metrics.get('n_samples', '-'):>12}  {test_metrics['n_samples']:>12}"
    )
    print("=" * 70)
    print()


def _print_single_report(metrics, model_name):
    print()
    print("=" * 60)
    print(f"  Evaluation Report: {model_name}")
    print("=" * 60)
    for key in ["mae", "rmse", "r2", "mape", "wmape", "smape", "bias"]:
        print(f"  {key:>10s}: {metrics[key]:>12.4f}")
    print(f"  {'n_samples':>10s}: {metrics['n_samples']:>12}")
    print("=" * 60)
    print()


def run_evaluation(model_path, X=None, y=None, output_dir="plots"):
    model_path = Path(model_path)
    model_name = _model_name_from_path(model_path)
    out = _safe_dir(output_dir, model_name)
    plots_dir = _safe_dir(out, "plots")

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    if X is None or y is None:
        raise ValueError("X and y must be provided")

    logger.info("Running predictions on %d samples", len(X))
    y_pred = model.predict(X)
    y_true = np.asarray(y).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics = compute_metrics(y_true, y_pred)
    metrics["n_samples"] = len(y_true)

    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    train_metrics = _load_train_metrics(model_path)

    _scatter_actual_vs_predicted(y_true, y_pred, plots_dir)
    _residuals_histogram(y_true, y_pred, plots_dir, train_residuals=None)
    _residuals_vs_predicted(y_true, y_pred, plots_dir)
    _error_by_feature(X, y_true, y_pred, plots_dir)
    logger.info("Plots saved to %s", plots_dir)

    if train_metrics:
        _print_comparison(train_metrics, metrics, model_name)
    else:
        _print_single_report(metrics, model_name)

    return metrics
