# Drift Detector

A framework for testing real-time data drift detection on regression models.

## Setup

```bash
uv sync
```

Generate and split a synthetic dataset:

```bash
uv run generate_data --n-samples 50000 --seed 42
uv run split_data --input data/raw/data.csv --test-size 0.2 --seed 42
```

## Usage

All commands use per-model consolidated YAML configs in `configs/`.

```bash
# Train a model (writes models/{name}.pkl, logs to MLflow)
uv run drift_detector train --config configs/xgboost.yml

# Optimise hyper-parameters with Optuna
uv run drift_detector optimise --config configs/xgboost.yml

# Evaluate a trained model against test data
uv run drift_detector eval --model models/xgboost.pkl --data data/test.csv

# Find the best run from MLflow
uv run drift_detector best --config configs/xgboost.yml

# Launch MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Commands

| Command | Description |
|---|---|
| `train --config <file>` | Train with model-specific config |
| `optimise --config <file>` | Optuna hyper-parameter search |
| `eval --model <pkl> --data <csv>` | Evaluate, compare to train metrics, generate plots |
| `best --config <file>` | Query MLflow for best run |

### Eval options

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | Path to `.pkl` pipeline |
| `--data` | (required) | Path to evaluation CSV |
| `--target` | `target` | Target column name |
| `--output-dir` | `plots` | Output directory for metrics + plots |

### Configs

Per-model YAML files at `configs/{model}.yml` containing:

| Section | Purpose |
|---|---|
| `tracking` | MLflow tracking URI |
| `data` | CSV path and target column |
| `features` | Numeric and categorical feature lists |
| `model` | Model name |
| `params` | Default hyper-parameters |
| `study` | Optuna study name and storage URI |
| `optuna` | Metric, n_trials, n_jobs, search space |

## Dataset

Synthetic regression dataset generated via `src/drift_detector/data/generate.py`.
Features are designed with known correlation strengths to the target, making it
suitable for testing drift detection methods under controlled conditions.

| Feature | Distribution | Correlation with target |
|---|---|---|
| `feature_1` | Gaussian (standardised) | +0.29 |
| `feature_2` | Gaussian (standardised) | +0.22 |
| `feature_3` | Gaussian (standardised) | -0.14 |
| `feature_gaussian` | Gaussian + noise | +0.28 |
| `feature_lognormal` | Log-normal (skewed) | -0.28 |
| `feature_exponential` | Exponential + noise | +0.29 |
| `category` | Categorical (A/B/C/D) | — |
| `type` | Categorical (type_1/2/3) | — |

Target is standardised to zero mean, unit variance.

## Plots

| Correlation Heatmap | Feature Distributions |
|---|---|
| ![Correlation Heatmap](plots/correlation_heatmap.png) | ![Feature Histograms](plots/feature_histograms.png) |

Evaluation plots are saved to `plots/{model_name}/plots/`. The eval command produces actual-vs-predicted scatter, residual histogram, residuals-vs-predicted scatter, and error-by-feature plots.

## To do
- [ ] optuna:
    - [X] Store features used
    - [X] Store more metrics
    - [X] Add plot artifacts to mlflow
    - [ ] Store models
- [ ] Add tests
- [ ] Pydantic validation
- [ ] Update README
- [X] Final retraining pipeline as separate CLI step
