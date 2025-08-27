# NLP AutoML (Optuna + Scikit-learn)

Fast text classification baseline with Optuna.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"

python -m nlp_automl_optuna train --data data/sample.csv --text-col text --label-col label --n-trials 10
python -m nlp_automl_optuna evaluate --model artifacts/model.joblib --data data/sample.csv --text-col text --label-col label
```
