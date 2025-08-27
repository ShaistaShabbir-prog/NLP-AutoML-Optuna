# NLP AutoML (Optuna + Scikit-learn)

Fast text classification baseline with Optuna.

## Quickstart
```bash
An AutoML pipeline for NLP text classification, powered by Optuna for hyperparameter optimization.
This project demonstrates how to automatically find the best model and preprocessing configuration for text classification tasks using scikit-learn, LightGBM, and Hugging Face Datasets.

🚀 Features

Load datasets directly from 🤗 Hugging Face Datasets
.

Preprocess text with TF-IDF vectorization.

Train and tune multiple models:

Logistic Regression

Naive Bayes

LightGBM

Hyperparameter optimization with Optuna.

Save and reload the best model + configuration.

CLI interface (typer) for training, evaluation, and prediction.

Extensible design — add new models or datasets with minimal changes.

📂 Repository Structure
nlp-automl-optuna/
├── src/nlp_automl/
│   ├── __init__.py
│   ├── cli.py          # CLI: train, evaluate, predict
│   ├── data.py         # Dataset loaders
│   ├── model.py        # Model definitions (LR, NB, LGBM)
│   ├── automl.py       # Optuna optimization logic
│   └── utils.py        # Helpers (logging, saving, etc.)
├── tests/test_smoke.py # Smoke tests
├── data/               # (Optional) local data storage
├── artifacts/          # Saved models, params, metrics
├── pyproject.toml      # Project + dependency setup
├── README.md           # This file
└── .github/workflows/ci.yml  # CI pipeline (lint + tests)

⚙️ Installation
# Clone the repo
git clone https://github.com/your-username/nlp-automl-optuna.git
cd nlp-automl-optuna

# Create virtual env
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -U pip wheel
pip install -e ".[dev]"

🏃 Usage
Train with AutoML

Run Optuna optimization (default 20 trials):

python -m nlp_automl train --dataset imdb --n-trials 20


This will:

Load the IMDB dataset (sentiment analysis).

Run 20 Optuna trials to find the best preprocessing + model parameters.

Save the best model to artifacts/model.joblib and params to artifacts/best_params.json.

Evaluate Best Model
python -m nlp_automl evaluate


Outputs a classification report (precision, recall, F1, accuracy).

Predict Custom Text
python -m nlp_automl predict --text "This movie was fantastic!"


Example output:

Predicted label: POSITIVE

🔧 How It Works

Dataset loading
Uses Hugging Face datasets or local CSVs. Example: imdb, ag_news.

Preprocessing
Text → numerical features using TfidfVectorizer. Hyperparameters like max_features and ngram_range are tuned.

Model search
Optuna explores:

Which model (LogReg, Naive Bayes, LGBM)

Model parameters (e.g., C for LogReg, alpha for NB, num_leaves for LGBM)

Vectorizer parameters

Optimization
The best trial is selected by validation accuracy.

Persistence
Best model and params are saved to artifacts/ for reuse.

🔄 Reuse & Generalization

This template is designed to be general-purpose:

Swap datasets
In data.py, add a new Hugging Face dataset loader or local CSV loader.

Add new models
In model.py, implement a new class or wrapper and register it in automl.py.

Change optimization target
In automl.py, replace accuracy with F1-score, ROC-AUC, or any custom metric.

Deploy anywhere
Saved models are scikit-learn or LightGBM pipelines → loadable in any Python service or API.

🧪 Testing

Run tests and linting:

pytest -q
ruff check . --fix

🛠️ Roadmap

 Add support for Hugging Face Transformer models (e.g., BERT fine-tuning).

 Support multi-label classification.

 Add Dockerfile + Makefile for reproducible runs.

 Visualization of Optuna optimization history.

📜 License

MIT License — see LICENSE
```