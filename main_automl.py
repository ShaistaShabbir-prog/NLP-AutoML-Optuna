"""
main_automl.py

AutoML pipeline with Optuna hyperparameter optimization for NLP text classification.

Usage:
    python main_automl.py --trials 50 --dataset data/sample.csv --text-col text --label-col label

Outputs:
    models/best_model.pkl   — sklearn model
    results/optuna_study.db — Optuna study (SQLite)
    results/summary.csv     — All trial results
"""
import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_data(path: str, text_col: str, label_col: str):
    """Load and split dataset."""
    df = pd.read_csv(path)
    assert text_col in df.columns, f"Column '{text_col}' not found"
    assert label_col in df.columns, f"Column '{label_col}' not found"
    df = df.dropna(subset=[text_col, label_col])
    X = df[text_col].tolist()
    y = df[label_col].tolist()
    logger.info(f"Loaded {len(X)} samples, {len(set(y))} classes")
    return X, y


def build_pipeline(trial, X_train, y_train, X_test, y_test):
    """Build and evaluate a trial pipeline (vectorizer + classifier)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score

    # ── Vectorizer params ──────────────────────────────────────
    max_features = trial.suggest_int("max_features", 1000, 20000, log=True)
    ngram_max    = trial.suggest_int("ngram_max", 1, 3)
    sublinear_tf = trial.suggest_categorical("sublinear_tf", [True, False])

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        sublinear_tf=sublinear_tf,
        strip_accents="unicode",
    )

    # ── Classifier selection ───────────────────────────────────
    clf_name = trial.suggest_categorical("classifier", ["logistic", "rf", "gbm"])

    if clf_name == "logistic":
        C = trial.suggest_float("lr_C", 1e-3, 100, log=True)
        solver = trial.suggest_categorical("lr_solver", ["liblinear", "lbfgs"])
        clf = LogisticRegression(C=C, solver=solver, max_iter=500)

    elif clf_name == "rf":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 300, step=50)
        max_depth    = trial.suggest_int("rf_max_depth", 3, 20)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)

    else:  # gbm
        n_estimators  = trial.suggest_int("gbm_n_estimators", 50, 200, step=50)
        learning_rate = trial.suggest_float("gbm_lr", 0.01, 0.3, log=True)
        max_depth     = trial.suggest_int("gbm_max_depth", 2, 8)
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    pipe = Pipeline([("vec", vectorizer), ("clf", clf)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    score = f1_score(y_test, preds, average="weighted")
    return score, pipe


def run_optimization(X, y, n_trials: int = 50, storage: str | None = None):
    """Run Optuna hyperparameter optimization."""
    import optuna
    from sklearn.model_selection import train_test_split

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_pipe = [None]
    best_score = [0.0]

    def objective(trial):
        score, pipe = build_pipeline(trial, X_train, y_train, X_test, y_test)
        if score > best_score[0]:
            best_score[0] = score
            best_pipe[0] = pipe
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="nlp_automl",
        storage=storage or "sqlite:///results/optuna_study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study, best_pipe[0], best_score[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",    type=int, default=30)
    parser.add_argument("--dataset",   default="data/sample.csv")
    parser.add_argument("--text-col",  default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--storage",   default=None)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    X, y = load_data(args.dataset, args.text_col, args.label_col)
    logger.info(f"Running {args.trials} Optuna trials...")
    study, best_model, best_score = run_optimization(X, y, args.trials, args.storage)

    # Save best model
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    logger.info(f"Best model saved: models/best_model.pkl (F1={best_score:.4f})")

    # Save trial summary
    df = study.trials_dataframe()
    df.to_csv("results/optuna_trials_summary.csv", index=False)
    logger.info(f"Saved {len(df)} trial results to results/optuna_trials_summary.csv")

    print(f"\nBest F1: {best_score:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
