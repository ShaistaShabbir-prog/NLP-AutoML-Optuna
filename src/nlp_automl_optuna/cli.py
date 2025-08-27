from __future__ import annotations
from pathlib import Path
import json
import joblib
import optuna
import pandas as pd
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report

app = typer.Typer(add_completion=False, help="NLP AutoML")

def _load(path: Path, text_col: str, label_col: str):
    df = pd.read_csv(path)
    return df[text_col].astype(str).tolist(), df[label_col].astype(str).tolist()

def _pipeline(c: float, ngram_max: int):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, C=c))
    ])

def _objective(trial, X, y):
    c = trial.suggest_float("C", 0.01, 10.0, log=True)
    ngram_max = trial.suggest_int("ngram_max", 1, 2)
    pipe = _pipeline(c, ngram_max)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean()

@app.command()
def train(data: Path = typer.Option(Path("data/sample.csv"), exists=True),
          text_col: str = typer.Option("text"),
          label_col: str = typer.Option("label"),
          n_trials: int = typer.Option(10),
          out: Path = typer.Option(Path("artifacts"))):
    X, y = _load(data, text_col, label_col)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: _objective(t, X, y), n_trials=n_trials)
    best = study.best_params
    pipe = _pipeline(best["C"], best["ngram_max"])
    pipe.fit(X, y)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    (out / "best_params.json").write_text(json.dumps(best, indent=2))
    typer.echo(f"Saved model to {out/ 'model.joblib'}")

@app.command()
def evaluate(model: Path = typer.Option(Path("artifacts/model.joblib"), exists=True),
             data: Path = typer.Option(Path("data/sample.csv"), exists=True),
             text_col: str = typer.Option("text"),
             label_col: str = typer.Option("label")):
    pipe = joblib.load(model)
    X, y = _load(data, text_col, label_col)
    preds = pipe.predict(X)
    typer.echo(classification_report(y, preds))

if __name__ == "__main__":
    app()
