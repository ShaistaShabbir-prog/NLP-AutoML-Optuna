"""
Issue #8: Built-in dataset support — AG News, IMDB, 20Newsgroups.
"""
from __future__ import annotations
import logging
from typing import Any

log = logging.getLogger(__name__)

AVAILABLE = ["ag_news", "imdb", "20newsgroups", "sst2"]


def load(name: str, max_samples: int | None = None
         ) -> tuple[list[str], list[int], list[str], list[int]]:
    """
    Load a built-in classification dataset.
    Returns (X_train, y_train, X_test, y_test).
    """
    if name == "ag_news":
        return _ag_news(max_samples)
    if name == "imdb":
        return _imdb(max_samples)
    if name == "20newsgroups":
        return _20newsgroups(max_samples)
    if name == "sst2":
        return _sst2(max_samples)
    raise ValueError(f"Unknown dataset: {name}. Available: {AVAILABLE}")


def _ag_news(n=None):
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("ag_news")
        tr = ds["train"].select(range(min(n or 10000, len(ds["train"]))))
        te = ds["test"].select(range(min(n or 2000, len(ds["test"])) // 5))
        return list(tr["text"]), list(tr["label"]), list(te["text"]), list(te["label"])
    except ImportError:
        return _demo_data("ag_news")


def _imdb(n=None):
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("imdb")
        tr = ds["train"].select(range(min(n or 5000, len(ds["train"]))))
        te = ds["test"].select(range(min(n or 1000, len(ds["test"]))))
        return list(tr["text"]), list(tr["label"]), list(te["text"]), list(te["label"])
    except ImportError:
        return _demo_data("imdb")


def _20newsgroups(n=None):
    from sklearn.datasets import fetch_20newsgroups
    tr = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    te = fetch_20newsgroups(subset="test",  remove=("headers", "footers", "quotes"))
    x_tr, y_tr = tr.data[:n], list(tr.target[:n])
    x_te, y_te = te.data[:n], list(te.target[:n])
    return x_tr, y_tr, x_te, y_te


def _sst2(n=None):
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("sst2")
        tr = ds["train"].select(range(min(n or 5000, len(ds["train"]))))
        te = ds["validation"]
        return list(tr["sentence"]), list(tr["label"]), list(te["sentence"]), list(te["label"])
    except ImportError:
        return _demo_data("sst2")


def _demo_data(name: str):
    """Tiny demo data when datasets library not available."""
    log.warning("datasets not installed — using tiny demo data. pip install datasets")
    texts = [
        "This is a positive example of text classification.",
        "Negative sentiment detected in this sentence.",
        "Neutral factual statement about the topic.",
        "Another example for training the model here.",
    ] * 10
    labels = [0, 1, 2, 1] * 10
    return texts[:30], labels[:30], texts[30:], labels[30:]
