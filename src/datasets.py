"""
Issue #8: Built-in dataset support for NLP AutoML experiments.
"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)


def load_dataset(name: str) -> tuple[list[str], list[int], list[str], list[int]]:
    """
    Load a standard NLP classification dataset.
    Returns (X_train, y_train, X_test, y_test).

    Supported: ag_news, imdb, 20newsgroups, sst2
    """
    name = name.lower()
    if name == "20newsgroups":
        return _load_20news()
    if name in ("ag_news", "imdb", "sst2"):
        return _load_hf(name)
    raise ValueError(f"Unknown dataset: {name}. Choices: ag_news, imdb, 20newsgroups, sst2")


def _load_20news() -> tuple:
    from sklearn.datasets import fetch_20newsgroups
    train = fetch_20newsgroups(subset="train", remove=("headers","footers","quotes"))
    test  = fetch_20newsgroups(subset="test",  remove=("headers","footers","quotes"))
    log.info("Loaded 20newsgroups: %d train / %d test", len(train.data), len(test.data))
    return train.data, list(train.target), test.data, list(test.target)


def _load_hf(name: str) -> tuple:
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    mapping = {"ag_news": ("ag_news",None), "imdb": ("imdb",None), "sst2": ("glue","sst2")}
    ds_name, config = mapping[name]
    ds = hf_load(ds_name, config)

    split = "train" if "train" in ds else list(ds.keys())[0]
    train = ds["train"]
    test  = ds.get("test", ds.get("validation", train.select(range(min(2000,len(train))))))

    text_col  = "text" if "text"  in train.features else "sentence"
    label_col = "label" if "label" in train.features else "labels"

    log.info("Loaded %s: %d train / %d test", name, len(train), len(test))
    return (list(train[text_col]), list(train[label_col]),
            list(test[text_col]),  list(test[label_col]))
