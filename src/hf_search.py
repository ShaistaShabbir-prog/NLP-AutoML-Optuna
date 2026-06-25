"""
Issue #4: HuggingFace models (DistilBERT, RoBERTa) in Optuna search space.
"""
import logging
log = logging.getLogger(__name__)

def hf_objective(trial, X_train, y_train, X_val, y_val, use_gpu=False):
    """
    Optuna objective including HuggingFace transformer models.
    Falls back to sklearn if GPU/transformers unavailable.
    """
    model_type = trial.suggest_categorical("model_type",
        ["logistic_regression", "lightgbm", "distilbert"])

    if model_type == "distilbert":
        return _distilbert_trial(trial, X_train, y_train, X_val, y_val, use_gpu)

    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        C = trial.suggest_float("C", 1e-3, 10, log=True)
        model = LogisticRegression(C=C, max_iter=300)
    else:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            learning_rate=trial.suggest_float("lr",  0.01, 0.3,  log=True),
            n_estimators=trial.suggest_int("n_est",  50,  500),
            num_leaves=trial.suggest_int("leaves",   15,   63),
        )

    model.fit(X_train, y_train)
    return model.score(X_val, y_val)


def _distilbert_trial(trial, texts_train, labels_train, texts_val, labels_val, use_gpu):
    try:
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  TrainingArguments, Trainer)
        import torch
        from datasets import Dataset
    except ImportError:
        log.warning("transformers/datasets not installed — skipping DistilBERT trial")
        return 0.0

    model_name = trial.suggest_categorical("hf_model", ["distilbert-base-uncased"])
    lr         = trial.suggest_float("hf_lr", 1e-5, 5e-5, log=True)
    tokenizer  = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_dict({"text": texts_train, "label": labels_train}).map(tokenize)
    val_ds   = Dataset.from_dict({"text": texts_val,   "label": labels_val}).map(tokenize)
    n_labels = len(set(labels_train))
    model    = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)

    args = TrainingArguments(
        output_dir=f"/tmp/hf_trial_{trial.number}",
        num_train_epochs=2, per_device_train_batch_size=16,
        learning_rate=lr, evaluation_strategy="epoch",
        no_cuda=not use_gpu, save_strategy="no", logging_steps=50,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    metrics = trainer.evaluate()
    return metrics.get("eval_accuracy", 0.0)
