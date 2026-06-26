"""
Issue #7: FastAPI model serving endpoint for best AutoML model.
"""
from __future__ import annotations
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    log.warning("fastapi not installed — pip install fastapi uvicorn")


if HAS_FASTAPI:
    app = FastAPI(title="NLP AutoML Serving", version="1.0.0")

    class PredictRequest(BaseModel):
        texts: list[str]

    class PredictResponse(BaseModel):
        predictions: list[dict[str, Any]]
        model_info:  dict[str, Any]

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        """Predict labels and confidence for input texts."""
        model, info = _load_model()
        if model is None:
            return {"predictions": [], "model_info": {"error": "No model found. Run training first."}}
        try:
            probs = model.predict_proba(req.texts)
            preds = [
                {"text": t[:50], "label": int(p.argmax()), "confidence": round(float(p.max()), 3)}
                for t, p in zip(req.texts, probs)
            ]
        except Exception:
            labels = model.predict(req.texts)
            preds  = [{"text": t[:50], "label": int(l), "confidence": None}
                      for t, l in zip(req.texts, labels)]
        return {"predictions": preds, "model_info": info}

    @app.get("/model/info")
    def model_info():
        _, info = _load_model()
        return info

    @app.get("/health")
    def health():
        model, info = _load_model()
        return {"status": "ok" if model else "no_model", "model_info": info}


def _load_model():
    """Load best model from MLflow registry or local file."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        client = mlflow.tracking.MlflowClient()
        runs   = client.search_runs(
            experiment_ids=["0"],
            order_by=["metrics.score DESC"],
            max_results=1,
        )
        if runs:
            run = runs[0]
            model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
            info  = {
                "source": "mlflow",
                "run_id": run.info.run_id,
                "score": run.data.metrics.get("score"),
                "params": run.data.params,
            }
            return model, info
    except Exception as e:
        log.debug("MLflow load failed: %s", e)

    # Fallback: local pickle
    model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
    try:
        import pickle
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model, {"source": "local", "path": model_path}
    except FileNotFoundError:
        return None, {"error": f"No model at {model_path}. Run: python -m nlp_automl_optuna train"}
