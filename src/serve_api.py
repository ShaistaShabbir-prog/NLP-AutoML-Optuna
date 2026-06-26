"""
Issue #7: FastAPI endpoint for serving the best AutoML model.
"""
from __future__ import annotations
import logging, os, pickle
from pathlib import Path
log = logging.getLogger(__name__)

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")


def load_best_model():
    p = Path(MODEL_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run training first.")
    with open(p, "rb") as f:
        return pickle.load(f)


if HAS_FASTAPI:
    app = FastAPI(title="NLP AutoML Serving API", version="1.0")
    _model = None

    class PredictRequest(BaseModel):
        texts: list[str]

    @app.on_event("startup")
    def _load():
        global _model
        try: _model = load_best_model()
        except Exception as e: log.warning("Model not loaded: %s", e)

    @app.post("/predict")
    def predict(req: PredictRequest):
        if _model is None:
            return {"error": "Model not loaded. Run training first."}
        try:
            probs  = _model.predict_proba(req.texts)
            preds  = probs.argmax(axis=1).tolist()
            confs  = probs.max(axis=1).tolist()
            return {"predictions": [{"label": p, "confidence": round(c, 4)}
                                     for p, c in zip(preds, confs)]}
        except Exception as e:
            return {"error": str(e)}

    @app.get("/model/info")
    def model_info():
        if _model is None:
            return {"status": "not_loaded"}
        return {"status": "loaded", "type": type(_model).__name__, "path": MODEL_PATH}

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": _model is not None}
