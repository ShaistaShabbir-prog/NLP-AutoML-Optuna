"""
Issue #2: MLflow experiment tracking for all Optuna trials.
"""
import logging, os
log = logging.getLogger(__name__)

def get_mlflow():
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "nlp-automl-optuna"))
        return mlflow
    except ImportError:
        log.warning("mlflow not installed — pip install mlflow"); return None

def log_trial(trial, score: float, model=None, artifact_dir: str | None = None) -> str | None:
    mlflow = get_mlflow()
    if mlflow is None: return None
    with mlflow.start_run(run_name=f"trial_{trial.number}") as run:
        mlflow.log_params(trial.params)
        mlflow.log_metric("score", score)
        mlflow.log_metric("trial_number", trial.number)
        if model:
            try: mlflow.sklearn.log_model(model, "model")
            except Exception as e: log.warning("model logging failed: %s", e)
        return run.info.run_id

def log_best(study, model=None) -> None:
    mlflow = get_mlflow()
    if mlflow is None: return
    with mlflow.start_run(run_name="best_trial"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_score", study.best_value)
        mlflow.log_metric("n_trials", len(study.trials))
        if model: mlflow.sklearn.log_model(model, "best_model")
        print(f"Best trial logged to MLflow. Run: mlflow ui")
