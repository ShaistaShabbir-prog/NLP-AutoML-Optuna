"""
Issue #3: SHAP explanations for the best AutoML model.
"""
import logging, os
log = logging.getLogger(__name__)

def explain(model, X_train, X_test, feature_names=None, output_dir="reports/"):
    """Generate SHAP beeswarm + bar plots for the best model."""
    os.makedirs(output_dir, exist_ok=True)
    try:
        import shap, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("pip install shap matplotlib"); return None

    try:
        explainer   = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
    except Exception:
        explainer   = shap.KernelExplainer(model.predict_proba if hasattr(model,"predict_proba")
                                           else model.predict, shap.sample(X_train, 50))
        shap_values = explainer.shap_values(X_test[:50])

    paths = {}
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        p = os.path.join(output_dir, "shap_beeswarm.png")
        plt.savefig(p, bbox_inches="tight", dpi=120); plt.close(); paths["beeswarm"] = p
    except Exception as e: log.warning("beeswarm failed: %s", e)

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.bar(shap_values, show=False)
        p = os.path.join(output_dir, "shap_bar.png")
        plt.savefig(p, bbox_inches="tight", dpi=120); plt.close(); paths["bar"] = p
    except Exception as e: log.warning("bar failed: %s", e)

    log.info("SHAP plots saved: %s", paths)
    return paths
