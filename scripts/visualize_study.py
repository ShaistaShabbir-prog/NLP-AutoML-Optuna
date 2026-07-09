"""
scripts/visualize_study.py

Generate Optuna study visualizations after hyperparameter optimization.

Usage:
    python scripts/visualize_study.py --study-name nlp_automl --storage sqlite:///optuna.db

Outputs (in results/ directory):
    optimization_history.html   — trial scores over time
    param_importances.html      — which hyperparameters matter most
    parallel_coordinate.html    — all trials as parallel lines
    contour.html                — 2D contour of top parameter pairs
    summary.csv                 — all trials as CSV

Research note:
    Hyperparameter importance (fANOVA analysis) shows which parameters
    most affect model performance — a publishable finding in itself.
    See: Hutter et al. (2014) "An Efficient Approach for Assessing Hyperparameter Importance"
"""
import os
import argparse
import optuna
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="nlp_automl")
    parser.add_argument("--storage",    default="sqlite:///optuna.db")
    parser.add_argument("--out-dir",    default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    except Exception as e:
        print(f"Could not load study: {e}")
        print("Run the AutoML optimization first to create a study.")
        return

    trials = study.trials
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Study: {args.study_name}")
    print(f"Trials: {len(trials)} total, {len(complete)} complete")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # ── 1. Optimization history ──
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"{args.out_dir}/optimization_history.html")
        print(f"✅ optimization_history.html")
    except Exception as e:
        print(f"⚠️  optimization_history: {e}")

    # ── 2. Parameter importances (fANOVA) ──
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"{args.out_dir}/param_importances.html")
        print(f"✅ param_importances.html — key research finding!")
    except Exception as e:
        print(f"⚠️  param_importances: {e}")

    # ── 3. Parallel coordinate plot ──
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(f"{args.out_dir}/parallel_coordinate.html")
        print(f"✅ parallel_coordinate.html")
    except Exception as e:
        print(f"⚠️  parallel_coordinate: {e}")

    # ── 4. Contour plot (top 2 params) ──
    try:
        importances = optuna.importance.get_param_importances(study)
        top2 = list(importances.keys())[:2]
        if len(top2) == 2:
            fig = optuna.visualization.plot_contour(study, params=top2)
            fig.write_html(f"{args.out_dir}/contour_{top2[0]}_vs_{top2[1]}.html")
            print(f"✅ contour_{top2[0]}_vs_{top2[1]}.html")
    except Exception as e:
        print(f"⚠️  contour: {e}")

    # ── 5. Summary CSV ──
    df = study.trials_dataframe()
    df.to_csv(f"{args.out_dir}/optuna_trials_summary.csv", index=False)
    print(f"✅ optuna_trials_summary.csv ({len(df)} rows)")

    print(f"\nAll visualizations saved to: {args.out_dir}/")
    print("Open any .html file in your browser to view interactive plots.")

if __name__ == "__main__":
    main()
