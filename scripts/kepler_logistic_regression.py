"""Train and evaluate a Logistic Regression model on the Kepler KOI cumulative catalogue.

The model predicts whether a Kepler Object of Interest (KOI) is confirmed or a
false positive based on the `koi_disposition` label. Only rows marked as
``CONFIRMED`` or ``FALSE POSITIVE`` are included; the former is treated as the
positive class.

Outputs:
- reports/kepler_logistic_regression_metrics.json: summary metrics and cross-validation scores
- reports/kepler_logistic_regression_confusion_matrix.png: confusion matrix heatmap
- reports/kepler_logistic_regression_top_coefficients.csv: most influential coefficients
- models/kepler_logistic_regression.joblib: trained pipeline for reuse

Usage:
    python scripts/kepler_logistic_regression.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    MODELS_DIR,
    REPORTS_DIR,
    DataFrameSimpleImputer,
    ensure_output_directories,
    load_kepler_dataset,
)


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the modelling pipeline for Logistic Regression."""
    return Pipeline(
        steps=[
            ("imputer", DataFrameSimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_splits: int = 5,
) -> Dict[str, object]:
    """Fit the pipeline, compute metrics, and run cross-validation."""
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["false_positive", "confirmed"],
            output_dict=True,
        ),
    }

    model: LogisticRegression = pipeline.named_steps["model"]
    metrics["model_params"] = {
        "penalty": model.penalty,
        "C": model.C,
        "solver": model.solver,
        "max_iter": model.max_iter,
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "roc_auc": "roc_auc", "f1": "f1"},
        return_train_score=False,
        n_jobs=-1,
    )

    metrics["cross_validation"] = {
        metric: {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "all_scores": scores.tolist(),
        }
        for metric, scores in cv_results.items()
        if metric.startswith("test_")
    }

    metrics["cross_validation"] = {
        key.replace("test_", ""): value for key, value in metrics["cross_validation"].items()
    }

    return metrics


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Create and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = ["False positive", "Confirmed"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_top_coefficients(
    model: LogisticRegression,
    feature_names: pd.Index,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Persist the top positive and negative coefficients by magnitude."""
    coefficients = model.coef_[0]
    abs_coeff = np.abs(coefficients)
    df_coeff = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": abs_coeff,
        }
    )
    df_coeff.sort_values(by="abs_coefficient", ascending=False, inplace=True)
    df_coeff.head(top_n).to_csv(output_path, index=False)


def main() -> None:
    ensure_output_directories()

    data = load_kepler_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data.features,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    pipeline = build_pipeline()
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)

    # Persist model
    dump(pipeline, MODELS_DIR / "kepler_logistic_regression.joblib")

    # Save metrics
    metrics_output = REPORTS_DIR / "kepler_logistic_regression_metrics.json"
    with metrics_output.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # Plot confusion matrix
    y_pred = pipeline.predict(X_test)
    save_confusion_matrix(
        y_test,
        y_pred,
        REPORTS_DIR / "kepler_logistic_regression_confusion_matrix.png",
    )

    # Save top coefficients
    model: LogisticRegression = pipeline.named_steps["model"]
    save_top_coefficients(
        model,
        X_train.columns,
        REPORTS_DIR / "kepler_logistic_regression_top_coefficients.csv",
    )

    print("Kepler Logistic Regression training complete. Key metrics:")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {metric}: {metrics[metric]:.4f}")

    print(f"Detailed metrics saved to {metrics_output}")
    print(f"Model saved to {MODELS_DIR / 'kepler_logistic_regression.joblib'}")


if __name__ == "__main__":
    main()
