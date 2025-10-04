"""Train and evaluate a Gradient Boosting (GBM) model on the K2 planet candidate catalogue."""
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
from sklearn.ensemble import GradientBoostingClassifier
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

from common import (
    MODELS_DIR,
    REPORTS_DIR,
    DataFrameSimpleImputer,
    ensure_output_directories,
    load_k2_dataset,
)


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the modelling pipeline for GBM."""
    return Pipeline(
        steps=[
            ("imputer", DataFrameSimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=random_state,
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

    model: GradientBoostingClassifier = pipeline.named_steps["model"]
    metrics["model_params"] = {
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "subsample": model.subsample,
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def collect_feature_importance(
    pipeline: Pipeline, feature_names: pd.Index, top_n: int = 20
) -> pd.DataFrame:
    """Extract feature importances from the fitted GBM model."""
    model: GradientBoostingClassifier = pipeline.named_steps["model"]
    importance = model.feature_importances_
    df_importance = pd.DataFrame({"feature": feature_names, "importance": importance})
    df_importance.sort_values(by="importance", ascending=False, inplace=True)
    return df_importance.head(top_n)


def main() -> None:
    ensure_output_directories()

    data = load_k2_dataset()
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
    dump(pipeline, MODELS_DIR / "k2_gbm.joblib")

    # Save metrics
    metrics_output = REPORTS_DIR / "k2_gbm_metrics.json"
    with metrics_output.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # Plot confusion matrix
    y_pred = pipeline.predict(X_test)
    save_confusion_matrix(y_test, y_pred, REPORTS_DIR / "k2_gbm_confusion_matrix.png")

    # Feature importances
    top_features = collect_feature_importance(pipeline, X_train.columns)
    top_features.to_csv(REPORTS_DIR / "k2_gbm_top_features.csv", index=False)

    print("K2 Gradient Boosting training complete. Key metrics:")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {metric}: {metrics[metric]:.4f}")

    print(f"Detailed metrics saved to {metrics_output}")
    print(f"Model saved to {MODELS_DIR / 'k2_gbm.joblib'}")


if __name__ == "__main__":
    main()
