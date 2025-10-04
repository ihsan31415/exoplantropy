"""Train and evaluate a Decision Tree model on the TESS TOI catalogue."""
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
from sklearn.impute import SimpleImputer
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
from sklearn.tree import DecisionTreeClassifier

from common import (
    MODELS_DIR,
    REPORTS_DIR,
    ensure_output_directories,
    load_tess_dataset,
)


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the modelling pipeline for Decision Tree."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=12,
                    min_samples_split=6,
                    min_samples_leaf=3,
                    class_weight="balanced",
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

    model: DecisionTreeClassifier = pipeline.named_steps["model"]
    metrics["model_params"] = {
        "criterion": model.criterion,
        "max_depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_feature_importances(
    model: DecisionTreeClassifier,
    feature_names: pd.Index,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Persist the top feature importances."""
    importances = model.feature_importances_
    df_importance = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_importance.sort_values(by="importance", ascending=False, inplace=True)
    df_importance.head(top_n).to_csv(output_path, index=False)


def main() -> None:
    ensure_output_directories()

    data = load_tess_dataset()
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
    dump(pipeline, MODELS_DIR / "tess_decision_tree.joblib")

    # Save metrics
    metrics_output = REPORTS_DIR / "tess_decision_tree_metrics.json"
    with metrics_output.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # Plot confusion matrix
    y_pred = pipeline.predict(X_test)
    save_confusion_matrix(y_test, y_pred, REPORTS_DIR / "tess_decision_tree_confusion_matrix.png")

    # Feature importances
    model: DecisionTreeClassifier = pipeline.named_steps["model"]
    save_feature_importances(
        model,
        X_train.columns,
        REPORTS_DIR / "tess_decision_tree_top_features.csv",
    )

    print("Decision Tree training complete. Key metrics:")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {metric}: {metrics[metric]:.4f}")

    print(f"Detailed metrics saved to {metrics_output}")
    print(f"Model saved to {MODELS_DIR / 'tess_decision_tree.joblib'}")


if __name__ == "__main__":
    main()
