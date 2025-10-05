"""Unified Random Forest trainer for the TESS, Kepler, and K2 catalogues."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
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
    Dataset,
    ensure_output_directories,
    load_k2_dataset,
    load_kepler_dataset,
    load_tess_dataset,
)

DATASET_LOADERS: Dict[str, Tuple[Callable[[], Dataset], str]] = {
    "tess": (load_tess_dataset, "TESS TOI"),
    "kepler": (load_kepler_dataset, "Kepler KOI"),
    "k2": (load_k2_dataset, "K2"),
}


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Create the modelling pipeline for Random Forest."""
    return Pipeline(
        steps=[
            ("imputer", DataFrameSimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
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

    model: RandomForestClassifier = pipeline.named_steps["model"]
    metrics["model_params"] = {
        "n_estimators": model.n_estimators,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "max_depth": model.max_depth,
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


def collect_feature_importance(
    pipeline: Pipeline,
    feature_names: pd.Index,
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract feature importances from the fitted Random Forest model."""
    model: RandomForestClassifier = pipeline.named_steps["model"]
    importance = model.feature_importances_
    df_importance = pd.DataFrame({"feature": feature_names, "importance": importance})
    df_importance.sort_values(by="importance", ascending=False, inplace=True)
    return df_importance.head(top_n)


def load_dataset(dataset_key: str) -> Dataset:
    """Load the dataset associated with a key."""
    try:
        loader, _ = DATASET_LOADERS[dataset_key]
    except KeyError as exc:  # pragma: no cover - argparse guards against this
        raise ValueError(f"Unknown dataset key: {dataset_key}") from exc
    return loader()


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Random Forest model for exoplanet catalogues.")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="kepler",
        choices=sorted(DATASET_LOADERS.keys()),
        help="Dataset to train on (tess, kepler, k2). Defaults to 'kepler' if omitted.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Number of top features to export for feature-importance reporting.",
    )
    return parser.parse_args(args)


def main(cli_args: Iterable[str] | None = None) -> None:
    args = parse_args(cli_args)
    dataset_key = args.dataset
    random_state = args.random_state

    ensure_output_directories()

    dataset = load_dataset(dataset_key)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=args.test_size,
        random_state=random_state,
        stratify=dataset.target,
    )

    pipeline = build_pipeline(random_state=random_state)
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test, cv_splits=args.cv_splits)

    model_suffix = f"{dataset_key}_random_forest"

    dump(pipeline, MODELS_DIR / f"{model_suffix}.joblib")

    metrics_output = REPORTS_DIR / f"{model_suffix}_metrics.json"
    with metrics_output.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    y_pred = pipeline.predict(X_test)
    save_confusion_matrix(y_test, y_pred, REPORTS_DIR / f"{model_suffix}_confusion_matrix.png")

    top_features = collect_feature_importance(pipeline, X_train.columns, top_n=args.top_features)
    top_features.to_csv(REPORTS_DIR / f"{model_suffix}_top_features.csv", index=False)

    _, dataset_name = DATASET_LOADERS[dataset_key]
    print(f"{dataset_name} Random Forest training complete. Key metrics:")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {metric}: {metrics[metric]:.4f}")

    print(f"Detailed metrics saved to {metrics_output}")
    print(f"Model saved to {MODELS_DIR / f'{model_suffix}.joblib'}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
