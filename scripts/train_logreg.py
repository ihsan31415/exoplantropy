"""Train Logistic Regression model for all datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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
    """Create the modelling pipeline for a Logistic Regression classifier."""
    return Pipeline(
        steps=[
            ("imputer", DataFrameSimpleImputer(strategy="median")),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_model(
    dataset_key: str = "kepler",
    test_size: float = 0.2,
    cv_splits: int = 5,
    random_state: int = 42,
) -> None:
    """Train and evaluate a Logistic Regression model."""
    
    ensure_output_directories()
    
    loader, dataset_name = DATASET_LOADERS[dataset_key]
    dataset = loader()
    
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.target,
        test_size=test_size,
        stratify=dataset.target,
        random_state=random_state,
    )
    
    print(f"\nTraining Logistic Regression on {dataset_name} dataset...")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        "dataset": dataset_name,
        "model": "Logistic Regression",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    
    print(f"\nTest Set Performance:")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall: {metrics['recall']:.4f}")
    print(f"  f1: {metrics['f1']:.4f}")
    print(f"  roc_auc: {metrics['roc_auc']:.4f}")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
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
        metric.replace("test_", ""): {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "all_scores": scores.tolist(),
        }
        for metric, scores in cv_results.items()
        if metric.startswith("test_")
    }
    
    # Classification report
    metrics["classification_report"] = classification_report(
        y_test,
        y_pred,
        target_names=["false_positive", "confirmed"],
        output_dict=True,
    )
    
    # Save model
    model_filename = f"{dataset_key}_logreg.joblib"
    model_path = MODELS_DIR / model_filename
    dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_filename = f"{dataset_key}_logreg_metrics.json"
    metrics_path = REPORTS_DIR / metrics_filename
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"Detailed metrics saved to {metrics_path}")
    
    # Get feature importance (coefficients)
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        abs_coeff = np.abs(coefficients)
        coeff_df = pd.DataFrame(
            {
                "feature": dataset.features.columns,
                "coefficient": coefficients,
                "abs_coefficient": abs_coeff,
            }
        ).sort_values("abs_coefficient", ascending=False)
        
        coeff_path = REPORTS_DIR / f"{dataset_key}_logreg_coefficients.csv"
        coeff_df.to_csv(coeff_path, index=False)
        print(f"Feature coefficients saved to {coeff_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression model for exoplanet classification"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kepler",
        choices=list(DATASET_LOADERS.keys()),
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    train_model(
        dataset_key=args.dataset,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
