"""Reusable backend logic for exoplanet machine-learning workflows.

The module powers the Flask web interface and shared training scripts,
providing dataset loaders, model registries, and Gemini integration helpers.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Optional dependency for environment loading
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional at runtime
    load_dotenv = None  # type: ignore

try:  # Optional dependency used for Gemini integration
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - optional at runtime
    genai = None
    genai_types = None

# Add imports for plotting and image encoding
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


from scripts.common import (  # noqa: E402 - loaded after sys.path adjustments in callers
    MODELS_DIR,
    REPORTS_DIR,
    DataFrameSimpleImputer,
    Dataset,
    load_k2_dataset,
    load_kepler_dataset,
    load_tess_dataset,
)

BASE_DIR = Path(__file__).resolve().parent.parent
GEMINI_ENV_PATH = BASE_DIR / "config" / "gemini.env"

UPLOAD_OPTION_LABEL = "Upload CSV (choose schema)"

DATASET_CONFIG: Dict[str, Dict[str, object]] = {
    "tess": {
        "label": "TESS catalogue (Bundled)",
        "loader": load_tess_dataset,
        "model_files": {
            "CatBoost": "catboost_model.joblib",
            "Gradient Boosting": "gradient_boosting_model.joblib",
            "LightGBM": "lightgbm_model.joblib",
            "Logistic Regression": "logreg_model.joblib",
            "MLP": "mlp_model.joblib",
            "Random Forest": "random_forest_model.joblib",
            "XGBoost": "xgboost_model.joblib",
        },
        "default_models": (
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "Gradient Boosting",
            "Random Forest",
            "MLP",
            "Logistic Regression",
        ),
        "identifier_priority": [
            "unified_id",
            "stellar_id",
            "disposition",
            "mission",
            "sample_index",
        ],
    },
    "kepler": {
        "label": "Kepler KOI catalogue (Bundled)",
        "loader": load_kepler_dataset,
        "model_files": {
            "CatBoost": "catboost_model.joblib",
            "Gradient Boosting": "gradient_boosting_model.joblib",
            "LightGBM": "lightgbm_model.joblib",
            "Logistic Regression": "logreg_model.joblib",
            "MLP": "mlp_model.joblib",
            "Random Forest": "random_forest_model.joblib",
            "XGBoost": "xgboost_model.joblib",
        },
        "default_models": (
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "Gradient Boosting",
            "Random Forest",
            "MLP",
            "Logistic Regression",
        ),
        "identifier_priority": [
            "unified_id",
            "stellar_id",
            "disposition",
            "mission",
            "sample_index",
        ],
    },
    "k2": {
        "label": "K2 planet candidate catalogue (Bundled)",
        "loader": load_k2_dataset,
        "model_files": {
            "CatBoost": "catboost_model.joblib",
            "Gradient Boosting": "gradient_boosting_model.joblib",
            "LightGBM": "lightgbm_model.joblib",
            "Logistic Regression": "logreg_model.joblib",
            "MLP": "mlp_model.joblib",
            "Random Forest": "random_forest_model.joblib",
            "XGBoost": "xgboost_model.joblib",
        },
        "default_models": (
            "LightGBM",
            "XGBoost",
            "CatBoost",
            "Gradient Boosting",
            "Random Forest",
            "MLP",
            "Logistic Regression",
        ),
        "identifier_priority": [
            "unified_id",
            "stellar_id",
            "disposition",
            "mission",
            "sample_index",
        ],
    },
}

DATASET_LABEL_TO_KEY = {cfg["label"]: key for key, cfg in DATASET_CONFIG.items()}
DATASET_LABELS = [cfg["label"] for cfg in DATASET_CONFIG.values()]

# Standard feature list from combined_all.csv (10 features)
STANDARD_FEATURE_ORDER = (
    "orbital_period_days",
    "transit_duration_hr",
    "transit_depth_ppm",
    "planet_radius_rearth",
    "equilibrium_temp_k",
    "insolation_flux_earth",
    "stellar_teff_k",
    "stellar_radius_rsun",
    "stellar_logg_cgs",
    "magnitude",
)

FEATURE_DISPLAY_ORDER: Dict[str, Tuple[str, ...]] = {
    "tess": STANDARD_FEATURE_ORDER,
    "kepler": STANDARD_FEATURE_ORDER,
    "k2": STANDARD_FEATURE_ORDER,
}

STANDARD_FEATURE_DESCRIPTIONS = {
    "orbital_period_days": "Orbital period in days",
    "transit_duration_hr": "Transit duration in hours",
    "transit_depth_ppm": "Transit depth in parts per million",
    "planet_radius_rearth": "Planet radius in Earth radii",
    "equilibrium_temp_k": "Planet equilibrium temperature in Kelvin",
    "insolation_flux_earth": "Insolation flux in Earth units",
    "stellar_teff_k": "Stellar effective temperature in Kelvin",
    "stellar_radius_rsun": "Stellar radius in Solar radii",
    "stellar_logg_cgs": "Stellar surface gravity (log g, cgs)",
    "magnitude": "Kepler, TESS, or K2 magnitude",
}

FEATURE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "tess": STANDARD_FEATURE_DESCRIPTIONS,
    "kepler": STANDARD_FEATURE_DESCRIPTIONS,
    "k2": STANDARD_FEATURE_DESCRIPTIONS,
}

TRAINING_LABEL_CONFIG: Dict[str, Dict[str, object]] = {
    "tess": {
        "label_column": "disposition",
        "positive_labels": {"CONFIRMED"},
        "allowed_labels": {"CONFIRMED", "FALSE_POSITIVE", "CANDIDATE"},
    },
    "kepler": {
        "label_column": "disposition",
        "positive_labels": {"CONFIRMED"},
        "allowed_labels": {"CONFIRMED", "FALSE_POSITIVE", "CANDIDATE"},
    },
    "k2": {
        "label_column": "disposition",
        "positive_labels": {"CONFIRMED"},
        "allowed_labels": {"CONFIRMED", "FALSE_POSITIVE", "CANDIDATE"},
    },
}

DEFAULT_IDENTIFIER_PRIORITY = [
    "unified_id",
    "stellar_id",
    "disposition",
    "mission",
    "sample_index",
]

DEFAULT_TOP_K = 10


_ENV_LOADED_FOR_GEMINI = False


class MissingModelArtifact(Exception):
    """Raised when a requested model artefact is unavailable."""


def metrics_filename_for_model(model_filename: str) -> str:
    """Return the expected metrics filename for a given model artefact."""
    stem = Path(model_filename).stem
    return f"{stem}_metrics.json"


@lru_cache(maxsize=None)
def _load_dataset_bundle_cached(dataset_key: str) -> Dataset:
    loader = DATASET_CONFIG[dataset_key]["loader"]  # type: ignore[index]
    dataset = loader()
    if not isinstance(dataset, Dataset):
        raise TypeError("Dataset loader must return a scripts.common.Dataset instance")
    return dataset


def load_dataset_bundle(dataset_key: str) -> Dataset:
    """Return a deep copy of the dataset bundle for the requested catalogue."""
    cached = _load_dataset_bundle_cached(dataset_key)
    metadata_copy = None if cached.metadata is None else cached.metadata.copy()
    return Dataset(
        features=cached.features.copy(),
        target=cached.target.copy(),
        metadata=metadata_copy,
    )


def load_dataset_features(dataset_key: str) -> pd.DataFrame:
    return load_dataset_bundle(dataset_key).features


def load_dataset_metadata(dataset_key: str) -> pd.DataFrame:
    bundle = load_dataset_bundle(dataset_key)
    if bundle.metadata is None:
        return pd.DataFrame(index=bundle.features.index)
    return bundle.metadata


def available_models(dataset_key: str) -> Sequence[str]:
    return tuple(DATASET_CONFIG[dataset_key]["model_files"].keys())  # type: ignore[index]


def default_models(dataset_key: str) -> Sequence[str]:
    return DATASET_CONFIG[dataset_key]["default_models"]  # type: ignore[index]


def load_models(
    dataset_key: str,
    model_names: Iterable[str],
) -> Tuple[Dict[str, object], list[tuple[str, Path]]]:
    """Load trained model pipelines for the dataset.

    Returns a tuple consisting of the successfully loaded models and a list of
    missing artefacts (model name, expected path).
    """

    model_files: Dict[str, str] = DATASET_CONFIG[dataset_key]["model_files"]  # type: ignore[index]
    models: Dict[str, object] = {}
    missing: list[tuple[str, Path]] = []

    for name in model_names:
        filename = model_files.get(name)
        if not filename:
            continue
        path = MODELS_DIR / filename
        if not path.exists():
            missing.append((name, path))
            continue
        
        # Add debugging to identify which model is causing KeyError: 72
        import logging
        logging.info(f"Loading model: {name} from {path}")
        try:
            models[name] = load(path)
            logging.info(f"✓ Successfully loaded: {name}")
        except Exception as e:
            # Don't crash the entire app if one model file fails to unpickle.
            # Record it as missing and continue so the UI can still function with
            # the available models. Include the exception message in logs.
            logging.error(f"✗ Failed to load {name}: {e}")
            missing.append((name, path))
            continue

    return models, missing


def load_metrics_for_models(
    dataset_key: str,
    model_names: Iterable[str],
) -> tuple[Dict[str, Dict[str, object]], list[tuple[str, Path]]]:
    """Load evaluation metrics JSON files for each selected model."""

    model_files: Dict[str, str] = DATASET_CONFIG[dataset_key]["model_files"]  # type: ignore[index]
    metrics: Dict[str, Dict[str, object]] = {}
    missing: list[tuple[str, Path]] = []

    for name in model_names:
        filename = model_files.get(name)
        if not filename:
            continue
        metrics_path = REPORTS_DIR / metrics_filename_for_model(filename)
        if not metrics_path.exists():
            missing.append((name, metrics_path))
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as fp:
                metrics[name] = json.load(fp)
        except json.JSONDecodeError:
            missing.append((name, metrics_path))

    return metrics, missing


def preprocess_uploaded_file(uploaded_file, dataset_key: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Validate and coerce an uploaded CSV to match the training feature schema."""

    df = pd.read_csv(uploaded_file)
    reference_features = load_dataset_features(dataset_key)
    required_columns = reference_features.columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded file is missing required feature columns: " + ", ".join(missing)
        )

    features = df[required_columns].copy()

    identifier_priority = DATASET_CONFIG[dataset_key]["identifier_priority"]  # type: ignore[index]
    metadata_columns = [col for col in identifier_priority if col in df.columns]
    metadata = df[metadata_columns].copy() if metadata_columns else None

    return features, metadata


def run_predictions(
    models: Dict[str, object],
    features: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Run predictions for each model and return enriched results."""

    results: Dict[str, pd.DataFrame] = {}
    if metadata is not None:
        metadata = metadata.reindex(features.index)
    else:
        metadata = pd.DataFrame(index=features.index)

    if metadata.shape[1] == 0:
        metadata = metadata.copy()
        metadata["sample_index"] = np.arange(1, len(metadata) + 1)

    for name, pipeline in models.items():
        prediction_features = features.copy()
        expected_features = None
        if hasattr(pipeline, 'feature_names_in_'):
            expected_features = pipeline.feature_names_in_
        elif hasattr(pipeline, 'steps'):
            for _, step_obj in pipeline.steps:
                if hasattr(step_obj, 'feature_names_in_'):
                    expected_features = step_obj.feature_names_in_
                    break

        debug_info = {
            'input_columns': prediction_features.columns.tolist(),
            'input_shape': prediction_features.shape,
            'expected_features': list(expected_features) if expected_features is not None else None
        }

        if expected_features is not None:
            expected_list = list(expected_features)
            missing_in_input = [f for f in expected_list if f not in prediction_features.columns]
            if missing_in_input:
                import logging
                logging.basicConfig(level=logging.INFO)
                logging.warning(
                    "Model '%s' expected features missing from input: %s - filling with medians/zeros",
                    name,
                    missing_in_input,
                )
                medians = prediction_features.median()
                for col in missing_in_input:
                    if col in medians.index and not np.isnan(medians[col]):
                        fill_value = float(medians[col])
                    else:
                        fill_value = 0.0
                    prediction_features[col] = fill_value
            prediction_features = prediction_features.reindex(columns=expected_list, fill_value=0.0)

        try:
            prediction_features = prediction_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        except Exception:
            pass

        try:
            proba = pipeline.predict_proba(prediction_features)
            if proba.ndim == 1:
                proba_vals = np.zeros(len(proba), dtype=float)
            else:
                proba_vals = proba[:, -1]
        except Exception as e:
            # Error detail to UI
            raise RuntimeError(f"Model '{name}' failed to produce predictions: {e}\nDebug info: {debug_info}")

        try:
            preds = pipeline.predict(prediction_features)
        except Exception as e:
            raise RuntimeError(f"Model '{name}' failed to produce final labels: {e}\nDebug info: {debug_info}")

        df = pd.DataFrame(
            {
                "planet_candidate_probability": proba_vals,
                "predicted_label_numeric": preds,
            },
            index=features.index,
        )
        enriched = metadata.join(df, how="inner")
        enriched["predicted_label_numeric"] = enriched["predicted_label_numeric"].astype(int)
        enriched["prediction_label"] = np.where(
            enriched["predicted_label_numeric"] == 1,
            "confirmed",
            "false positive",
        )
        results[name] = enriched
    return results


def generate_transit_simulation(
    dataset_key: str,
    feature_values: Mapping[str, object],
    stats: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Generate a simplified transit light-curve simulation for UI playback.

    The simulation is heuristic—intended for visualization only—and attempts to
    derive reasonable parameters from the provided feature inputs. Missing
    values fall back to dataset medians when available.
    """

    if not feature_values:
        raise ValueError("Feature values are required to build a transit simulation.")

    def _lookup(keys: Sequence[str], fallback: Optional[float] = None) -> Optional[float]:
        for candidate in keys:
            if candidate in feature_values:
                try:
                    value = feature_values[candidate]
                    if value is None:
                        continue
                    return float(value)
                except (TypeError, ValueError):
                    continue
            if stats is not None and candidate in stats.index:
                try:
                    return float(stats.loc[candidate, "median"])
                except Exception:  # pragma: no cover - defensive fallback
                    continue
        return fallback

    # Updated to use standard feature names
    period_days = _lookup(["orbital_period_days", "pl_orbper", "koi_period"], 5.0) or 5.0
    period_days = max(0.5, period_days)

    duration_hours = _lookup(["transit_duration_hr", "pl_trandurh"], None)
    if duration_hours is None:
        duration_hours = max(1.0, period_days * 24 * 0.04)
    duration_hours = float(duration_hours)

    depth_ppm = _lookup(["transit_depth_ppm", "pl_trandep"], None)
    impact_parameter = _lookup(["pl_imppar"], 0.3) or 0.3
    impact_parameter = float(min(0.9, max(0.0, impact_parameter)))

    semi_major_ratio = _lookup(["koi_dor"], None)
    if semi_major_ratio is None:
        semi_major_ratio = 8.0 if dataset_key == "kepler" else 6.0
    semi_major_ratio = float(max(2.0, semi_major_ratio))

    radius_ratio: Optional[float] = None
    depth: float
    if depth_ppm is not None:
        depth = abs(depth_ppm) / 1_000_000.0
        depth = float(min(0.25, max(0.00005, depth)))
        radius_ratio = float(min(0.5, max(0.02, depth ** 0.5)))
    else:
        planet_radius_earth = _lookup(["planet_radius_rearth", "pl_rade", "koi_prad"], None)
        if planet_radius_earth is not None:
            radius_ratio = float(min(0.5, max(0.02, planet_radius_earth / 109.2)))
        else:
            radius_ratio = 0.12
        depth = float(min(0.25, max(0.00005, radius_ratio ** 2)))

    period_hours = float(period_days * 24.0)
    duration_fraction = duration_hours / period_hours if period_hours else 0.05
    duration_fraction = float(min(0.35, max(0.02, duration_fraction)))
    sigma = duration_fraction / 3.2

    phases = np.linspace(0.0, 1.0, 360)
    profile = np.exp(-0.5 * ((phases - 0.5) / sigma) ** 2)
    flux = 1.0 - depth * profile

    return {
        "phase": [round(float(value), 6) for value in phases.tolist()],
        "flux": [round(float(value), 6) for value in flux.tolist()],
        "radius_ratio": float(radius_ratio),
        "impact_parameter": impact_parameter,
        "semi_major_ratio": semi_major_ratio,
        "depth": depth,
        "duration_hours": duration_hours,
        "period_hours": period_hours,
        "duration_fraction": duration_fraction,
    }


def prepare_additional_training_data(
    dataset_key: str,
    uploaded_file,
    reference_columns: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert an uploaded labelled CSV into features/targets for retraining."""

    if uploaded_file is None:
        raise ValueError("No uploaded file provided for additional training data.")

    if dataset_key not in TRAINING_LABEL_CONFIG:
        raise ValueError(f"Dataset '{dataset_key}' does not support manual retraining.")

    config = TRAINING_LABEL_CONFIG[dataset_key]
    df = pd.read_csv(uploaded_file)

    missing = [col for col in reference_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded training file is missing required feature columns: " + ", ".join(missing)
        )

    label_column = config["label_column"]  # type: ignore[index]
    if label_column not in df.columns:
        raise ValueError(
            f"Uploaded training file must contain the label column '{label_column}'."
        )

    label_series = df[label_column].astype(str).str.upper()
    allowed_labels = config.get("allowed_labels")
    if allowed_labels:
        mask_allowed = label_series.isin(allowed_labels)
        if not mask_allowed.any():
            raise ValueError(
                "No rows in the uploaded file contain supported label values. "
                f"Allowed labels: {sorted(allowed_labels)}"
            )
        if not mask_allowed.all():
            df = df.loc[mask_allowed].copy()
            label_series = label_series.loc[mask_allowed]

    positive_labels = config["positive_labels"]  # type: ignore[index]
    target = label_series.isin(positive_labels).astype(int)

    features = df[reference_columns].copy()
    return features, target


def train_custom_model(
    dataset_key: str,
    algorithm: str,
    params: Dict[str, object],
    extra_training_file=None,
) -> tuple[Pipeline, Dict[str, object], pd.Index]:
    """Train a model with user-selected hyperparameters and return metrics."""

    bundle = load_dataset_bundle(dataset_key)
    X = bundle.features
    y = bundle.target

    def _to_float(value, default):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _to_int(value, default):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    test_size = _to_float(params.get("test_size", 0.2), 0.2)
    random_state = _to_int(params.get("random_state", 42), 42)
    cv_splits = _to_int(params.get("cv_splits", 5), 5)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    if extra_training_file is not None:
        extra_features, extra_target = prepare_additional_training_data(
            dataset_key,
            extra_training_file,
            X.columns,
        )
        X = pd.concat([X, extra_features], axis=0)
        y = pd.concat([y, extra_target], axis=0)

    algorithm_lower = algorithm.lower()
    model_param_keys: set[str]
    if algorithm_lower == "random forest":
        max_depth_param_raw = params.get("max_depth", None)
        max_depth_param = None
        if max_depth_param_raw is not None:
            if isinstance(max_depth_param_raw, str):
                cleaned = max_depth_param_raw.strip()
                if cleaned and cleaned.lower() not in {"auto", "none"}:
                    try:
                        max_depth_param = int(cleaned)
                    except ValueError:
                        max_depth_param = None
            else:
                try:
                    max_depth_param = int(max_depth_param_raw)
                except (TypeError, ValueError):
                    max_depth_param = None

        model_param_keys = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"}
        pipeline = Pipeline(
            steps=[
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=_to_int(params.get("n_estimators", 500), 500),
                        max_depth=max_depth_param,
                        min_samples_split=_to_int(params.get("min_samples_split", 2), 2),
                        min_samples_leaf=_to_int(params.get("min_samples_leaf", 1), 1),
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif algorithm_lower == "gradient boosting":
        learning_rate = _to_float(params.get("learning_rate", 0.05), 0.05)
        max_depth_raw = params.get("max_depth", 3)
        if isinstance(max_depth_raw, str):
            max_depth_raw = max_depth_raw.strip() or 3
        max_depth = _to_int(max_depth_raw, 3)

        subsample = _to_float(params.get("subsample", 1.0), 1.0)

        model_param_keys = {"n_estimators", "learning_rate", "max_depth", "subsample"}
        pipeline = Pipeline(
            steps=[
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=_to_int(params.get("n_estimators", 500), 500),
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        subsample=subsample,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif algorithm_lower == "mlp":
        hidden_layer_sizes_raw = params.get("hidden_layer_sizes", "256,128")
        if isinstance(hidden_layer_sizes_raw, (list, tuple)):
            hidden_layer_sizes = tuple(int(layer) for layer in hidden_layer_sizes_raw)
        else:
            hidden_layer_sizes = tuple(
                int(part.strip())
                for part in str(hidden_layer_sizes_raw).split(",")
                if part.strip()
            ) or (256, 128)

        model_param_keys = {
            "hidden_layer_sizes",
            "activation",
            "alpha",
            "learning_rate_init",
            "max_iter",
        }
        pipeline = Pipeline(
            steps=[
                ("imputer", DataFrameSimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=str(params.get("activation", "relu")),
                        alpha=_to_float(params.get("alpha", 0.0001), 0.0001),
                        learning_rate_init=_to_float(params.get("learning_rate_init", 0.001), 0.001),
                        max_iter=_to_int(params.get("max_iter", 400), 400),
                        random_state=random_state,
                        early_stopping=True,
                        n_iter_no_change=15,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics: Dict[str, object] = {
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

    model = pipeline.named_steps["model"]
    metrics["model_params"] = {
        key: value
        for key, value in model.get_params(deep=False).items()
        if key in model_param_keys
    }

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

    return pipeline, metrics, X.columns


def save_trained_model(
    pipeline: Pipeline,
    metrics: Dict[str, object],
    target_filename: str,
    feature_names: Sequence[str],
) -> None:
    """Persist a trained pipeline and associated reports to disk."""

    model_path = MODELS_DIR / target_filename
    dump(pipeline, model_path)

    metrics_path = REPORTS_DIR / metrics_filename_for_model(target_filename)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    artifact_stem = Path(target_filename).stem
    model = pipeline.named_steps["model"]

    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        abs_coeff = np.abs(coefficients)
        coeff_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": abs_coeff,
            }
        ).sort_values("abs_coefficient", ascending=False)
        coeff_df.head(20).to_csv(
            REPORTS_DIR / f"{artifact_stem}_top_coefficients.csv",
            index=False,
        )
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)
        importance_df.head(20).to_csv(
            REPORTS_DIR / f"{artifact_stem}_top_features.csv",
            index=False,
        )


def format_summary_table(
    df: pd.DataFrame,
    top_k: Optional[int] = DEFAULT_TOP_K,
    identifier_priority: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return the highest-probability rows (or all rows if ``top_k`` is ``None``)."""

    priority = list(identifier_priority or DEFAULT_IDENTIFIER_PRIORITY)
    identifier_columns = [col for col in priority if col in df.columns]
    df_display = df.copy()

    value_columns = [
        col
        for col in (
            "planet_candidate_probability",
            "prediction_label",
            "predicted_label_numeric",
        )
        if col in df_display.columns
    ]

    if top_k is None:
        subset = df_display.copy()
        return subset[identifier_columns + value_columns]

    subset = df_display.nlargest(top_k, "planet_candidate_probability").copy()
    subset["rank"] = np.arange(1, len(subset) + 1)
    ordered_columns = ["rank"] + identifier_columns + value_columns
    return subset[ordered_columns]


@dataclass
class FunctionCall:
    """Represents a function call requested by the Gemini model."""
    name: str
    args: Dict[str, object]


def search_exoplanet_candidates(dataset_key: str, query: str, model_name: str | None = None) -> str:
    """Search for exoplanet candidates by name or identifier with smart normalization."""
    try:
        import re
        
        bundle = load_dataset_bundle(dataset_key)
        selected_models = [model_name] if model_name else list(default_models(dataset_key))
        model_map, _ = load_models(dataset_key, selected_models)
        if not model_map:
            return "Model not available for this dataset."

        predictions = run_predictions(model_map, bundle.features, bundle.metadata)
        identifier_priority = DATASET_CONFIG[dataset_key]["identifier_priority"]
        results = {}
        
        # Normalize query: handle "KOI 700" -> "K700", "TOI 700" -> "700"
        normalized_query = query.lower().strip()
        normalized_query = re.sub(r'^koi\s*', 'k', normalized_query)  # "KOI 700" -> "k700"
        normalized_query = re.sub(r'^toi\s*', '', normalized_query)    # "TOI 700" -> "700"
        normalized_query = re.sub(r'[\s\-_]', '', normalized_query)    # Remove spaces/hyphens
        
        lower_query = query.lower()

        for model, df in predictions.items():
            search_columns = [col for col in identifier_priority if col in df.columns]
            
            # Try exact match first
            exact_mask = pd.Series(False, index=df.index)
            for col in search_columns:
                exact_mask |= df[col].astype(str).str.lower() == lower_query
            
            if exact_mask.any():
                filtered = df.loc[exact_mask]
            else:
                # Try normalized match
                normalized_mask = pd.Series(False, index=df.index)
                for col in search_columns:
                    normalized_values = df[col].astype(str).str.lower().str.replace(r'[\s\-_]', '', regex=True)
                    normalized_mask |= normalized_values == normalized_query
                    normalized_mask |= normalized_values.str.startswith(normalized_query + '.')
                
                if normalized_mask.any():
                    filtered = df.loc[normalized_mask]
                else:
                    # Fall back to partial match
                    partial_mask = pd.Series(False, index=df.index)
                    for col in search_columns:
                        partial_mask |= df[col].astype(str).str.lower().str.contains(lower_query, na=False)
                    
                    filtered = df.loc[partial_mask]
            
            if not filtered.empty:
                # Return only first match for each model to keep response concise
                results[model] = format_summary_table(
                    filtered.head(1), top_k=None, identifier_priority=identifier_priority
                ).to_dict(orient="records")

        if not results:
            return f"No candidates found matching '{query}' in {dataset_key} dataset. Try a different dataset or identifier."
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error during search: {e}"


def get_top_candidate_predictions(dataset_key: str, top_k: int = 5) -> str:
    """Get the top N exoplanet candidates with the highest prediction probability."""
    try:
        bundle = load_dataset_bundle(dataset_key)
        selected_models = list(default_models(dataset_key))
        model_map, _ = load_models(dataset_key, selected_models)
        if not model_map:
            return "No models available to get predictions."

        predictions = run_predictions(model_map, bundle.features, bundle.metadata)
        identifier_priority = DATASET_CONFIG[dataset_key]["identifier_priority"]
        
        summaries = {}
        for model_name, df in predictions.items():
            summary_df = format_summary_table(df, top_k, identifier_priority)
            summaries[model_name] = summary_df.to_dict(orient="records")
        
        return json.dumps(summaries, indent=2)
    except Exception as e:
        return f"Error getting top candidates: {e}"


def get_model_performance_metrics(dataset_key: str, model_name: str) -> str:
    """Get the performance metrics for a specified model."""
    try:
        metrics, missing = load_metrics_for_models(dataset_key, [model_name])
        if missing:
            return f"Metrics for model '{model_name}' not found."
        if not metrics:
            return f"No metrics available for model '{model_name}'."
        
        return json.dumps(metrics.get(model_name, {}), indent=2)
    except Exception as e:
        return f"Error getting model metrics: {e}"


def plot_pixelfile(starname: str) -> str:
    """
    Searches for and plots the Target Pixel File (TPF) for a given star name
    from the TESS or Kepler/K2 mission data using lightkurve.
    Returns information about the data availability and a Base64 encoded plot.
    """
    try:
        # Import lightkurve only when needed (optional dependency)
        from lightkurve import search_targetpixelfile
        
        print(f"Executing plot_pixelfile for: {starname}...")
        
        # Search for target pixel files
        search_result = search_targetpixelfile(starname)
        
        if search_result is None or len(search_result) == 0:
            return json.dumps({
                "status": "error",
                "message": f"No Target Pixel File found for '{starname}'. This target may not have been observed by TESS or Kepler/K2, or the data may not be publicly available yet."
            })
        
        # Download the first available TPF with a timeout to prevent long waits
        print(f"Downloading Target Pixel File for {starname}...")
        import warnings
        warnings.filterwarnings("ignore")  # Suppress warnings for faster processing
        
        # Use a smaller figure size for faster rendering
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Download and plot with minimal data processing
        pixelfile = search_result[0].download()
        
        # Plot only the first frame with minimal processing
        pixelfile.plot(ax=ax, frame=0)
        ax.set_title(f'TPF for {starname}')
        
        # Save plot to a memory buffer with lower quality for faster processing
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Encode image to Base64
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Get only essential TPF information
        tpf_info = {
            "target_name": str(pixelfile.targetid) if hasattr(pixelfile, 'targetid') else starname,
            "mission": str(pixelfile.mission) if hasattr(pixelfile, 'mission') else "Unknown",
            "ra": float(pixelfile.ra) if hasattr(pixelfile, 'ra') else None,
            "dec": float(pixelfile.dec) if hasattr(pixelfile, 'dec') else None,
            "time_range": f"{pixelfile.time.min().value:.2f} to {pixelfile.time.max().value:.2f} BJD" if hasattr(pixelfile, 'time') else "N/A",
            "cadence_count": int(len(pixelfile.time)) if hasattr(pixelfile, 'time') else 0,
            "aperture_shape": f"{pixelfile.shape[1]}x{pixelfile.shape[2]} pixels" if hasattr(pixelfile, 'shape') and len(pixelfile.shape) >= 3 else "N/A",
        }
        
        result = {
            "status": "success",
            "message": f"Target Pixel File downloaded and plotted successfully for '{starname}'",
            "available_observations": len(search_result),
            "downloaded_tpf": tpf_info,
            "plot_base64": plot_base64  # Add the base64 string to the result
        }
        
        return json.dumps(result, indent=2)
        
    except ImportError:
        return json.dumps({
            "status": "error",
            "message": "Error: lightkurve package is not installed. Install it with: pip install lightkurve"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error downloading or processing Target Pixel File for '{starname}': {str(e)}"
        })


AVAILABLE_TOOLS = {
    "search_exoplanet_candidates": search_exoplanet_candidates,
    "get_top_candidate_predictions": get_top_candidate_predictions,
    "get_model_performance_metrics": get_model_performance_metrics,
    "plot_pixelfile": plot_pixelfile,
}

def get_gemini_tools():
    """Returns the list of tool definitions for Gemini using new google.genai API."""
    if genai_types is None:
        return []
    
    return [
        genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name="search_exoplanet_candidates",
                    description="Search for exoplanet candidates by name or identifier within a given dataset (tess, kepler, k2).",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "dataset_key": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="The dataset to search (tess, kepler, or k2)."
                            ),
                            "query": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="The name or identifier of the exoplanet candidate to search for."
                            ),
                            "model_name": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="Optional: The specific prediction model to use."
                            ),
                        },
                        required=["dataset_key", "query"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="get_top_candidate_predictions",
                    description="Get the top N exoplanet candidates with the highest prediction probability from a dataset.",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "dataset_key": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="The dataset to get predictions from (tess, kepler, or k2)."
                            ),
                            "top_k": genai_types.Schema(
                                type=genai_types.Type.INTEGER,
                                description="The number of top candidates to return. Defaults to 5."
                            ),
                        },
                        required=["dataset_key"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="get_model_performance_metrics",
                    description="Get the performance metrics (like accuracy, precision, recall, f1-score, roc_auc) for a specified model on a given dataset.",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "dataset_key": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="The dataset the model was trained on (tess, kepler, or k2)."
                            ),
                            "model_name": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="The name of the model to get metrics for (e.g., 'Random Forest', 'XGBoost')."
                            ),
                        },
                        required=["dataset_key", "model_name"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="plot_pixelfile",
                    description="Searches for and downloads the Target Pixel File (TPF) for a given star name (e.g., 'KIC 8462852', 'Kepler-10', 'TOI 700') from the TESS or Kepler/K2 mission data using lightkurve. Returns detailed information about available observations and downloaded data.",
                    parameters=genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={
                            "starname": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="The name or target ID of the astronomical target (star, exoplanet system) to download data for. Examples: 'KIC 8462852', 'Kepler-10', 'TOI 700', 'TIC 307210830'."
                            )
                        },
                        required=["starname"],
                    ),
                ),
            ]
        )
    ]

def build_context_for_ai(
    model_results: Dict[str, pd.DataFrame],
    top_k: Optional[int],
    dataset_label: str,
    identifier_priority: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Structure the model outputs for LLM consumption."""

    context: Dict[str, object] = {"dataset": dataset_label}
    for model_name, df in model_results.items():
        top_rows = format_summary_table(df, top_k, identifier_priority)
        context[model_name] = top_rows.to_dict(orient="records")
        context[f"{model_name}_metrics"] = {
            "total_samples": int(df.shape[0]),
            "positive_predictions": int(df["predicted_label_numeric"].sum()),
            "mean_probability": float(df["planet_candidate_probability"].mean()),
        }
    return context


def configure_gemini(force_reload: bool = False) -> Optional[str]:
    """Return the Gemini API key if available and return client instance."""

    global _ENV_LOADED_FOR_GEMINI

    if load_dotenv is not None:
        needs_reload = (
            force_reload
            or not _ENV_LOADED_FOR_GEMINI
            or os.environ.get("GEMINI_API_KEY") is None
        )
        if needs_reload:
            if GEMINI_ENV_PATH.exists():
                load_dotenv(dotenv_path=str(GEMINI_ENV_PATH), override=True)
            else:  # Fallback to default behaviour for backwards compatibility
                load_dotenv(override=False)
            _ENV_LOADED_FOR_GEMINI = True

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    if genai is None:
        raise RuntimeError("google-generativeai is not installed; install it to enable Gemini.")
    
    # Return API key, client will be created in call_gemini
    return api_key


def call_gemini(model_name: str, prompt: str, context: Dict[str, object]) -> str:
    """Generate a Gemini response using the provided context and function calling with new API."""

    if genai is None or genai_types is None:
        raise RuntimeError("google.genai is not installed.")
    
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment.")
    
    # Create client with API key
    client = genai.Client(api_key=api_key)
    
    # Get tool definitions
    tools = get_gemini_tools()
    
    # Create initial user prompt with context
    user_prompt = f"""You are an expert AI assistant helping to interpret exoplanet candidate predictions.

Context Data:
{json.dumps(context, indent=2)}

User Question: {prompt}

Please provide a detailed and helpful answer. You can use the available functions to search for specific candidates or get additional information.

CRITICAL INSTRUCTION: If you use the plot_pixelfile function and it returns a JSON with plot_base64, you MUST include the complete JSON response (including the plot_base64 field) in your answer. Do not summarize or omit the plot_base64 data. The entire JSON response is needed to display the image on the webpage."""
    
    # Turn 1: Send initial request with tools
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(tools=tools)
    )
    
    # Check if model called a function
    if response.function_calls:
        # Handle function calls
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        # Build conversation history
        contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=user_prompt)],
            )
        ]
        
        while response.function_calls and iteration < max_iterations:
            iteration += 1
            function_call = response.function_calls[0]
            function_name = function_call.name
            function_args = dict(function_call.args)
            
            # Add assistant's function call to history
            contents.append(
                genai_types.Content(
                    role="assistant",
                    parts=[
                        genai_types.Part.from_function_call(
                            name=function_name,
                            args=function_args,
                        )
                    ],
                )
            )
            
            # Execute the function
            if function_name in AVAILABLE_TOOLS:
                try:
                    tool_function = AVAILABLE_TOOLS[function_name]
                    result = tool_function(**function_args)
                    function_output = {"result": result}
                except Exception as e:
                    function_output = {"error": str(e)}
            else:
                function_output = {"error": f"Unknown function: {function_name}"}
            
            # Add tool response to history
            contents.append(
                genai_types.Content(
                    role="tool",
                    parts=[
                        genai_types.Part.from_function_response(
                            name=function_name,
                            response=function_output,
                        )
                    ],
                )
            )
            
            # Send back to model with function result
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(tools=tools)
            )
    
    # Return final text response
    return response.text if hasattr(response, "text") and response.text else "(No response generated.)"


def dataset_feature_statistics(dataset_key: str) -> pd.DataFrame:
    """Return median, mean, and std for each feature (used for manual inputs)."""

    features = load_dataset_features(dataset_key)
    stats = pd.DataFrame(
        {
            "median": features.median(),
            "mean": features.mean(),
            "std": features.std().fillna(0.0),
        }
    )
    return stats


def ensure_output_directories() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)


__all__ = [
    "UPLOAD_OPTION_LABEL",
    "DATASET_CONFIG",
    "DATASET_LABELS",
    "DATASET_LABEL_TO_KEY",
    "FEATURE_DISPLAY_ORDER",
    "FEATURE_DESCRIPTIONS",
    "TRAINING_LABEL_CONFIG",
    "DEFAULT_IDENTIFIER_PRIORITY",
    "DEFAULT_TOP_K",
    "MissingModelArtifact",
    "metrics_filename_for_model",
    "load_dataset_bundle",
    "load_dataset_features",
    "load_dataset_metadata",
    "available_models",
    "default_models",
    "load_models",
    "load_metrics_for_models",
    "preprocess_uploaded_file",
    "run_predictions",
    "prepare_additional_training_data",
    "train_custom_model",
    "save_trained_model",
    "format_summary_table",
    "build_context_for_ai",
    "configure_gemini",
    "call_gemini",
    "dataset_feature_statistics",
    "generate_transit_simulation",
    "ensure_output_directories",
    "search_exoplanet_candidates",
    "get_top_candidate_predictions",
    "get_model_performance_metrics",
]
