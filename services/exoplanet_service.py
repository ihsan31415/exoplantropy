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
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - optional at runtime
    genai = None

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
        "label": "TESS catalogue (bundled)",
        "loader": load_tess_dataset,
        "model_files": {
            "TESS Random Forest": "tess_random_forest.joblib",
            "TESS Gradient Boosting": "tess_gbm.joblib",
            "TESS XGBoost": "tess_xgboost.joblib",
            "TESS LightGBM": "tess_lightgbm.joblib",
            "TESS CatBoost": "tess_catboost.joblib",
            "TESS MLP": "tess_mlp.joblib",
        },
        "default_models": (
            "TESS LightGBM",
            "TESS XGBoost",
            "TESS CatBoost",
            "TESS Gradient Boosting",
            "TESS Random Forest",
            "TESS MLP",
        ),
        "identifier_priority": [
            "toi",
            "tid",
            "ctoi_alias",
            "tfopwg_disp",
            "sample_index",
        ],
    },
    "kepler": {
        "label": "Kepler KOI catalogue (bundled)",
        "loader": load_kepler_dataset,
        "model_files": {
            "Kepler Random Forest": "kepler_random_forest.joblib",
            "Kepler Gradient Boosting": "kepler_gbm.joblib",
            "Kepler XGBoost": "kepler_xgboost.joblib",
            "Kepler LightGBM": "kepler_lightgbm.joblib",
            "Kepler CatBoost": "kepler_catboost.joblib",
            "Kepler MLP": "kepler_mlp.joblib",
        },
        "default_models": (
            "Kepler LightGBM",
            "Kepler XGBoost",
            "Kepler CatBoost",
            "Kepler Gradient Boosting",
            "Kepler Random Forest",
            "Kepler MLP",
        ),
        "identifier_priority": [
            "kepid",
            "kepoi_name",
            "kepler_name",
            "koi_disposition",
            "sample_index",
        ],
    },
    "k2": {
        "label": "K2 planet candidate catalogue (bundled)",
        "loader": load_k2_dataset,
        "model_files": {
            "K2 Random Forest": "k2_random_forest.joblib",
            "K2 Gradient Boosting": "k2_gbm.joblib",
            "K2 XGBoost": "k2_xgboost.joblib",
            "K2 LightGBM": "k2_lightgbm.joblib",
            "K2 CatBoost": "k2_catboost.joblib",
            "K2 MLP": "k2_mlp.joblib",
        },
        "default_models": (
            "K2 LightGBM",
            "K2 XGBoost",
            "K2 CatBoost",
            "K2 Gradient Boosting",
            "K2 Random Forest",
            "K2 MLP",
        ),
        "identifier_priority": [
            "tic_id",
            "k2_name",
            "pl_name",
            "disposition",
            "sample_index",
        ],
    },
}

DATASET_LABEL_TO_KEY = {cfg["label"]: key for key, cfg in DATASET_CONFIG.items()}
DATASET_LABELS = [cfg["label"] for cfg in DATASET_CONFIG.values()]

FEATURE_DISPLAY_ORDER: Dict[str, Tuple[str, ...]] = {
    "tess": (
        "pl_trandep",
        "pl_tranmid",
        "pl_trandurh",
        "pl_orbper",
        "st_tmag",
        "pl_eqt",
        "st_dist",
        "pl_insol",
        "st_pmra",
        "st_pmdec",
    ),
    "kepler": (
        "koi_dikco_msky",
        "koi_max_mult_ev",
        "koi_dicco_msky",
        "koi_fwm_srao",
        "koi_prad",
        "koi_srho",
        "koi_dor",
        "koi_fwm_sdeco",
        "koi_max_sngle_ev",
        "koi_period",
    ),
    "k2": (
        "pl_trandep",
        "pl_orbper",
        "pl_tranmid",
        "sy_pm",
        "sy_dist",
        "pl_rade",
        "sy_tmag",
        "sy_gaiamag",
        "pl_eqt",
        "pl_imppar",
    ),
}

FEATURE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "tess": {
        "pl_trandep": "Transit depth—the fractional drop in stellar brightness when the planet crosses the star.",
        "pl_tranmid": "Mid-transit time in Barycentric Julian Date (BJD), marking the moment the planet lies directly in front of the star.",
        "pl_trandurh": "Total transit duration in hours from ingress through egress.",
        "pl_orbper": "Planetary orbital period in days; the time needed to complete one orbit around the star.",
        "st_tmag": "TESS apparent magnitude of the host star, indicating how bright it appears to the telescope.",
        "pl_eqt": "Estimated planetary equilibrium temperature in Kelvin assuming zero albedo.",
        "st_dist": "Distance to the star in parsecs (1 parsec ≈ 3.26 light years).",
        "pl_insol": "Stellar irradiance received by the planet relative to Earth (S⊕).",
        "st_pmra": "Proper motion of the star in right ascension, measured in milliarcseconds per year.",
        "st_pmdec": "Proper motion of the star in declination, measured in milliarcseconds per year.",
    },
    "kepler": {
        "koi_dikco_msky": "Differential image centroid offset in the column direction (mas) during transit; useful for spotting false-light sources.",
        "koi_max_mult_ev": "Maximum number of detected multi-planet transit events for this target.",
        "koi_dicco_msky": "Differential image centroid offset in the row direction (mas), another indicator of position shifts during transit.",
        "koi_fwm_srao": "Full width at half maximum (FWHM) of the stellar image along the row axis in arcseconds—reflects focus and seeing.",
        "koi_prad": "Planetary radius in Earth radii derived from transit modeling.",
        "koi_srho": "Stellar density (grams per cubic centimeter) inferred from the light curve.",
        "koi_dor": "Orbital distance to stellar radius ratio (a/R★), describing how far the planet orbits from the stellar surface.",
        "koi_fwm_sdeco": "Image FWHM along the column axis, helping assess observation quality.",
        "koi_max_sngle_ev": "Strongest single transit event signal with the highest signal-to-noise ratio.",
        "koi_period": "Candidate orbital period in days derived from the spacing between transits.",
    },
    "k2": {
        "pl_trandep": "Transit depth—the fraction of starlight lost when the planet passes in front of the star.",
        "pl_orbper": "Planetary orbital period in days.",
        "pl_tranmid": "Mid-transit time in BJD for K2 events.",
        "sy_pm": "Total proper motion of the stellar system (mas per year), combining RA and Dec components.",
        "sy_dist": "Distance from the system to Earth in parsecs.",
        "pl_rade": "Planetary radius in Earth radii estimated from the transit fit.",
        "sy_tmag": "TESS magnitude of the system—an indicator of brightness in the T band.",
        "sy_gaiamag": "Gaia G-band magnitude, a broad brightness measurement from the Gaia satellite.",
        "pl_eqt": "Planetary equilibrium temperature in Kelvin.",
        "pl_imppar": "Impact parameter describing how close the planet’s path comes to the stellar disk center (in stellar radii).",
    },
}

TRAINING_LABEL_CONFIG: Dict[str, Dict[str, object]] = {
    "tess": {
        "label_column": "tfopwg_disp",
        "positive_labels": {"PC"},
        "allowed_labels": {"PC", "CP", "FP", "FA", "KP"},
    },
    "kepler": {
        "label_column": "koi_disposition",
        "positive_labels": {"CONFIRMED"},
        "allowed_labels": {"CONFIRMED", "FALSE POSITIVE"},
    },
    "k2": {
        "label_column": "disposition",
        "positive_labels": {"CONFIRMED"},
        "allowed_labels": {"CONFIRMED", "FALSE POSITIVE"},
    },
}

DEFAULT_IDENTIFIER_PRIORITY = [
    "toi",
    "tid",
    "ctoi_alias",
    "tfopwg_disp",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "tic_id",
    "k2_name",
    "pl_name",
    "disposition",
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
        models[name] = load(path)

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
        proba = pipeline.predict_proba(features)[:, 1]
        preds = pipeline.predict(features)
        df = pd.DataFrame(
            {
                "planet_candidate_probability": proba,
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

    period_days = _lookup(["pl_orbper", "koi_period"], 5.0) or 5.0
    period_days = max(0.5, period_days)

    duration_hours = _lookup(["pl_trandurh"], None)
    if duration_hours is None:
        duration_hours = max(1.0, period_days * 24 * 0.04)
    duration_hours = float(duration_hours)

    depth_ppm = _lookup(["pl_trandep"], None)
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
        planet_radius_earth = _lookup(["pl_rade", "koi_prad"], None)
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
    """Return the Gemini API key if available and configure the client."""

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
    genai.configure(api_key=api_key)
    return api_key


def call_gemini(model_name: str, prompt: str, context: Dict[str, object]) -> str:
    """Generate a Gemini response using the provided context."""

    if genai is None:
        raise RuntimeError("google-generativeai is not installed.")
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
        [
            "You are helping to interpret exoplanet candidate predictions.",
            "Context JSON:",
            json.dumps(context, indent=2),
            "User prompt:",
            prompt,
        ]
    )
    return response.text if hasattr(response, "text") else "(No response received.)"


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
]
