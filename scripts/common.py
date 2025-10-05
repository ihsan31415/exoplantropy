"""Shared utilities for training exoplanet models from combined dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXOPLANTROPY_MAIN_DIR = PROJECT_ROOT / "exoplantropy-main"
DATA_PATH = EXOPLANTROPY_MAIN_DIR / "tabular_data" / "combined_all.csv"
TESS_DATA_PATH = DATA_PATH
KEPLER_DATA_PATH = DATA_PATH
K2_DATA_PATH = DATA_PATH
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = EXOPLANTROPY_MAIN_DIR / "models"

# Identifier columns from combined_all.csv
IDENTIFIER_COLUMNS = ("unified_id", "stellar_id", "disposition", "mission")

# Standard features from combined_all.csv (10 features used for training)
STANDARD_FEATURES = (
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

TOP_FEATURES: dict[str, tuple[str, ...]] = {
    "tess": STANDARD_FEATURES,
    "kepler": STANDARD_FEATURES,
    "k2": STANDARD_FEATURES,
}


def _restrict_to_top_features(features: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Return a dataframe limited to the standard features."""
    selected = TOP_FEATURES.get(dataset_key, STANDARD_FEATURES)
    available_columns = [col for col in selected if col in features.columns]
    if not available_columns:
        return features
    return features[available_columns].copy()


@dataclass
class Dataset:
    """Feature matrix and target vector."""

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame | None = None


class DataFrameSimpleImputer(SimpleImputer):
    """SimpleImputer variant that preserves pandas DataFrame structure."""

    def transform(self, X):
        result = super().transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        return result

    def fit_transform(self, X, y=None):
        result = super().fit_transform(X, y)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        return result


def load_combined_dataset(
    path: Path = DATA_PATH,
    drop_identifier_columns: Iterable[str] | None = None,
) -> Dataset:
    """Load the combined exoplanet dataset and return features/labels."""
    df = pd.read_csv(path, na_values=["", " "])
    # --- CRITICAL FIX: Ensure all column names are strings ---
    df.columns = df.columns.astype(str)
    
    label_column = "disposition"
    if label_column not in df:
        raise KeyError(f"Column '{label_column}' is required in the combined dataset")

    df = df.dropna(subset=[label_column])

    # Binary label: CONFIRMED vs. FALSE_POSITIVE
    positive_labels = {"CONFIRMED"}
    allowed_labels = {"CONFIRMED", "FALSE_POSITIVE", "CANDIDATE"}
    
    label_series = df[label_column].astype(str).str.upper()
    mask_allowed = label_series.isin(allowed_labels)
    
    df = df.loc[mask_allowed].copy()
    target = label_series.loc[mask_allowed].isin(positive_labels).astype(int)

    # Metadata (identifiers)
    metadata_cols = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
    metadata = df[metadata_cols].copy() if metadata_cols else None

    # Features - select only the standard 10 features
    available_features = [col for col in STANDARD_FEATURES if col in df.columns]
    if not available_features:
        raise KeyError(f"No standard features found in dataset. Expected: {STANDARD_FEATURES}")
    
    features = df[available_features].copy()

    # Ensure indices match
    if metadata is not None:
        metadata = metadata.reindex(features.index)
    target = target.reindex(features.index)

    return Dataset(features=features, target=target, metadata=metadata)


def load_tess_dataset(
    path: Path = DATA_PATH,
    drop_identifier_columns: Iterable[str] | None = None,
) -> Dataset:
    """Load the combined dataset and return cleaned numeric features and binary target."""
    return load_combined_dataset(path, drop_identifier_columns)


def load_kepler_dataset(
    path: Path = DATA_PATH,
    drop_identifier_columns: Iterable[str] | None = None,
) -> Dataset:
    """Load the combined dataset and return features/labels."""
    return load_combined_dataset(path, drop_identifier_columns)


def load_k2_dataset(
    path: Path = DATA_PATH,
    drop_identifier_columns: Iterable[str] | None = None,
) -> Dataset:
    """Load the combined dataset and return features/labels."""
    return load_combined_dataset(path, drop_identifier_columns)


def ensure_output_directories() -> None:
    """Create directories for models and reports if they do not already exist."""
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
