"""Shared utilities for training TESS exoplanet models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "tabular_data" / "TOI_2025.10.02_20.46.33.csv"
TESS_DATA_PATH = DATA_PATH
KEPLER_DATA_PATH = PROJECT_ROOT / "tabular_data" / "cumulative_2025.10.02_21.17.01.csv"
K2_DATA_PATH = PROJECT_ROOT / "tabular_data" / "k2pandc_2025.10.02_20.47.13.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

IDENTIFIER_COLUMNS = ("rowid", "toi", "toipfx", "ctoi_alias", "tid")
KEPLER_IDENTIFIER_COLUMNS = ("rowid", "kepid")
K2_IDENTIFIER_COLUMNS = ("rowid", "tic_id")

TOP_FEATURES: dict[str, tuple[str, ...]] = {
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


def _restrict_to_top_features(features: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    """Return a dataframe limited to the top-ranked features for the dataset."""
    selected = TOP_FEATURES.get(dataset_key)
    if not selected:
        return features

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


def load_tess_dataset(
    path: Path = TESS_DATA_PATH,
    drop_identifier_columns: Iterable[str] | None = IDENTIFIER_COLUMNS,
) -> Dataset:
    """Load the TESS TOI catalogue and return cleaned numeric features and binary target."""
    df = pd.read_csv(path, na_values=["", " "])
    if "tfopwg_disp" not in df:
        raise KeyError("Column 'tfopwg_disp' is required in the dataset")

    df = df.dropna(subset=["tfopwg_disp"])

    # Binary label: Planet Candidate (PC) vs. everything else
    target = (df["tfopwg_disp"].str.upper() == "PC").astype(int)

    metadata_columns = [col for col in IDENTIFIER_COLUMNS if col != "rowid"] + [
        "tfopwg_disp"
    ]
    metadata = df[[col for col in metadata_columns if col in df.columns]].copy()

    numeric_columns = df.select_dtypes(include=["number"]).columns
    features = df[numeric_columns].copy()

    if drop_identifier_columns:
        drop_cols = [col for col in drop_identifier_columns if col in features.columns]
        if drop_cols:
            features.drop(columns=drop_cols, inplace=True)

    # Remove columns with no information
    features.dropna(axis=1, how="all", inplace=True)
    constant_columns = [col for col in features.columns if features[col].nunique(dropna=False) <= 1]
    if constant_columns:
        features.drop(columns=constant_columns, inplace=True)

    metadata = metadata.reindex(features.index)

    features = _restrict_to_top_features(features, "tess")

    return Dataset(features=features, target=target, metadata=metadata)


def _prepare_binary_dataset(
    df: pd.DataFrame,
    label_column: str,
    positive_labels: Sequence[str],
    negative_labels: Sequence[str],
    drop_identifier_columns: Iterable[str] | None,
    metadata_columns: Iterable[str],
) -> Dataset:
    """Utility to create a balanced Dataset object for binary classification."""

    if label_column not in df:
        raise KeyError(f"Column '{label_column}' is required in the dataset")

    df = df.dropna(subset=[label_column])

    label_series = df[label_column].astype(str).str.upper()
    positive_set = {label.upper() for label in positive_labels}
    negative_set = {label.upper() for label in negative_labels}

    allowed = positive_set | negative_set
    mask = label_series.isin(allowed)
    df = df.loc[mask].copy()
    label_series = label_series.loc[mask]

    target = label_series.isin(positive_set).astype(int)

    metadata = None
    metadata_cols = [col for col in metadata_columns if col in df.columns]
    if metadata_cols:
        metadata = df[metadata_cols].copy()

    numeric_columns = df.select_dtypes(include=["number"]).columns
    features = df[numeric_columns].copy()

    if drop_identifier_columns:
        drop_cols = [col for col in drop_identifier_columns if col in features.columns]
        if drop_cols:
            features.drop(columns=drop_cols, inplace=True)

    features.dropna(axis=1, how="all", inplace=True)
    constant_columns = [col for col in features.columns if features[col].nunique(dropna=False) <= 1]
    if constant_columns:
        features.drop(columns=constant_columns, inplace=True)

    if metadata is not None:
        metadata = metadata.reindex(features.index)

    target = target.reindex(features.index)

    return Dataset(features=features, target=target, metadata=metadata)


def load_kepler_dataset(
    path: Path = KEPLER_DATA_PATH,
    drop_identifier_columns: Iterable[str] | None = KEPLER_IDENTIFIER_COLUMNS,
) -> Dataset:
    """Load the Kepler cumulative KOI catalogue and return features/labels."""
    df = pd.read_csv(path, na_values=["", " "])
    metadata_columns = [
        "kepid",
        "kepoi_name",
        "kepler_name",
        "koi_disposition",
    ]
    dataset = _prepare_binary_dataset(
        df,
        label_column="koi_disposition",
        positive_labels=["CONFIRMED"],
        negative_labels=["FALSE POSITIVE"],
        drop_identifier_columns=drop_identifier_columns,
        metadata_columns=metadata_columns,
    )
    dataset.features = _restrict_to_top_features(dataset.features, "kepler")
    return dataset


def load_k2_dataset(
    path: Path = K2_DATA_PATH,
    drop_identifier_columns: Iterable[str] | None = K2_IDENTIFIER_COLUMNS,
) -> Dataset:
    """Load the K2 planet candidate catalogue and return features/labels."""
    df = pd.read_csv(path, na_values=["", " "])
    metadata_columns = [
        "tic_id",
        "k2_name",
        "pl_name",
        "disposition",
    ]
    dataset = _prepare_binary_dataset(
        df,
        label_column="disposition",
        positive_labels=["CONFIRMED"],
        negative_labels=["FALSE POSITIVE"],
        drop_identifier_columns=drop_identifier_columns,
        metadata_columns=metadata_columns,
    )
    dataset.features = _restrict_to_top_features(dataset.features, "k2")
    return dataset


def ensure_output_directories() -> None:
    """Create directories for models and reports if they do not already exist."""
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
