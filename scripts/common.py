"""Shared utilities for training TESS exoplanet models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "tabular_data" / "TOI_2025.10.02_20.46.33.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

IDENTIFIER_COLUMNS = ("rowid", "toi", "toipfx", "ctoi_alias", "tid")


@dataclass
class Dataset:
    """Feature matrix and target vector."""

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame | None = None


def load_tess_dataset(
    path: Path = DATA_PATH,
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

    return Dataset(features=features, target=target, metadata=metadata)


def ensure_output_directories() -> None:
    """Create directories for models and reports if they do not already exist."""
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
