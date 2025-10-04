"""Interactive Streamlit dashboard for exoplanet candidate predictions.

Features
--------
- Load pre-trained models for the TESS, Kepler, and K2 catalogues, including the
    classic baselines (Random Forest, SVM, KNN, Logistic Regression, Decision Tree)
    and the newer gradient boosting family (Gradient Boosting, XGBoost, LightGBM,
    CatBoost) for each dataset.
- Run batched predictions on the bundled catalogues or on a user-provided CSV that
        matches one of the supported schemas.
- Summarise the results directly in the UI and optionally forward the summary plus a
    custom prompt to Gemini for natural-language responses.

Before running the app, make sure that:
1. The project virtual environment is active and `requirements.txt` has been installed.
2. The model artefacts (.joblib files) already exist in `models/` (run each training
   script once if needed).
3. The Gemini API key is stored in the `GEMINI_API_KEY` environment variable (or in
   `st.secrets["GEMINI_API_KEY"]`).

Launch locally with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency at runtime
    genai = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.common import (
    MODELS_DIR,
    REPORTS_DIR,
    load_k2_dataset,
    load_kepler_dataset,
    load_tess_dataset,
)

UPLOAD_OPTION_LABEL = "Upload CSV (choose schema)"

DATASET_CONFIG: Dict[str, Dict[str, object]] = {
    "tess": {
        "label": "TESS catalogue (bundled)",
        "loader": load_tess_dataset,
        "model_files": {
            "TESS Random Forest": "tess_random_forest.joblib",
            "TESS Support Vector Machine": "tess_svm.joblib",
            "TESS K-Nearest Neighbors": "tess_knn.joblib",
            "TESS Logistic Regression": "tess_logistic_regression.joblib",
            "TESS Decision Tree": "tess_decision_tree.joblib",
            "TESS Gradient Boosting": "tess_gbm.joblib",
            "TESS XGBoost": "tess_xgboost.joblib",
            "TESS LightGBM": "tess_lightgbm.joblib",
            "TESS CatBoost": "tess_catboost.joblib",
        },
        "default_models": (
            "TESS LightGBM",
            "TESS XGBoost",
            "TESS CatBoost",
            "TESS Gradient Boosting",
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
            "Kepler Support Vector Machine": "kepler_svm.joblib",
            "Kepler K-Nearest Neighbors": "kepler_knn.joblib",
            "Kepler Logistic Regression": "kepler_logistic_regression.joblib",
            "Kepler Decision Tree": "kepler_decision_tree.joblib",
            "Kepler Gradient Boosting": "kepler_gbm.joblib",
            "Kepler XGBoost": "kepler_xgboost.joblib",
            "Kepler LightGBM": "kepler_lightgbm.joblib",
            "Kepler CatBoost": "kepler_catboost.joblib",
        },
        "default_models": (
            "Kepler LightGBM",
            "Kepler XGBoost",
            "Kepler CatBoost",
            "Kepler Gradient Boosting",
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
            "K2 Support Vector Machine": "k2_svm.joblib",
            "K2 K-Nearest Neighbors": "k2_knn.joblib",
            "K2 Logistic Regression": "k2_logistic_regression.joblib",
            "K2 Decision Tree": "k2_decision_tree.joblib",
            "K2 Gradient Boosting": "k2_gbm.joblib",
            "K2 XGBoost": "k2_xgboost.joblib",
            "K2 LightGBM": "k2_lightgbm.joblib",
            "K2 CatBoost": "k2_catboost.joblib",
        },
        "default_models": (
            "K2 LightGBM",
            "K2 XGBoost",
            "K2 CatBoost",
            "K2 Gradient Boosting",
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

DEFAULT_TOP_K = 10

SUMMARY_COLUMN_CONFIG = {
    "rank": st.column_config.NumberColumn("Rank", format="%d"),
    "toi": st.column_config.TextColumn("TOI"),
    "tid": st.column_config.TextColumn("TIC ID"),
    "ctoi_alias": st.column_config.TextColumn("CTOI alias"),
    "tfopwg_disp": st.column_config.TextColumn("TFOP disposition"),
    "kepid": st.column_config.TextColumn("Kepler ID"),
    "kepoi_name": st.column_config.TextColumn("KOI name"),
    "kepler_name": st.column_config.TextColumn("Kepler planet name"),
    "koi_disposition": st.column_config.TextColumn("Disposition"),
    "tic_id": st.column_config.TextColumn("TIC ID"),
    "k2_name": st.column_config.TextColumn("K2 name"),
    "pl_name": st.column_config.TextColumn("Planet name"),
    "disposition": st.column_config.TextColumn("Disposition"),
    "sample_index": st.column_config.NumberColumn("Sample #", format="%d"),
    "planet_candidate_probability": st.column_config.NumberColumn(
        "Model probability",
        format="%.3f",
        help="Estimated probability that this object is a planet candidate.",
    ),
    "prediction_label": st.column_config.TextColumn(
        "Prediction",
        help="Model classification (confirmed vs. false positive).",
    ),
    "predicted_label_numeric": st.column_config.NumberColumn(
        "Prediction (numeric)",
        format="%d",
        help="Original numeric encoding (1 = confirmed, 0 = false positive).",
    ),
}


@st.cache_resource(show_spinner=False)
def load_models(dataset_key: str, model_names: Iterable[str]):
    """Load selected joblib pipelines for a given dataset and cache them across reruns."""
    model_files: Dict[str, str] = DATASET_CONFIG[dataset_key]["model_files"]  # type: ignore[index]
    models = {}
    for name in model_names:
        filename = model_files.get(name)
        if not filename:
            continue
        path = MODELS_DIR / filename
        if not path.exists():
            st.warning(f"Model artefact for '{name}' is missing: {path}")
            continue
        models[name] = load(path)
    return models


def metrics_filename_for_model(model_filename: str) -> str:
    """Return the expected metrics filename for a given model artefact."""
    stem = Path(model_filename).stem
    return f"{stem}_metrics.json"


def load_metrics_for_models(
    dataset_key: str,
    model_names: Iterable[str],
) -> tuple[Dict[str, Dict[str, object]], list[tuple[str, Path]]]:
    """Load evaluation metrics JSON for each selected model."""
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


@st.cache_data(show_spinner=False)
def load_dataset_bundle(dataset_key: str):
    """Load and cache the dataset bundle (features, target, metadata)."""
    loader = DATASET_CONFIG[dataset_key]["loader"]  # type: ignore[index]
    return loader()


@st.cache_data(show_spinner=False)
def load_dataset_features(dataset_key: str) -> pd.DataFrame:
    """Return the cached feature matrix for the selected dataset."""
    return load_dataset_bundle(dataset_key).features


@st.cache_data(show_spinner=False)
def load_dataset_metadata(dataset_key: str) -> pd.DataFrame:
    """Return metadata aligned with the cached dataset."""
    bundle = load_dataset_bundle(dataset_key)
    if bundle.metadata is None:
        return pd.DataFrame(index=bundle.features.index)
    return bundle.metadata.copy()


def preprocess_uploaded_file(uploaded_file, dataset_key: str) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Validate and coerce an uploaded CSV to match training features for a dataset."""
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
    """Run predictions for each model and return a dataframe of results."""
    results = {}
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


def configure_gemini() -> Optional[str]:
    """Configure the Gemini client and return the active API key if available."""
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        st.info(
            "Set the `GEMINI_API_KEY` environment variable (or add it to Streamlit secrets) "
            "to enable Gemini-assisted responses."
        )
        return None
    if genai is None:
        st.warning("google-generativeai is not installed. Gemini responses will be disabled.")
        return None
    genai.configure(api_key=api_key)
    return api_key


def call_gemini(model_name: str, prompt: str, context: Dict[str, object]) -> str:
    """Generate a Gemini response using the provided context."""
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


def build_context_for_ai(
    model_results: Dict[str, pd.DataFrame],
    top_k: Optional[int],
    dataset_label: str,
    identifier_priority: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Structure the model outputs for the LLM."""
    context = {"dataset": dataset_label}
    for model_name, df in model_results.items():
        top_rows = format_summary_table(df, top_k, identifier_priority)
        context[model_name] = top_rows.to_dict(orient="records")
        context[model_name + "_metrics"] = {
            "total_samples": int(df.shape[0]),
            "positive_predictions": int(df["predicted_label_numeric"].sum()),
            "mean_probability": float(df["planet_candidate_probability"].mean()),
        }
    return context


def main() -> None:
    st.set_page_config(page_title="Exoplanet Candidate Explorer", layout="wide")
    st.title("🔭 Exoplanet Candidate Explorer")
    st.caption(
        "Run pre-trained machine learning models to score objects of interest and ask Gemini "
        "for natural-language summaries."
    )

    with st.sidebar:
        st.header("Configuration")
        dataset_choice_label = st.selectbox(
            "Data source",
            DATASET_LABELS + [UPLOAD_OPTION_LABEL],
            help="Uploaded files must contain the same numeric columns used during training.",
        )

        is_upload = dataset_choice_label == UPLOAD_OPTION_LABEL
        if is_upload:
            schema_label = st.radio(
                "Schema for uploaded data",
                DATASET_LABELS,
                index=0,
            )
            dataset_key = DATASET_LABEL_TO_KEY[schema_label]
            dataset_label_display = f"Uploaded CSV ({schema_label})"
        else:
            dataset_key = DATASET_LABEL_TO_KEY[dataset_choice_label]
            dataset_label_display = DATASET_CONFIG[dataset_key]["label"]  # type: ignore[index]

        dataset_config = DATASET_CONFIG[dataset_key]
        model_options = list(dataset_config["model_files"].keys())  # type: ignore[index]

        selected_models = st.multiselect(
            "Models to run",
            options=model_options,
            default=list(dataset_config["default_models"]),  # type: ignore[index]
            help="Pilih satu atau lebih model yang tersedia untuk dijalankan.",
        )

        gemini_model_name = st.text_input(
            "Gemini model name",
            value="models/gemini-2.5-flash-lite",
            help=(
                "Use a fully qualified model name such as 'models/gemini-2.5-flash-lite'. "
                "Refer to https://ai.google.dev/api/rest/v1beta/models/list for available options."
            ),
        )

    top_k: Optional[int] = None

    if not selected_models:
        st.warning("Please select at least one model before running predictions.")
        st.stop()

    models = load_models(dataset_key, tuple(selected_models))
    if not models:
        st.error(
            "None of the requested models could be loaded. Ensure the training scripts ran successfully."
        )
        st.stop()

    model_metrics, missing_metrics = load_metrics_for_models(dataset_key, models.keys())
    if missing_metrics:
        formatted_missing = "\n".join(
            f"• {name}: {path.name}" for name, path in missing_metrics
        )
        st.warning(
            "Metrics files were not found for some models. Predictions will still run, but evaluation "
            "details may be incomplete:\n" + formatted_missing
        )

    # Load dataset
    metadata: Optional[pd.DataFrame] = None

    if not is_upload:
        bundle = load_dataset_bundle(dataset_key)
        features = bundle.features
        metadata = bundle.metadata if bundle.metadata is not None else None
        st.success(
            f"Loaded {dataset_label_display} with {features.shape[0]} samples and {features.shape[1]} features."
        )
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
        if not uploaded_file:
            st.info("Upload a CSV file that matches the training schema to continue.")
            st.stop()
        try:
            features, metadata = preprocess_uploaded_file(uploaded_file, dataset_key)
        except ValueError as exc:  # surface validation errors in the UI
            st.error(str(exc))
            st.stop()
        st.success(
            f"Uploaded dataset accepted with {features.shape[0]} samples and {features.shape[1]} features using the {dataset_config['label']} schema."
        )

    with st.spinner("Running model predictions..."):
        model_outputs = run_predictions(models, features, metadata)

    tabs = st.tabs(list(model_outputs.keys()))
    for tab, (model_name, df_predictions) in zip(tabs, model_outputs.items()):
        with tab:
            st.subheader(model_name)
            metrics_payload = model_metrics.get(model_name)
            if metrics_payload:
                st.markdown("#### Evaluation overview")
                metric_labels = [
                    ("Accuracy", "accuracy"),
                    ("Precision", "precision"),
                    ("Recall", "recall"),
                    ("F1 score", "f1"),
                    ("ROC AUC", "roc_auc"),
                ]
                cols = st.columns(len(metric_labels))
                for col, (label, key) in zip(cols, metric_labels):
                    value = metrics_payload.get(key)
                    display_value = f"{value:.3f}" if isinstance(value, (int, float)) else "–"
                    col.metric(label, display_value)

                with st.expander("Classification report", expanded=False):
                    report = metrics_payload.get("classification_report")
                    if isinstance(report, dict):
                        st.json(report)
                    else:
                        st.info("Classification report not available.")

                with st.expander("Cross-validation details", expanded=False):
                    cv = metrics_payload.get("cross_validation")
                    if isinstance(cv, dict) and cv:
                        cv_df = pd.DataFrame(cv).T
                        st.dataframe(cv_df, use_container_width=True)
                    else:
                        st.info("Cross-validation summary not available.")

                with st.expander("Model parameters", expanded=False):
                    params = metrics_payload.get("model_params")
                    if isinstance(params, dict):
                        st.json(params)
                    else:
                        st.info("Model parameters not recorded.")

            st.markdown("#### Predictions")
            st.metric(
                label="Mean candidate probability",
                value=f"{df_predictions['planet_candidate_probability'].mean():.3f}",
            )
            st.metric(
                label="Predicted candidates",
                value=int(df_predictions["predicted_label_numeric"].sum()),
                delta=f"{df_predictions['predicted_label_numeric'].mean():.1%} of total",
            )
            summary_table = format_summary_table(
                df_predictions,
                top_k,
                dataset_config["identifier_priority"],  # type: ignore[index]
            )
            st.dataframe(
                summary_table,
                width="stretch",
                column_config=SUMMARY_COLUMN_CONFIG,
                hide_index=True,
            )
            st.caption(
                "Kolom 'prediction_label' menampilkan hasil klasifikasi: 'confirmed' berarti objek diyakini sebagai planet terkonfirmasi, sedangkan 'false positive' menandakan bukan planet. Kolom numerik tetap tersedia untuk referensi (1 = confirmed, 0 = false positive)."
            )

    st.divider()
    st.header("Ask Gemini about the predictions")
    api_key = configure_gemini()

    user_prompt = st.text_area(
        "Prompt",
        placeholder="e.g. Ringkas 10 kandidat teratas dan jelaskan metrik pentingnya.",
        help="Context from the predictions will be appended automatically.",
    )

    if st.button("Generate Gemini response", type="primary", disabled=not api_key or genai is None):
        try:
            context_payload = build_context_for_ai(
                model_outputs,
                top_k,
                dataset_label_display,
                dataset_config["identifier_priority"],  # type: ignore[index]
            )
            response_text = call_gemini(gemini_model_name, user_prompt, context_payload)
            st.subheader("Gemini response")
            st.write(response_text)
        except Exception as exc:  # pragma: no cover - runtime interaction only
            st.error(f"Gemini call failed: {exc}")
    else:
        st.info("Enter a prompt and click the button once Gemini is configured.")


if __name__ == "__main__":
    main()
