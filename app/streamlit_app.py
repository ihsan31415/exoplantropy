"""Interactive Streamlit dashboard for exoplanet candidate predictions.

Features
--------
- Load the pre-trained TESS models (Random Forest, SVM, KNN, Logistic Regression, Decision Tree).
- Run batched predictions on the shipped TESS dataset or on a user-provided CSV file
  containing identical feature columns.
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

from scripts.common import DATA_PATH, MODELS_DIR, load_tess_dataset

MODEL_FILES: Dict[str, str] = {
    "Random Forest": "tess_random_forest.joblib",
    "Support Vector Machine": "tess_svm.joblib",
    "K-Nearest Neighbors": "tess_knn.joblib",
    "Logistic Regression": "tess_logistic_regression.joblib",
    "Decision Tree": "tess_decision_tree.joblib",
    # ExoMiner Pipeline model
    "ExoMiner MLP": "tess_exominer.joblib",
}

DEFAULT_TOP_K = 10

SUMMARY_COLUMN_CONFIG = {
    "rank": st.column_config.NumberColumn("Rank", format="%d"),
    "toi": st.column_config.TextColumn("TOI"),
    "tid": st.column_config.TextColumn("TIC ID"),
    "ctoi_alias": st.column_config.TextColumn("CTOI alias"),
    "tfopwg_disp": st.column_config.TextColumn("TFOP disposition"),
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


def load_models(model_names: Iterable[str]):
    """Load selected joblib pipelines and cache them across reruns."""
    models = {}
    for name in model_names:
        filename = MODEL_FILES.get(name)
        if not filename:
            continue
        path = MODELS_DIR / filename
        if not path.exists():
            st.warning(f"Model artefact for '{name}' is missing: {path}")
            continue
        # Load classical scikit-learn pipelines
        if filename.endswith('.joblib'):
            models[name] = load(path)
        # Load Keras model and wrap for predict/predict_proba
        elif filename.endswith(('.h5', '.keras')):
            from tensorflow import keras
            from joblib import load as jl_load
            # Load Keras model and corresponding scaler
            keras_model = keras.models.load_model(str(path))
            scaler_path = MODELS_DIR / 'tess_exominer_scaler.joblib'
            scaler = jl_load(scaler_path) if scaler_path.exists() else None
            # Custom wrapper to apply scaling then predict
            class ExoMinerPipeline:
                def __init__(self, model, scaler=None):
                    self.model = model
                    self.scaler = scaler
                def predict(self, X):
                    Xs = self.scaler.transform(X) if self.scaler else X
                    proba = self.model.predict(Xs).ravel()
                    return (proba > 0.5).astype(int)
                def predict_proba(self, X):
                    Xs = self.scaler.transform(X) if self.scaler else X
                    proba = self.model.predict(Xs).ravel()
                    return np.vstack([1 - proba, proba]).T
            models[name] = ExoMinerPipeline(keras_model, scaler)
        else:
            st.warning(f"Unrecognized model format for '{name}': {filename}")
    return models


@st.cache_data(show_spinner=False)
def load_default_dataset_bundle():
    """Load the cached TESS dataset once to share features and metadata."""
    return load_tess_dataset()


@st.cache_data(show_spinner=False)
def load_default_dataset() -> pd.DataFrame:
    """Return the cleaned TESS feature matrix (without labels)."""
    return load_default_dataset_bundle().features


@st.cache_data(show_spinner=False)
def load_default_metadata() -> pd.DataFrame:
    """Return identifier metadata aligned with the default dataset."""
    bundle = load_default_dataset_bundle()
    if bundle.metadata is None:
        return pd.DataFrame(index=bundle.features.index)
    return bundle.metadata.copy()


def preprocess_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Validate and coerce an uploaded CSV to match training features."""
    df = pd.read_csv(uploaded_file)
    required_columns = load_default_dataset().columns
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded file is missing required feature columns: " + ", ".join(missing)
        )
    # Re-order and select only the columns used during training
    df = df[required_columns]
    return df


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
        preds = pipeline.predict(features).ravel()
        df = pd.DataFrame(
            {
                "planet_candidate_probability": proba,
                "predicted_label": preds,
            },
            index=features.index,
        )
        enriched = metadata.join(df, how="inner")
        enriched["prediction_label"] = np.where(
            enriched["predicted_label"] == 1,
            "confirmed",
            "false positive",
        )
        results[name] = enriched
    return results


def format_summary_table(
    df: pd.DataFrame,
    top_k: Optional[int] = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Return the highest-probability rows (or all rows if ``top_k`` is ``None``)."""
    identifier_columns = [
        col
        for col in ("toi", "tid", "ctoi_alias", "tfopwg_disp", "sample_index")
        if col in df.columns
    ]
    df_display = df.copy()
    if "predicted_label" in df_display.columns and "predicted_label_numeric" not in df_display.columns:
        df_display["predicted_label_numeric"] = df_display["predicted_label"]

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
) -> Dict[str, object]:
    """Structure the model outputs for the LLM."""
    context = {}
    for model_name, df in model_results.items():
        top_rows = format_summary_table(df, top_k)
        context[model_name] = top_rows.to_dict(orient="records")
        context[model_name + "_metrics"] = {
            "total_samples": int(df.shape[0]),
            "positive_predictions": int(df["predicted_label"].sum()),
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
        dataset_choice = st.selectbox(
            "Data source",
            (
                "TESS catalogue (bundled)",
                "Upload CSV (matching training schema)",
            ),
            help="Uploaded files must contain the same numeric columns used during training.",
        )

        selected_models = st.multiselect(
            "Models to evaluate",
            list(MODEL_FILES.keys()),
            default=list(MODEL_FILES.keys())[:3],
        )

        limit_mode = st.radio(
            "Jumlah baris yang ingin disertakan",
            ("Top N", "Semua baris"),
            index=0,
        )

        if limit_mode == "Top N":
            top_k: Optional[int] = st.slider(
                "Pilih N (baris dengan probabilitas tertinggi)",
                min_value=3,
                max_value=200,
                value=DEFAULT_TOP_K,
            )
        else:
            top_k = None

        gemini_model_name = st.text_input(
            "Gemini model name",
            value="models/gemini-2.5-flash-lite",
            help=(
                "Use a fully qualified model name such as 'models/gemini-2.5-flash-lite'. "
                "Refer to https://ai.google.dev/api/rest/v1beta/models/list for available options."
            ),
        )

    if not selected_models:
        st.warning("Please select at least one model before running predictions.")
        st.stop()

    models = load_models(selected_models)
    if not models:
        st.error("None of the requested models could be loaded. Ensure the training scripts ran successfully.")
        st.stop()

    # Load dataset
    metadata: Optional[pd.DataFrame] = None

    if dataset_choice.startswith("TESS"):
        features = load_default_dataset()
        metadata = load_default_metadata()
        st.success(
            f"Loaded TESS dataset with {features.shape[0]} samples and {features.shape[1]} features."
        )
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
        if not uploaded_file:
            st.info("Upload a CSV file that matches the training schema to continue.")
            st.stop()
        try:
            features = preprocess_uploaded_file(uploaded_file)
        except ValueError as exc:  # surface validation errors in the UI
            st.error(str(exc))
            st.stop()
        st.success(
            f"Uploaded dataset accepted with {features.shape[0]} samples and {features.shape[1]} features."
        )

    with st.spinner("Running model predictions..."):
        model_outputs = run_predictions(models, features, metadata)

    tabs = st.tabs(list(model_outputs.keys()))
    for tab, (model_name, df_predictions) in zip(tabs, model_outputs.items()):
        with tab:
            st.subheader(model_name)
            st.metric(
                label="Mean candidate probability",
                value=f"{df_predictions['planet_candidate_probability'].mean():.3f}",
            )
            st.metric(
                label="Predicted candidates",
                value=int(df_predictions["predicted_label"].sum()),
                delta=f"{df_predictions['predicted_label'].mean():.1%} of total",
            )
            summary_table = format_summary_table(df_predictions, top_k)
            st.dataframe(
                summary_table,
                width="stretch",
                column_config=SUMMARY_COLUMN_CONFIG,
                hide_index=True,
            )
            if top_k is None:
                st.caption(
                    "Menampilkan seluruh baris. Pertimbangkan untuk kembali ke 'Top N' bila performa menurun."
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
            context_payload = build_context_for_ai(model_outputs, top_k)
            response_text = call_gemini(gemini_model_name, user_prompt, context_payload)
            st.subheader("Gemini response")
            st.write(response_text)
        except Exception as exc:  # pragma: no cover - runtime interaction only
            st.error(f"Gemini call failed: {exc}")
    else:
        st.info("Enter a prompt and click the button once Gemini is configured.")


if __name__ == "__main__":
    main()
