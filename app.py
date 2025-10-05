from __future__ import annotations

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from services.exoplanet_service import (
    DATASET_CONFIG,
    DATASET_LABELS,
    DATASET_LABEL_TO_KEY,
    DEFAULT_TOP_K,
    FEATURE_DESCRIPTIONS,
    FEATURE_DISPLAY_ORDER,
    UPLOAD_OPTION_LABEL,
    available_models,
    build_context_for_ai,
    configure_gemini,
    dataset_feature_statistics,
    generate_transit_simulation,
    default_models,
    format_summary_table,
    load_dataset_bundle,
    load_dataset_features,
    load_dataset_metadata,
    load_metrics_for_models,
    load_models,
    call_gemini,
    preprocess_uploaded_file,
    run_predictions,
    save_trained_model,
    train_custom_model,
)


TEAM_MEMBERS: List[Dict[str, str]] = [
    {
        "name": "Muhammad Khoirul Ihsan",
        "role": "Universitas Negeri Semarang",
        "photo": "img/team/member1.jpg",
        "linkedin": "https://www.linkedin.com/in/Khoirul-ihsan-syntropy/",
    },
    {
        "name": "Riski Yuniar Pratama",
        "role": "Universitas Negeri Semarang",
        "photo": "img/team/member2.jpg",
        "linkedin": "https://www.linkedin.com/in/riski-yuniar-a08851184/",
    },
    {
        "name": "Lyon Ambrosio Djuanda",
        "role": "Universitas Negeri Semarang",
        "photo": "img/team/member3.jpg",
        "linkedin": "https://www.linkedin.com/in/lyon-ambrosio-djuanda-567298287/",
    },
    {
        "name": "Fikri Achmad Fadilah",
        "role": "Universitas Negeri Semarang",
        "photo": "img/team/member4.jpg",
        "linkedin": "https://www.linkedin.com/in/fikri-achmad-fadilah-7659b22bb/",
    },
    {
        "name": "Sayyid Muhammad Muslim As'ad Sunarko",
        "role": "Universitas Bina Nusantara",
        "photo": "img/team/member5.jpg",
        "linkedin": "https://www.linkedin.com/in/sayyid-muhammad/",
    },
    {
        "name": "Muhammad Dany Hidayat",
        "role": "Universitas Negeri Semarang",
        "photo": "img/team/member6.jpg",
        "linkedin": "https://www.linkedin.com/in/muhammad-dany-hidayat-740091385/",
    },
]


SUPERVISORS: List[Dict[str, str]] = [
    {
        "name": "BRIN",
        "title": "Supervising Mentor",
        "intro": "BRIN has been instrumental in guiding our research direction and providing valuable insights into exoplanet studies.",
        "photo": "img/team/supervisor1.webp",
        "linkedin": "https://www.linkedin.com/in/supervisor",
    }
]


THANKS_TO: List[Dict[str, str]] = [
    {
        "name": "BRIN",
        "photo": "img/team/brin.jpg",
        "link": "https://www.brin.go.id/",
        "caption": "National Research and Innovation Agency",
        "thanks": "thanks to Mr. Thomas djamaludin and Mr. Immanuel Sungging Mumpuni for the insight about the exoplanet data"
    },
    {
        "name": "NASA",
        "photo": "img/team/nasa.jpg",
        "link": "https://www.nasa.gov/",
        "caption": "National Aeronautics and Space Administration",
        "thanks": "thanks to Mr. Irwan Prabowo on the insight about how to deliver the data to the user"
    },
]


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-entropy")

    @app.context_processor
    def inject_globals():
        return {
            "dataset_labels": DATASET_LABELS,
            "dataset_label_to_key": DATASET_LABEL_TO_KEY,
            "default_dataset_key": "tess",
            "upload_option_label": UPLOAD_OPTION_LABEL,
        }

    def _resolve_dataset_key(raw_key: str | None) -> str:
        if not raw_key:
            return "tess"
        if raw_key in DATASET_CONFIG:
            return raw_key
        if raw_key in DATASET_LABEL_TO_KEY:
            return DATASET_LABEL_TO_KEY[raw_key]
        return "tess"

    @app.route("/")
    def index():
        key = request.args.get("dataset")
        dataset_key = _resolve_dataset_key(key)
        stats = dataset_feature_statistics(dataset_key)
        feature_order = FEATURE_DISPLAY_ORDER.get(dataset_key, tuple(stats.index))
        ordered_features = [name for name in feature_order if name in stats.index]
        if not ordered_features:
            ordered_features = list(stats.index)
        stats = stats.loc[ordered_features]
        feature_preview = stats.head(5).reset_index().to_dict(orient="records")
        models = available_models(dataset_key)
        return render_template(
            "index.html",
            dataset_key=dataset_key,
            dataset_label=DATASET_CONFIG[dataset_key]["label"],
            feature_preview=feature_preview,
            model_count=len(models),
            feature_descriptions=FEATURE_DESCRIPTIONS.get(dataset_key, {}),
            feature_order=ordered_features,
        )

    @app.route("/about")
    def about():
        return render_template(
            "about.html",
            team_members=TEAM_MEMBERS,
            supervisors=SUPERVISORS,
            thanks_to=THANKS_TO,
        )

    @app.route("/manual", methods=["GET", "POST"])
    def manual():
        dataset_key = _resolve_dataset_key(
            request.form.get("dataset_key") or request.args.get("dataset")
        )
        stats = dataset_feature_statistics(dataset_key)
        feature_order = FEATURE_DISPLAY_ORDER.get(dataset_key, tuple(stats.index))
        ordered_features = [name for name in feature_order if name in stats.index]
        if not ordered_features:
            ordered_features = list(stats.index)
        stats = stats.loc[ordered_features]
        desc_map = FEATURE_DESCRIPTIONS.get(dataset_key, {})
        feature_list = [
            {
                "name": feature,
                "median": float(stats.loc[feature, "median"]),
                "std": float(stats.loc[feature, "std"]),
                "description": desc_map.get(feature, ""),
            }
            for feature in ordered_features
        ]

        models_available = list(available_models(dataset_key))
        raw_selection = request.form.getlist("models")
        selected_models = [name for name in raw_selection if name in models_available]
        if not selected_models:
            selected_models = list(default_models(dataset_key))

        prediction_results: List[Dict[str, object]] | None = None
        missing_models: List[str] = []
        manual_inputs: Dict[str, float] | None = None
        transit_data: Dict[str, object] | None = None

        if request.method == "POST":
            try:
                manual_inputs = {}
                for feature in ordered_features:
                    field_name = f"feature_{feature}"
                    raw_value = request.form.get(field_name, "")
                    if raw_value.strip() == "":
                        manual_inputs[feature] = float(stats.loc[feature, "median"])
                    else:
                        manual_inputs[feature] = float(raw_value)

                # Ensure all features from stats.index are present
                for feature in stats.index:
                    if feature not in manual_inputs:
                        manual_inputs[feature] = float(stats.loc[feature, "median"])

                model_map, missing = load_models(dataset_key, selected_models)
                missing_models = [name for name, _ in missing]
                if not model_map:
                    flash("Tidak ada model yang berhasil dimuat. Periksa artefak di folder models/.", "error")
                else:
                    # Ensure the DataFrame is created with the correct column order
                    manual_df = pd.DataFrame([manual_inputs], columns=stats.index)
                    predictions = run_predictions(model_map, manual_df)
                    prediction_results = []
                    for model_name, df in predictions.items():
                        row = df.iloc[0]
                        prediction_results.append(
                            {
                                "model": model_name,
                                "probability": float(row["planet_candidate_probability"]),
                                "label": row["prediction_label"],
                            }
                        )
                    prediction_results.sort(key=lambda item: item["probability"], reverse=True)
                    if manual_inputs:
                        try:
                            transit_data = generate_transit_simulation(dataset_key, manual_inputs, stats)
                        except Exception as exc:  # pragma: no cover - runtime guard
                            flash(f"Gagal membuat simulasi transit: {exc}", "warning")
            except ValueError as exc:
                flash(str(exc), "error")
            except Exception as exc:  # pragma: no cover - runtime guard
                import logging
                logging.basicConfig(level=logging.INFO)
                logging.error(f"Error during manual prediction: {exc}", exc_info=True)
                if 'manual_inputs' in locals():
                    logging.info(f"Manual inputs received: {manual_inputs}")
                if 'manual_df' in locals():
                    logging.info(f"DataFrame shape for prediction: {manual_df.shape}")
                    logging.info(f"DataFrame columns: {manual_df.columns.tolist()}")
                    logging.info(f"DataFrame content:\n{manual_df.head().to_string()}")
                flash(f"Gagal menjalankan prediksi: {exc}", "error")

        return render_template(
            "manual.html",
            dataset_key=dataset_key,
            dataset_label=DATASET_CONFIG[dataset_key]["label"],
            features=feature_list,
            models=models_available,
            selected_models=selected_models,
            predictions=prediction_results,
            missing_models=missing_models,
            transit_data=transit_data,
        )

    @app.route("/batch", methods=["GET", "POST"])
    def batch():
        dataset_key = _resolve_dataset_key(
            request.form.get("dataset_key") or request.args.get("dataset")
        )
        feature_descriptions = FEATURE_DESCRIPTIONS.get(dataset_key, {})
        feature_order = FEATURE_DISPLAY_ORDER.get(dataset_key, tuple())
        models_available = list(available_models(dataset_key))
        raw_selection = request.form.getlist("models")
        selected_models = [name for name in raw_selection if name in models_available]
        if not selected_models:
            selected_models = list(default_models(dataset_key))
        data_source = request.form.get("data_source", "bundled")

        top_k_raw = request.form.get("top_k")
        if top_k_raw is None or top_k_raw.strip() == "":
            top_k_limit = DEFAULT_TOP_K
            top_k_input_value = DEFAULT_TOP_K
        else:
            try:
                parsed_top_k = int(float(top_k_raw))
            except (TypeError, ValueError):
                parsed_top_k = DEFAULT_TOP_K

            if parsed_top_k <= 0:
                top_k_limit = None
                top_k_input_value = 0
            else:
                top_k_limit = parsed_top_k
                top_k_input_value = parsed_top_k

        summaries: Dict[str, List[Dict[str, object]]] | None = None
        missing_models: List[str] = []
        download_token: str | None = None

        if request.method == "POST":
            try:
                model_map, missing = load_models(dataset_key, selected_models)
                missing_models = [name for name, _ in missing]
                if not model_map:
                    flash("Tidak ada model yang tersedia untuk prediksi.", "error")
                else:
                    if data_source == "bundled":
                        bundle = load_dataset_bundle(dataset_key)
                        features = bundle.features
                        metadata = bundle.metadata
                    else:
                        uploaded_file = request.files.get("data_file")
                        if uploaded_file is None or uploaded_file.filename == "":
                            flash("Unggah file CSV terlebih dahulu.", "error")
                            return redirect(url_for("batch", dataset=dataset_key))
                        features, metadata = preprocess_uploaded_file(uploaded_file, dataset_key)

                    predictions = run_predictions(model_map, features, metadata)
                    identifier_priority = DATASET_CONFIG[dataset_key]["identifier_priority"]  # type: ignore[index]
                    summaries = {}
                    for model_name, df in predictions.items():
                        summary_df = format_summary_table(df, top_k_limit, identifier_priority)
                        summaries[model_name] = summary_df.to_dict(orient="records")

                    # Save combined CSV to in-memory buffer for download link
                    csv_buffer = io.BytesIO()
                    export_frames = []
                    for model_name, df in predictions.items():
                        labelled = df.copy()
                        labelled["model_name"] = model_name
                        export_frames.append(labelled)
                    combined = pd.concat(export_frames, axis=0)
                    combined.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    token = f"predictions_{dataset_key}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                    app.config.setdefault("DOWNLOAD_CACHE", {})[token] = csv_buffer
                    download_token = token
            except ValueError as exc:
                flash(str(exc), "error")
            except Exception as exc:  # pragma: no cover - runtime guard
                flash(f"Prediksi batch gagal: {exc}", "error")

        return render_template(
            "batch.html",
            dataset_key=dataset_key,
            dataset_label=DATASET_CONFIG[dataset_key]["label"],
            models=models_available,
            selected_models=selected_models,
            data_source=data_source,
            top_k=top_k_limit,
            top_k_input_value=top_k_input_value,
            summaries=summaries,
            missing_models=missing_models,
            download_token=download_token,
            feature_descriptions=feature_descriptions,
            feature_order=feature_order,
        )

    @app.route("/download/<token>")
    def download_predictions(token: str):
        cache: Dict[str, io.BytesIO] = app.config.get("DOWNLOAD_CACHE", {})
        if token not in cache:
            flash("File unduhan tidak ditemukan atau sudah kadaluarsa.", "error")
            return redirect(url_for("batch"))
        buffer = cache.pop(token)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"{token}.csv",
            mimetype="text/csv",
        )

    @app.route("/search", methods=["GET", "POST"])
    def search():
        dataset_key = _resolve_dataset_key(
            request.form.get("dataset_key") or request.args.get("dataset")
        )
        query = request.form.get("query", "").strip()
        results: Dict[str, List[Dict[str, object]]] | None = None
        models_available = list(available_models(dataset_key))
        raw_selection = request.form.getlist("models")
        selected_models = [name for name in raw_selection if name in models_available]
        if not selected_models:
            selected_models = list(default_models(dataset_key))

        if request.method == "POST" and query:
            try:
                bundle = load_dataset_bundle(dataset_key)
                model_map, _ = load_models(dataset_key, selected_models)
                if not model_map:
                    flash("Model belum tersedia untuk dataset ini.", "error")
                else:
                    predictions = run_predictions(model_map, bundle.features, bundle.metadata)
                    identifier_priority = DATASET_CONFIG[dataset_key]["identifier_priority"]  # type: ignore[index]
                    results = {}
                    
                    # Normalize query: remove spaces, hyphens, underscores for flexible matching
                    import re
                    
                    # Replace "KOI" with "K" for Kepler objects (e.g., "KOI 700" -> "K700")
                    # Replace "TOI" prefix for TESS objects (e.g., "TOI 700" -> "700")
                    normalized_query = query.lower().strip()
                    normalized_query = re.sub(r'^koi\s*', 'k', normalized_query)  # "KOI 700" -> "k700"
                    normalized_query = re.sub(r'^toi\s*', '', normalized_query)    # "TOI 700" -> "700"
                    # Remove spaces and hyphens but KEEP periods for decimal matching
                    normalized_query = re.sub(r'[\s\-_]', '', normalized_query)    # Remove spaces/hyphens but keep dots
                    
                    lower_query = query.lower()
                    
                    for model_name, df in predictions.items():
                        search_columns = [col for col in identifier_priority if col in df.columns]
                        
                        # First, try exact match (case-insensitive)
                        exact_mask = pd.Series(False, index=df.index)
                        for col in search_columns:
                            exact_mask |= df[col].astype(str).str.lower() == lower_query
                        
                        # If exact match found, use only the first exact match
                        if exact_mask.any():
                            filtered = df.loc[exact_mask].head(1)
                        else:
                            # Try normalized match (ignoring spaces, hyphens, etc.)
                            normalized_mask = pd.Series(False, index=df.index)
                            for col in search_columns:
                                # Normalize each value in the column for comparison (keep periods for decimal IDs)
                                normalized_values = df[col].astype(str).str.lower().str.replace(r'[\s\-_]', '', regex=True)
                                normalized_mask |= normalized_values == normalized_query
                                # Also try partial match for cases like "700" matching "700.01"
                                normalized_mask |= normalized_values.str.startswith(normalized_query + '.')
                            
                            if normalized_mask.any():
                                filtered = df.loc[normalized_mask].head(1)
                            else:
                                # If still no match, try partial match but only return the first result
                                partial_mask = pd.Series(False, index=df.index)
                                for col in search_columns:
                                    partial_mask |= df[col].astype(str).str.lower().str.contains(lower_query, na=False)
                                
                                if partial_mask.any():
                                    filtered = df.loc[partial_mask].head(1)
                                else:
                                    continue
                        
                        results[model_name] = format_summary_table(
                            filtered,
                            top_k=None,
                            identifier_priority=identifier_priority,
                        ).to_dict(orient="records")
                    
                    if not results:
                        flash("Target tidak ditemukan dalam prediksi terbaru.", "info")
            except Exception as exc:  # pragma: no cover - runtime guard
                flash(f"Pencarian gagal: {exc}", "error")

        return render_template(
            "search.html",
            dataset_key=dataset_key,
            dataset_label=DATASET_CONFIG[dataset_key]["label"],
            query=query,
            results=results,
            models=models_available,
            selected_models=selected_models,
        )

    @app.route("/ai", methods=["GET", "POST"])
    def ai():
        dataset_key = _resolve_dataset_key(
            request.form.get("dataset_key") or request.args.get("dataset")
        )
        top_k = int(request.form.get("top_k", DEFAULT_TOP_K))
        prompt = request.form.get("prompt", "")
        model_choice = request.form.get("llm_model", "models/gemini-2.5-flash")
        response_text = None
        context_preview = None
        error_message = None

        if request.method == "POST":
            try:
                bundle = load_dataset_bundle(dataset_key)
                selected_models = list(default_models(dataset_key))
                model_map, missing = load_models(dataset_key, selected_models)
                if missing:
                    flash(
                        "Beberapa model default belum tersedia. Pertimbangkan untuk melatih ulang terlebih dahulu.",
                        "warning",
                    )
                if not model_map:
                    flash("Tidak ada model yang tersedia untuk membuat konteks AI.", "error")
                else:
                    predictions = run_predictions(model_map, bundle.features, bundle.metadata)
                    identifier_priority = DATASET_CONFIG[dataset_key]["identifier_priority"]  # type: ignore[index]
                    context_preview = build_context_for_ai(
                        predictions,
                        top_k,
                        DATASET_CONFIG[dataset_key]["label"],
                        identifier_priority,
                    )
                    if not prompt:
                        flash("Masukkan pertanyaan terlebih dahulu sebelum menghubungi Gemini.", "info")
                    else:
                        api_key = configure_gemini()
                        if not api_key:
                            error_message = "GEMINI_API_KEY belum diatur di environment."
                        else:
                            response_text = call_gemini(model_choice, prompt, context_preview)
            except Exception as exc:  # pragma: no cover - runtime guard
                error_message = str(exc)

        return render_template(
            "ai.html",
            dataset_key=dataset_key,
            dataset_label=DATASET_CONFIG[dataset_key]["label"],
            top_k=top_k,
            prompt=prompt,
            llm_model=model_choice,
            response_text=response_text,
            context_preview=context_preview,
            error_message=error_message,
        )

    @app.route("/project", methods=["GET", "POST"])
    def project():
        dataset_key = _resolve_dataset_key(
            request.form.get("dataset_key") or request.args.get("dataset")
        )
        algorithm = request.form.get("algorithm", "Random Forest")
        metrics = None
        saved_filename = None
        missing_models: List[str] = []

        if request.method == "POST":
            algorithm_lower = algorithm.lower()
            params = {
                "test_size": request.form.get("test_size", "0.2"),
                "cv_splits": request.form.get("cv_splits", "5"),
                "random_state": request.form.get("random_state", "42"),
            }

            if algorithm_lower == "random forest":
                params.update(
                    {
                        "n_estimators": request.form.get("n_estimators", "500"),
                        "max_depth": request.form.get("max_depth", "auto"),
                        "min_samples_split": request.form.get("min_samples_split", "2"),
                        "min_samples_leaf": request.form.get("min_samples_leaf", "1"),
                    }
                )
            elif algorithm_lower == "gradient boosting":
                params.update(
                    {
                        "n_estimators": request.form.get("n_estimators", "500"),
                        "learning_rate": request.form.get("learning_rate", "0.05"),
                        "max_depth": request.form.get("max_depth", "3"),
                        "subsample": request.form.get("subsample", "1.0"),
                    }
                )
            elif algorithm_lower == "mlp":
                params.update(
                    {
                        "hidden_layer_sizes": request.form.get("hidden_layer_sizes", "256,128"),
                        "activation": request.form.get("activation", "relu"),
                        "alpha": request.form.get("alpha", "0.0001"),
                        "learning_rate_init": request.form.get("learning_rate_init", "0.001"),
                        "max_iter": request.form.get("max_iter", "400"),
                    }
                )
            else:
                flash("Algoritme tidak dikenal untuk pelatihan manual.", "error")
                return render_template(
                    "project.html",
                    dataset_key=dataset_key,
                    dataset_label=DATASET_CONFIG[dataset_key]["label"],
                    algorithm=algorithm,
                    metrics=metrics,
                    saved_filename=saved_filename,
                    missing_models=missing_models,
                    available_algorithms=["Random Forest", "Gradient Boosting", "MLP"],
                )
            extra_training_file = request.files.get("extra_training")
            if extra_training_file and extra_training_file.filename == "":
                extra_training_file = None

            try:
                pipeline, metrics, feature_names = train_custom_model(
                    dataset_key,
                    algorithm,
                    params,
                    extra_training_file,
                )

                model_files: Dict[str, str] = DATASET_CONFIG[dataset_key]["model_files"]  # type: ignore[index]
                keyword_map = {
                    "random forest": "random_forest",
                    "gradient boosting": "gbm",
                    "mlp": "mlp",
                }
                keyword = keyword_map.get(algorithm_lower)
                target_entry = next(
                    (
                        filename
                        for filename in model_files.values()
                        if keyword and keyword in filename
                    ),
                    None,
                )
                if not target_entry:
                    flash("Tidak bisa menemukan artefak model dasar untuk diganti.", "error")
                else:
                    save_trained_model(pipeline, metrics, target_entry, feature_names)
                    saved_filename = target_entry
                    flash("Model berhasil dilatih ulang dan disimpan.", "success")

                # Refresh metrics information for UI display
                _, missing = load_metrics_for_models(dataset_key, default_models(dataset_key))
                missing_models = [name for name, _ in missing]
            except ValueError as exc:
                flash(str(exc), "error")
            except Exception as exc:  # pragma: no cover - runtime guard
                flash(f"Pelatihan gagal: {exc}", "error")

        return render_template(
            "project.html",
            dataset_key=dataset_key,
            dataset_label=DATASET_CONFIG[dataset_key]["label"],
            algorithm=algorithm,
            metrics=metrics,
            saved_filename=saved_filename,
            missing_models=missing_models,
            available_algorithms=["Random Forest", "Gradient Boosting", "MLP"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)

