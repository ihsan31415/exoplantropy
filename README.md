# exoplantropy

Toolkit and experiments for detecting exoplanet candidates from space telescope
catalogues (TESS, Kepler, K2).

## Random Forest model for the TESS TOI catalogue

The script `scripts/tess_random_forest.py` trains a binary classifier that
predicts whether a TESS Object of Interest (TOI) is a likely exoplanet
candidate (`tfopwg_disp == "PC"`). It uses a Random Forest pipeline with
median imputation, class-weight balancing, and cross-validation diagnostics.

### Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # or install pandas, scikit-learn, seaborn
```

### Run the model

```powershell
.venv\Scripts\python.exe scripts/tess_random_forest.py
```

### Outputs

- `models/tess_random_forest.joblib` – fitted pipeline for reuse.
- `reports/tess_random_forest_metrics.json` – accuracy, precision, recall, F1,
	ROC AUC, and cross-validation scores.
- `reports/tess_random_forest_confusion_matrix.png` – confusion matrix heatmap.
- `reports/tess_random_forest_top_features.csv` – top 20 features by
	importance.

## Next steps

- Extend coverage to the remaining ensemble methods (AdaBoost, stacking)
	following the shared preprocessing pattern.
- Build shared utilities for consistent data cleaning and evaluation across
	datasets (Kepler, K2).

## Support Vector Machine (SVM) model for the TESS TOI catalogue

The script `scripts/tess_svm.py` follows the same data preparation steps but
trains an RBF-kernel SVM with class balancing and probability calibration.

```powershell
.venv\Scripts\python.exe scripts\tess_svm.py
```

Outputs are stored alongside the Random Forest results with SVM-specific file
names (e.g., `tess_svm_metrics.json`, `tess_svm_confusion_matrix.png`,
`tess_svm.joblib`).

## K-Nearest Neighbors (KNN) model for the TESS TOI catalogue

`scripts/tess_knn.py` trains a distance-weighted KNN classifier with standard
scaling and cross-validation diagnostics.

```powershell
.venv\Scripts\python.exe scripts\tess_knn.py
```

Outputs mirror the previous models (`tess_knn_metrics.json`,
`tess_knn_confusion_matrix.png`, `tess_knn.joblib`).

## Logistic Regression model for the TESS TOI catalogue

`scripts/tess_logistic_regression.py` builds an L2-regularised logistic
classifier with class balancing and probability calibration.

```powershell
.venv\Scripts\python.exe scripts\tess_logistic_regression.py
```

Alongside the usual metrics/plot/model artefacts, it also exports
`tess_logistic_regression_top_coefficients.csv` to highlight the most
influential features.

## Decision Tree model for the TESS TOI catalogue

`scripts/tess_decision_tree.py` fits a class-balanced decision tree and exports
feature importances for interpretability.

```powershell
.venv\Scripts\python.exe scripts\tess_decision_tree.py
```

Artefacts include `tess_decision_tree_metrics.json`,
`tess_decision_tree_confusion_matrix.png`,
`tess_decision_tree_top_features.csv`, and the serialized model
`tess_decision_tree.joblib`.

## Streamlit dashboard & Gemini integration

Launch the interactive explorer defined in `app/streamlit_app.py` to compare
model outputs and forward summaries to Gemini:

```powershell
streamlit run app/streamlit_app.py
```

Configure Gemini securely via environment variable before starting the app:

```powershell
$env:GEMINI_API_KEY = "<your-key-here>"
```

Alternatively, you can store the key in `.streamlit/secrets.toml` under the
`GEMINI_API_KEY` entry. Never commit API keys to version control. The app loads
pre-trained models from the `models/` directory; run each training script once
to generate them if they are missing.
