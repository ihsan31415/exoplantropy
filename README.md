# exoplantropy

Machine-learning pipelines and a Flask dashboard for ranking exoplanet
candidate observations from NASA's TESS, Kepler, and K2 catalogues.

## Highlights

- Shared preprocessing utilities with balanced training splits for all three
  missions.
- Unified training entry points for Gradient Boosting, Random Forest, XGBoost,
  LightGBM, CatBoost, and an MLP classifier. Each script exports a pipeline,
  metrics JSON, and feature importance/inspection artefacts.
- Flask web experience (`app.py`) for manual predictions, batch scoring,
  AI-assisted insights, and custom retraining.

## Getting started

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure the Gemini API key

1. Copy `config/gemini.env.example` to `config/gemini.env`.
2. Replace the placeholder value for `GEMINI_API_KEY`.

The Flask app automatically reads this file (via `python-dotenv`). You can still
override the key via regular environment variables if required.

### Optional environment overrides

Create a local `.env` (based on `.env.example`) to provide additional Flask or
debug settings. This file is no longer used for the Gemini key.

## Training scripts

Run any of the training modules to refresh models and reports:

```powershell
.venv\Scripts\python.exe scripts/train_random_forest.py
.venv\Scripts\python.exe scripts/train_gradient_boosting.py
.venv\Scripts\python.exe scripts/train_xgboost.py
.venv\Scripts\python.exe scripts/train_lightgbm.py
.venv\Scripts\python.exe scripts/train_catboost.py
.venv\Scripts\python.exe scripts/train_mlp.py
```

Each trainer accepts optional CLI arguments for dataset selection and
hyper-parameters. Generated artefacts are stored in `models/` and `reports/`.

## Flask web interface

```powershell
.venv\Scripts\python.exe app.py
```

Key routes:

- `/manual` – score custom feature inputs and preview a transit simulation.
- `/batch` – upload CSVs or use the bundled catalogues for bulk predictions
  (set Top-K to `0` to list all rows).
- `/ai` – send summarised model outputs to Gemini for a narrative explanation.
- `/project` – retrain Random Forest, Gradient Boosting, or MLP pipelines with
  custom hyper-parameters and optional extra labelled data.

## Repository layout

```
config/                  # Local (ignored) Gemini credentials
models/, reports/        # Generated artefacts from training routines
scripts/                 # Dataset loaders and trainer entry points
services/                # Shared business logic used by Flask & CLIs
templates/, static/      # Flask UI
```

## Notes

- Ensure you have run the desired training scripts before starting the Flask
  app so the expected artefacts exist.
- Keep secrets out of version control—`config/gemini.env` is ignored by
  default.
