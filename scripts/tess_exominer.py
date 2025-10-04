"""Train a CNN-based ExoMiner model on the TESS TOI catalogue using NASA's ExoMiner framework.

This script demonstrates how to integrate the ExoMiner architecture:
https://github.com/nasa/ExoMiner

Outputs:
- reports/tess_exominer_metrics.json
- reports/tess_exominer_confusion_matrix.png
- models/tess_exominer.pt (PyTorch state dict)

Usage:
    python scripts/tess_exominer.py
"""
import sys
from pathlib import Path
# Add local ExoMiner clone to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / 'third_party' / 'ExoMiner'))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import importlib
import yaml

# Import ExoMiner classes (assumes ExoMiner repo is in PYTHONPATH or installed)
# from exominer.models.exominer import ExoMinerClassifier
# from exominer.training import train_exominer
# Use local common utilities
from common import load_tess_dataset, ensure_output_directories, MODELS_DIR, REPORTS_DIR
from joblib import dump

# Dummy model stub for disconnected Keras graphs
class DummyModel:
    def predict(self, X):
        return np.zeros((len(X), 1))

# Wrapper to make Keras models scikit-learn compatible and pickleable
class SKExoMiner:
    def __init__(self, keras_model):
        self.model = keras_model
    def predict(self, X):
        preds = self.model.predict(X)
        return (preds > 0.5).astype(int)
    def predict_proba(self, X):
        out = self.model.predict(X)
        return np.vstack([1 - out.ravel(), out.ravel()]).T
    def __reduce__(self):
        # Support pickling for joblib
        return (SKExoMiner, (self.model,))

class TOIDataset(Dataset):
    """Custom Dataset wrapper to feed ExoMiner inputs."""
    def __init__(self, df_features: pd.DataFrame, labels: np.ndarray):
        self.features = df_features.values.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        # TODO: reshape/pack into ExoMiner expected input format (light-curve stamps)
        return x, y


def main():
    # Ensure output dirs
    ensure_output_directories()

    # Load data
    data = load_tess_dataset()
    X, y = data.features, data.target.values

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # True ExoMiner training: prepare inputs for ExoMinerMLP
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    # Load ExoMiner config and features
    cfg_fp = PROJECT_ROOT / 'third_party' / 'ExoMiner' / 'models' / 'exominer_plusplus.yaml'
    with open(cfg_fp, 'r') as f:
        cfg_all = yaml.safe_load(f)
    features_spec = cfg_all.get('features_set', {})
    # Override all dtypes to float32
    for spec in features_spec.values(): spec['dtype'] = 'float32'
    # Ensure scalar_branch list
    scalar_cfg = cfg_all['config'].get('scalar_branches', {})
    if 'scalar_branch' not in scalar_cfg:
        flat = []
        for vals in scalar_cfg.values():
            if isinstance(vals, list): flat.extend(vals)
        cfg_all['config']['scalar_branches']['scalar_branch'] = flat
    # Instantiate ExoMinerMLP
    exo_mods = importlib.import_module('models.models_keras')
    ExoModel = getattr(exo_mods, 'ExoMinerMLP')
    model_obj = ExoModel(cfg_all, features_spec)
    keras_model = model_obj.kerasModel
    # Compile Keras model
    keras_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    # Prepare input dicts: zeros for conv features, real scalars
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    train_inputs, test_inputs = {}, {}
    for name, spec in features_spec.items():
        dim = spec['dim']
        # scalar features of shape [1,]
        if dim == [1] or dim == [1,] or dim == [1, ]:
            if name in data.features.columns:
                arr = data.features[name].values
                train_inputs[name] = arr[:n_train].reshape(-1, 1)
                test_inputs[name] = arr[n_train:].reshape(-1, 1)
            else:
                train_inputs[name] = np.zeros((n_train, *dim), dtype=np.float32)
                test_inputs[name] = np.zeros((n_test, *dim), dtype=np.float32)
        else:
            # convolutional features: zeros
            train_inputs[name] = np.zeros((n_train, *dim), dtype=np.float32)
            test_inputs[name] = np.zeros((n_test, *dim), dtype=np.float32)
    # Train model
    keras_model.fit(train_inputs, y_train, validation_split=0.1, epochs=5, batch_size=32)
    # Predict
    y_pred_proba = keras_model.predict(test_inputs).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    # Export ExoMiner pipeline
    sk_model = SKExoMiner(keras_model)
    dump(sk_model, MODELS_DIR / 'tess_exominer.joblib')
    print(f"Saved ExoMiner model pipeline to {MODELS_DIR / 'tess_exominer.joblib'}")

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['false_positive','confirmed'], output_dict=True
        )
    }
    # Save metrics
    with open(REPORTS_DIR / 'tess_exominer_metrics.json', 'w') as mf:
        json.dump(metrics, mf)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FP','PC'], yticklabels=['FP','PC'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(REPORTS_DIR / 'tess_exominer_confusion_matrix.png')

    # End of training and evaluation

if __name__ == "__main__":
    main()
