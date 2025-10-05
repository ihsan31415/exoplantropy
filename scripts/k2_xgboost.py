"""Compatibility wrapper for the unified XGBoost trainer."""

from train_xgboost import main as train_main


if __name__ == "__main__":
    train_main(["k2"])
