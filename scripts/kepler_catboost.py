"""Compatibility wrapper for the unified CatBoost trainer."""

from train_catboost import main as train_main


if __name__ == "__main__":
    train_main(["kepler"])