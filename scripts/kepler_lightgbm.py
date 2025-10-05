"""Compatibility wrapper for the unified LightGBM trainer."""

from train_lightgbm import main as train_main


if __name__ == "__main__":
    train_main(["kepler"])
