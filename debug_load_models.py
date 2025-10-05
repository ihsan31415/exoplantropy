import joblib
import os
from pathlib import Path

# Define the directory where the models are stored
MODELS_DIR = Path("exoplantropy-main/models")

def inspect_sklearn_model(model_path):
    """Loads a scikit-learn model and inspects its properties."""
    print(f"--- Inspecting {model_path.name} ---")
    if not model_path.exists():
        print("File does not exist.")
        print("-" * (len(model_path.name) + 16))
        print()
        return

    try:
        pipeline = joblib.load(model_path)
        print(f"Successfully loaded: {model_path.name}")

        # Attempt to find feature names from the pipeline or model
        feature_names = None
        if hasattr(pipeline, 'feature_names_in_'):
            feature_names = pipeline.feature_names_in_
        elif hasattr(pipeline, 'steps'):
            # Search for feature names within the steps of a pipeline
            for step_name, step_obj in pipeline.steps:
                if hasattr(step_obj, 'feature_names_in_'):
                    feature_names = step_obj.feature_names_in_
                    print(f"Found feature names in pipeline step: '{step_name}'")
                    break
        
        if feature_names is not None:
            print(f"Model expects {len(feature_names)} features:")
            print(feature_names.tolist())
        else:
            print("Could not automatically determine expected feature names from this model artifact.")

    except Exception as e:
        print(f"Error loading or inspecting model: {e}")
    finally:
        print("-" * (len(model_path.name) + 16))
        print()


def main():
    """Main function to inspect all models in the specified directory."""
    if not MODELS_DIR.is_dir():
        print(f"Error: Models directory not found at '{MODELS_DIR.resolve()}'")
        return

    print(f"Inspecting models in: {MODELS_DIR.resolve()}")
    
    # List of scikit-learn style models to check
    joblib_models = [
        "random_forest_model.joblib",
        "gradient_boosting_model.joblib",
        "lightgbm_model.joblib",
        "xgboost_model.joblib",
        "catboost_model.joblib",
        "logreg_model.joblib",
    ]

    for model_file in joblib_models:
        inspect_sklearn_model(MODELS_DIR / model_file)
    
    # Special handling for the Keras/TensorFlow model
    h5_model_path = MODELS_DIR / "keras_mlp_model.h5"
    print(f"--- Inspecting {h5_model_path.name} ---")
    if h5_model_path.exists():
        try:
            # Suppress TensorFlow logging for a cleaner output
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            model = tf.keras.models.load_model(h5_model_path, compile=False)
            print(f"Successfully loaded Keras model: {h5_model_path.name}")
            if hasattr(model, 'input_shape'):
                 print(f"Keras model input shape: {model.input_shape}")
            print("Keras model summary:")
            model.summary()
        except ImportError:
            print("TensorFlow/Keras is not installed. Cannot inspect .h5 file.")
        except Exception as e:
            print(f"Error loading Keras model: {e}")
    else:
        print("Keras model file does not exist.")
    print("-" * (len(h5_model_path.name) + 16))


if __name__ == "__main__":
    main()

