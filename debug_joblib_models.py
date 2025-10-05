import joblib
import pickle
import sys
from pathlib import Path

# Define the directory where the models are stored
MODELS_DIR = Path("exoplantropy-main/models")

def debug_model_file(model_path):
    """Debug a model file with more detailed error information."""
    print(f"--- Debugging {model_path.name} ---")
    if not model_path.exists():
        print("File does not exist.")
        print("-" * (len(model_path.name) + 16))
        print()
        return

    try:
        # Try to get basic file info
        file_size = model_path.stat().st_size
        print(f"File size: {file_size} bytes")
        
        # Try to read the file with pickle directly to see if it's a pickle format issue
        print("Attempting to read with pickle...")
        with open(model_path, 'rb') as f:
            # Try to peek at the first few bytes
            f.seek(0)
            header = f.read(10)
            print(f"File header (first 10 bytes): {header}")
            
            # Reset file position
            f.seek(0)
            
            # Try to load with pickle
            try:
                data = pickle.load(f)
                print("Successfully loaded with pickle!")
                print(f"Data type: {type(data)}")
            except Exception as pickle_error:
                print(f"Pickle error: {pickle_error}")
                print(f"Pickle error type: {type(pickle_error)}")
        
        # Try with joblib
        print("\nAttempting to load with joblib...")
        try:
            pipeline = joblib.load(model_path)
            print("Successfully loaded with joblib!")
            print(f"Pipeline type: {type(pipeline)}")
        except Exception as joblib_error:
            print(f"Joblib error: {joblib_error}")
            print(f"Joblib error type: {type(joblib_error)}")
            
            # Get more detailed error info
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()

    except Exception as e:
        print(f"General error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("-" * (len(model_path.name) + 16))
        print()

def main():
    """Main function to debug all models in the specified directory."""
    if not MODELS_DIR.is_dir():
        print(f"Error: Models directory not found at '{MODELS_DIR.resolve()}'")
        return

    print(f"Debugging models in: {MODELS_DIR.resolve()}")
    print(f"Python version: {sys.version}")
    print(f"Joblib version: {joblib.__version__}")
    print(f"Pickle protocol version: {pickle.HIGHEST_PROTOCOL}")
    print()
    
    # List of scikit-learn style models to check
    joblib_models = [
        "catboost_model.joblib",
        "random_forest_model.joblib",
    ]

    for model_file in joblib_models:
        debug_model_file(MODELS_DIR / model_file)

if __name__ == "__main__":
    main()