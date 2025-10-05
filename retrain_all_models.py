"""
Retrain all models to fix the pickle/joblib compatibility issue.
This script will train all models using the current Python environment.
"""
import subprocess
import sys
from pathlib import Path

# Training scripts in order of training time (fastest first)
TRAINING_SCRIPTS = [
    "scripts/train_gradient_boosting.py",
    "scripts/train_lightgbm.py",
    "scripts/train_xgboost.py",
    "scripts/train_random_forest.py",
    "scripts/train_catboost.py",
    "scripts/train_mlp.py",
]

def main():
    print("="*70)
    print("RETRAINING ALL MODELS")
    print("="*70)
    print("\nThis will fix the KeyError: 72 issue by retraining all models")
    print("with your current Python version.\n")
    
    project_root = Path(__file__).parent
    
    for i, script_path in enumerate(TRAINING_SCRIPTS, 1):
        full_path = project_root / script_path
        model_name = script_path.split("/")[-1].replace("train_", "").replace(".py", "")
        
        print(f"\n[{i}/{len(TRAINING_SCRIPTS)}] Training {model_name.upper()}...")
        print("-" * 70)
        
        if not full_path.exists():
            print(f"[WARNING] Script not found: {full_path}")
            continue
        
        try:
            result = subprocess.run(
                [sys.executable, str(full_path)],
                capture_output=True,
                text=True,
                cwd=str(project_root)
            )
            
            if result.returncode == 0:
                print(f"[OK] {model_name.upper()} trained successfully!")
                if result.stdout:
                    # Print last few lines of output
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:
                        print(f"  {line}")
            else:
                print(f"[ERROR] {model_name.upper()} training failed!")
                print(f"Error: {result.stderr}")
        
        except Exception as e:
            print(f"[ERROR] Error running {model_name}: {e}")
    
    print("\n" + "="*70)
    print("RETRAINING COMPLETE!")
    print("="*70)
    print("\nAll models have been retrained with your current Python environment.")
    print("The KeyError: 72 issue should now be resolved.")
    print("\nPlease restart your Flask application and try again.")

if __name__ == "__main__":
    main()
