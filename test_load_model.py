import sys
import joblib
import sklearn
from pathlib import Path
import traceback

print('python', sys.version.split()[0])
print('joblib', joblib.__version__)
print('sklearn', sklearn.__version__)

p = Path('exoplantropy-main/models/lightgbm_model.joblib')
print('path', p)
print('exists', p.exists())
if p.exists():
    try:
        print('size', p.stat().st_size)
    except Exception as e:
        print('stat error', e)

try:
    print('\nTrying to load model...')
    m = joblib.load(p)
    print('Loaded model type:', type(m))
except Exception as e:
    print('Load failed:')
    traceback.print_exc()
