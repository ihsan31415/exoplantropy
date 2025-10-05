import traceback
import json
from pathlib import Path
import pandas as pd

try:
    from services import exoplanet_service as svc
except Exception as e:
    print('ERROR importing exoplanet_service:', e)
    raise

print('STANDARD_FEATURE_ORDER:', svc.STANDARD_FEATURE_ORDER)

dataset='tess'
selected = list(svc.default_models(dataset))
print('Default models:', selected)
models, missing = svc.load_models(dataset, selected)
print('Loaded models:', list(models.keys()))
print('Missing models:', missing)

features = {f: 1.0 for f in svc.STANDARD_FEATURE_ORDER}
features_df = pd.DataFrame([features])
print('Features DF columns:', features_df.columns.tolist())
print('Features DF shape:', features_df.shape)

try:
    results = svc.run_predictions(models, features_df)
    print('run_predictions succeeded. Keys:', list(results.keys()))
    for k,v in results.items():
        print(k, v.head().to_dict())
except Exception as e:
    print('run_predictions raised an exception:')
    traceback.print_exc()
    msg = str(e)
    try:
        start = msg.find('Debug info:')
        if start!=-1:
            print('\nFound debug info:')
            print(msg[start:])
    except Exception:
        pass
