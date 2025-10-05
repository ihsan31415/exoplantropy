import pandas as pd
from scripts.common import load_tess_dataset
from services.exoplanet_service import run_predictions, available_models, load_models

print('Loading dataset...')
ds = load_tess_dataset()
med = ds.features.median()
manual_df = pd.DataFrame([med], columns=ds.features.columns)
model_names = list(available_models('tess'))
model_map, missing = load_models('tess', model_names)

for name, model in model_map.items():
    print(f'\nTesting model: {name}')
    try:
        result = run_predictions({name: model}, manual_df)
        print('OK')
    except Exception as e:
        print('ERROR:', repr(e))
