import joblib

ens = joblib.load('training/model_outputs_v9/q3_ensemble.joblib')
print('Ensemble structure:')
for k, v in ens.items():
    if k == 'models':
        print(f'  {k}:')
        for mk, mv in v.items():
            print(f'    {mk}: {type(mv).__name__} - keys: {list(mv.keys()) if hasattr(mv, "keys") else "N/A"}')
    else:
        print(f'  {k}: {v}')

# Try prediction
print('\nTesting prediction:')
sample_features = {f'feat_{i}': 0.5 for i in range(100)}
print(f'Sample features keys: {len(sample_features)}')

# Load one regular model for comparison
logreg_model = joblib.load('training/model_outputs_v9/q3_logreg.joblib')
print(f'LogReg vectorizer features: {logreg_model["vectorizer"].get_feature_names_out()}')
