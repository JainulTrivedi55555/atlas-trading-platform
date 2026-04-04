import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.utils.config import TICKERS
from src.models.data_loader import load_splits

MODEL_DIR  = Path('experiments/models')
EXPERIMENT = 'ATLAS-Production-Registry'

def register_all_models():
    """Register all 20 models in MLflow registry."""
    mlflow.set_tracking_uri('mlruns')
    mlflow.set_experiment(EXPERIMENT)
    results = []
    
    for ticker in TICKERS:
        for model_type in ['xgboost', 'lgbm']:
            model_path = MODEL_DIR / f'{model_type}_{ticker}.pkl'
            if not model_path.exists():
                print(f'  Skip {ticker} {model_type} — not found')
                continue
                
            try:
                model = joblib.load(model_path)
                X_train, y_train, X_val, y_val, X_test, y_test = (
                    load_splits(ticker)
                )
                
                val_auc  = roc_auc_score(
                    y_val,
                    model.predict_proba(X_val)[:, 1]
                )
                test_auc = roc_auc_score(
                    y_test,
                    model.predict_proba(X_test)[:, 1]
                )
                
                with mlflow.start_run(
                    run_name=f'{model_type}_{ticker}_prod'
                ):
                    mlflow.log_param('ticker',     ticker)
                    mlflow.log_param('model_type', model_type)
                    mlflow.log_param('n_features', 43)
                    mlflow.log_metric('val_auc',   round(val_auc, 4))
                    mlflow.log_metric('test_auc',  round(test_auc, 4))
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path=f'{model_type}_{ticker}',
                        registered_model_name=(
                            f'atlas_{model_type}_{ticker.lower()}'
                        )
                    )
                    
                print(f'  Registered {ticker} {model_type}: '
                      f'val={val_auc:.4f} test={test_auc:.4f}')
                
                results.append({
                    'ticker':     ticker,
                    'model':      model_type,
                    'val_auc':    round(val_auc, 4),
                    'test_auc':   round(test_auc, 4),
                    'registered': True,
                })
            except Exception as e:
                print(f'  ERROR {ticker} {model_type}: {e}')
                
    return pd.DataFrame(results)

if __name__ == '__main__':
    print('Registering all ATLAS models in MLflow...')
    df = register_all_models()
    print('\n' + '='*50)
    print('REGISTRY COMPLETE')
    print(df.to_string())
    print('\nView at: mlflow ui -> http://localhost:5000')