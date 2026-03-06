import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import mlflow
import joblib
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.data_loader import load_splits, get_class_weight
from src.models.experiment_tracker import setup_mlflow

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def lgbm_objective(trial, X_train, y_train):
    """Optuna objective for LightGBM."""
    params = {
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'max_depth':        trial.suggest_int('max_depth', 3, 8),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves':       trial.suggest_int('num_leaves', 20, 100),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'min_child_samples':trial.suggest_int('min_child_samples', 5, 50),
        'scale_pos_weight': get_class_weight(y_train),
        'random_state':     42,
        'verbose':          -1,
    }
    
    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for tr_idx, val_idx in tscv.split(X_train):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train.iloc[tr_idx]
        X_v  = X_train.iloc[val_idx]
        y_v  = y_train.iloc[val_idx]
        
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr,
              eval_set=[(X_v, y_v)],
              callbacks=[lgb.early_stopping(10, verbose=False)])
        
        proba = m.predict_proba(X_v)[:, 1]
        scores.append(roc_auc_score(y_v, proba))
        
    return np.mean(scores)

def train_lgbm(ticker='AAPL', n_trials=50):
    """Full LightGBM training pipeline."""
    setup_mlflow()
    print(f'Training LightGBM for {ticker}...')
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(ticker)
    
    print(f'Running Optuna ({n_trials} trials)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: lgbm_objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_params = study.best_params
    best_cv_auc = study.best_value
    print(f'Best CV AUC: {best_cv_auc:.4f}')
    
    # Train final model
    best_params['scale_pos_weight'] = get_class_weight(y_train)
    best_params['random_state']      = 42
    best_params['verbose']           = -1 
    
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    val_proba = final_model.predict_proba(X_val)[:, 1]
    val_pred  = final_model.predict(X_val)
    val_auc   = roc_auc_score(y_val, val_proba)
    
    print(f'Validation AUC: {val_auc:.4f}')
    print(classification_report(y_val, val_pred))
    
    with mlflow.start_run(run_name=f'LightGBM_{ticker}'):
        mlflow.log_params(best_params)
        mlflow.log_metrics({'cv_auc': best_cv_auc, 'val_auc': val_auc})
        
    save_dir = Path('experiments/models')
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, save_dir / f'lgbm_{ticker}.pkl')
    print(f'Model saved!')
    
    return final_model, best_params, {'cv_auc': best_cv_auc, 'val_auc': val_auc}

if __name__ == '__main__':
    model, params, metrics = train_lgbm('AAPL', n_trials=50)
    print('LightGBM training complete!')