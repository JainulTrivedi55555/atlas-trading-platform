"""
Accuracy Comparison
Compares regime model AUC vs base model AUC on test set.
"""
import sys
sys.path.insert(0, '.')
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.models.data_loader import load_splits
from src.models.regime_detector import label_regimes
from src.utils.config import TICKERS

MODEL_DIR = Path('experiments/models')

print("Accuracy Comparison: Base vs Regime Models")
print("=" * 65)
print(f"{'Ticker':6} | {'Base AUC':9} | {'Regime AUC':10} | {'Improvement':11} | Regime")
print("-" * 65)

for ticker in TICKERS:
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_splits(ticker)
        X_test.index = pd.to_datetime(X_test.index).normalize()
        
        # Base model AUC
        base_path = MODEL_DIR / f'xgboost_{ticker}.pkl'
        base_model = joblib.load(base_path)
        
        try:
            feat_cols = base_model.get_booster().feature_names
        except:
            feat_cols = list(X_test.columns)
            
        for c in feat_cols:
            if c not in X_test.columns: 
                X_test[c] = 0
                
        base_prob = base_model.predict_proba(X_test[feat_cols])[:, 1]
        base_auc  = roc_auc_score(y_test, base_prob)
        
        # Regime model AUC (test the dominant regime)
        regimes = label_regimes(ticker)
        test_regimes = regimes.reindex(X_test.index).fillna('bull')
        dominant_regime = test_regimes.value_counts().index[0]
        
        regime_path  = MODEL_DIR / f'regime_xgb_{dominant_regime}_{ticker}.pkl'
        scaler_path  = MODEL_DIR / f'scaler_{dominant_regime}_{ticker}.pkl'
        
        if regime_path.exists() and scaler_path.exists():
            regime_model = joblib.load(regime_path)
            scaler       = joblib.load(scaler_path)
            mask         = test_regimes == dominant_regime
            X_t_regime   = X_test[mask].copy()
            
            try:
                fc = regime_model.get_booster().feature_names
            except:
                fc = list(X_t_regime.columns)
                
            for c in fc:
                if c not in X_t_regime.columns: 
                    X_t_regime[c] = 0
                    
            X_scaled   = scaler.transform(X_t_regime[fc])
            reg_prob   = regime_model.predict_proba(X_scaled)[:, 1]
            y_regime   = y_test[mask]
            
            regime_auc = roc_auc_score(y_regime, reg_prob) if len(set(y_regime)) > 1 else 0.5
            improvement = regime_auc - base_auc
            
            # Formatting improvement flag
            flag = "★" if improvement > 0.02 else ""
            
            print(f"{ticker:6} | {base_auc:.4f}    | {regime_auc:.4f}     | "
                  f"{improvement:+.4f} {flag:1}     | {dominant_regime}")
        else:
            print(f"{ticker:6} | {base_auc:.4f}    | N/A        | N/A         | "
                  f"Train first")
                  
    except Exception as e:
        print(f"{ticker:6} | ERROR: {e}")

print("=" * 65)
print("Run: python train_regime_models.py if regime column shows N/A")