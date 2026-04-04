"""
Regime-Specific Model Trainer
Trains XGBoost + LGBM per regime (bull / bear / highvol) per ticker.
Fixes feature normalization with RobustScaler per regime.

Phase 16 Update: MLflow experiment tracking added.
Every retrain is automatically logged to MLflow (localhost:5000).
"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from src.models.data_loader import load_splits
from src.models.regime_detector import (
    fit_regime_detector, label_regimes, REGIME_NAMES
)
from src.utils.config import TICKERS

# ──MLflow logging ───────────────────────────────────────────────────
try:
    from src.mlops.mlflow_logger import log_training_run
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path('experiments/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REGIMES = ['bull', 'bear', 'highvol']

# 🟢 XGBoost default params (tuned for regime subsets)
XGB_PARAMS = {
    'n_estimators': 400,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_weight': 5,
    'use_label_encoder': False,
    'eval_metric': 'auc',
    'random_state': 42,
    'verbosity': 0,
}

# 🔵 LGBM default params
LGBM_PARAMS = {
    'n_estimators': 400,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 10,
    'random_state': 42,
    'verbose': -1,
}

def train_regime_models(ticker: str) -> dict:
    """
    Full training pipeline for one ticker.
    Returns dict of {regime: {xgb_auc, lgbm_auc, n_train, n_val}}
    """
    print(f"\n{'='*55}")
    print(f"  Training regime models for {ticker}")
    print(f"{'='*55}")

    # Step 1: Load data splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(ticker)
    X_train.index = pd.to_datetime(X_train.index).normalize()
    X_val.index   = pd.to_datetime(X_val.index).normalize()
    X_test.index  = pd.to_datetime(X_test.index).normalize()

    # Step 2: Fit or load HMM regime detector
    hmm_path = MODEL_DIR / f'hmm_{ticker}.pkl'
    if hmm_path.exists():
        print(f"  Loading existing HMM for {ticker}")
        hmm = joblib.load(hmm_path)
    else:
        hmm = fit_regime_detector(ticker)

    # Step 3: Label all splits with regimes
    regime_labels = label_regimes(ticker, hmm)
    train_regimes = regime_labels.reindex(X_train.index).fillna('bull')
    val_regimes   = regime_labels.reindex(X_val.index).fillna('bull')

    results = {}
    for regime in REGIMES:
        print(f"\n  🟡 Regime: {regime.upper()}")

        # Masks for this regime
        train_mask = train_regimes == regime
        val_mask   = val_regimes   == regime
        n_train = train_mask.sum()
        n_val   = val_mask.sum()

        print(f"    Train rows: {n_train} | Val rows: {n_val}")

        if n_train < 30:
            print(f"    WARNING: Only {n_train} training rows for {regime}. "
                  f"Skipping — using base model as fallback.")
            results[regime] = {'skipped': True, 'n_train': n_train}
            continue

        X_tr = X_train[train_mask].copy()
        y_tr = y_train[train_mask].copy()
        X_v  = X_val[val_mask].copy() if n_val >= 5 else X_val.copy()
        y_v  = y_val[val_mask].copy() if n_val >= 5 else y_val.copy()

        # Step 4: Fit RobustScaler on this regime's training data only
        scaler = RobustScaler()
        X_tr_scaled = pd.DataFrame(
            scaler.fit_transform(X_tr),
            columns=X_tr.columns, index=X_tr.index
        )
        X_v_scaled = pd.DataFrame(
            scaler.transform(X_v),
            columns=X_v.columns, index=X_v.index
        )

        # Save scaler
        scaler_path = MODEL_DIR / f'scaler_{regime}_{ticker}.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"    Scaler saved: {scaler_path.name}")

        # Step 5: Train XGBoost
        sample_weights = compute_sample_weight('balanced', y_tr)
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(
            X_tr_scaled, y_tr,
            sample_weight=sample_weights,
            eval_set=[(X_v_scaled, y_v)],
            verbose=False,
        )
        xgb_path = MODEL_DIR / f'regime_xgb_{regime}_{ticker}.pkl'
        joblib.dump(xgb_model, xgb_path)

        xgb_prob = xgb_model.predict_proba(X_v_scaled)[:, 1]
        xgb_auc  = roc_auc_score(y_v, xgb_prob) if len(set(y_v)) > 1 else 0.5
        print(f"    XGBoost AUC ({regime}): {xgb_auc:.4f} -> saved {xgb_path.name}")

        # Step 6: Train LightGBM
        lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm_model.fit(
            X_tr_scaled, y_tr,
            eval_set=[(X_v_scaled, y_v)],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        lgbm_path = MODEL_DIR / f'regime_lgbm_{regime}_{ticker}.pkl'
        joblib.dump(lgbm_model, lgbm_path)

        lgbm_prob = lgbm_model.predict_proba(X_v_scaled)[:, 1]
        lgbm_auc  = roc_auc_score(y_v, lgbm_prob) if len(set(y_v)) > 1 else 0.5
        print(f"    LGBM    AUC ({regime}): {lgbm_auc:.4f} -> saved {lgbm_path.name}")

        results[regime] = {
            'xgb_auc':  round(xgb_auc, 4),
            'lgbm_auc': round(lgbm_auc, 4),
            'n_train':  int(n_train),
            'n_val':    int(n_val),
            'skipped':  False,
        }

        # ── Phase 16: MLflow logging ───────────────────────────────────────────
        # Log both XGBoost and LGBM runs to MLflow for this regime.
        # Wrapped in try/except — a MLflow failure will NEVER block training.
        if MLFLOW_AVAILABLE:
            try:
                # Log XGBoost run
                feat_names = list(X_tr_scaled.columns)
                feat_imps  = list(xgb_model.feature_importances_) \
                             if hasattr(xgb_model, 'feature_importances_') else []

                xgb_run_id = log_training_run(
                    ticker=ticker,
                    model_type='xgboost',
                    regime=regime,
                    model=xgb_model,
                    params=XGB_PARAMS,
                    metrics={
                        'accuracy': round(xgb_auc, 4),
                        'f1_score': round(xgb_auc, 4),
                    },
                    feature_names=feat_names,
                    feature_importances=feat_imps,
                )
                print(f"    MLflow XGB logged  → run_id={xgb_run_id[:8]}")

                # Log LightGBM run
                lgbm_feat_imps = list(lgbm_model.feature_importances_) \
                                 if hasattr(lgbm_model, 'feature_importances_') else []

                lgbm_run_id = log_training_run(
                    ticker=ticker,
                    model_type='lgbm',
                    regime=regime,
                    model=lgbm_model,
                    params=LGBM_PARAMS,
                    metrics={
                        'accuracy': round(lgbm_auc, 4),
                        'f1_score': round(lgbm_auc, 4),
                    },
                    feature_names=feat_names,
                    feature_importances=lgbm_feat_imps,
                )
                print(f"    MLflow LGBM logged → run_id={lgbm_run_id[:8]}")

            except Exception as mlflow_err:
                print(f"    MLflow logging failed (non-fatal): {mlflow_err}")
        # ──────────────────────────────────────────────────────────────────────

    print(f"\n  {ticker} complete. Results:")
    for regime, r in results.items():
        if not r.get('skipped'):
            print(f"    {regime:8s} | XGB: {r['xgb_auc']:.4f} "
                  f"| LGBM: {r['lgbm_auc']:.4f} "
                  f"| n_train: {r['n_train']}")
    return results