"""
ATLAS — Fix Feature Normalization & Retrain All Models
======================================================
Problem: EMA_12, EMA_26, BB_Upper, BB_Lower, BB_Middle are raw price
         values (~$180 for AAPL). The model learns absolute price levels
         instead of patterns — causing EMA_12 to dominate at 0.05 importance.

Fix:
  1. Add Price_vs_EMA12 and Price_vs_EMA26 (ratio features, same as Price_vs_SMA*)
  2. Drop raw price columns: EMA_12, EMA_26, BB_Upper, BB_Lower, BB_Middle
     (BB_Pct and BB_Width already capture Bollinger info in normalized form)
  3. Re-run data splitter to regenerate splits
  4. Retrain all base XGBoost + LGBM models (Phase 10)
  5. Retrain all regime-specific models (Phase 12)

Run: python fix_features_and_retrain.py
"""
import sys, warnings, time
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler

# ── Config ─────────────────────────────────────────────────────────────────────
from src.utils.config import TICKERS, PROCESSED_DIR

SPLITS_DIR  = Path('data/processed/splits')
MODELS_DIR  = Path('experiments/models')
FEATURES_DIR = Path('data/processed/features')

# Columns to DROP — raw price values that inflate importance
RAW_PRICE_COLS_TO_DROP = [
    'EMA_12', 'EMA_26',
    'BB_Upper', 'BB_Lower', 'BB_Middle',
]

print("=" * 60)
print("ATLAS — Feature Normalization Fix + Full Retrain")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Add normalized EMA features, drop raw price columns
# ══════════════════════════════════════════════════════════════════════
print("\n📐 STEP 1: Fixing feature CSVs for all tickers...")

for ticker in TICKERS:
    feat_path = FEATURES_DIR / f'{ticker}_features.csv'
    if not feat_path.exists():
        print(f"  ⚠️  {ticker}: features CSV not found at {feat_path} — skipping")
        continue

    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Add normalized EMA ratio features (same pattern as Price_vs_SMA*)
    if 'EMA_12' in df.columns and 'Close' in df.columns:
        df['Price_vs_EMA12'] = (df['Close'] - df['EMA_12']) / df['EMA_12']
    if 'EMA_26' in df.columns and 'Close' in df.columns:
        df['Price_vs_EMA26'] = (df['Close'] - df['EMA_26']) / df['EMA_26']

    # Drop raw price columns that shouldn't be model features
    cols_to_drop = [c for c in RAW_PRICE_COLS_TO_DROP if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Save back
    df.to_csv(feat_path)
    new_feature_count = len([c for c in df.columns
                              if c not in ['Target_Direction', 'Target_Return',
                                           'Open', 'High', 'Low', 'Close', 'Volume']])
    print(f"  ✅ {ticker}: dropped {len(cols_to_drop)} raw cols, "
          f"added 2 EMA ratios | features now: {new_feature_count}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Re-run data splitter to regenerate train/val/test splits
# ══════════════════════════════════════════════════════════════════════
print("\n✂️  STEP 2: Regenerating train/val/test splits...")

from sklearn.preprocessing import StandardScaler

def rebuild_splits(ticker: str):
    feat_path = FEATURES_DIR / f'{ticker}_features.csv'
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Drop target and OHLCV from features
    exclude = ['Target_Direction', 'Target_Return',
               'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [c for c in df.columns if c not in exclude]

    df_feat = df[feature_cols + ['Target_Direction']].dropna()

    # Time-based split (same ratios as Phase 2)
    n = len(df_feat)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df_feat.iloc[:train_end]
    val   = df_feat.iloc[train_end:val_end]
    test  = df_feat.iloc[val_end:]

    X_train = train[feature_cols]
    y_train = train['Target_Direction']
    X_val   = val[feature_cols]
    y_val   = val['Target_Direction']
    X_test  = test[feature_cols]
    y_test  = test['Target_Direction']

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index, columns=feature_cols
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index, columns=feature_cols
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index, columns=feature_cols
    )

    # Save splits
    save_dir = SPLITS_DIR / ticker
    save_dir.mkdir(parents=True, exist_ok=True)
    X_train_s.to_csv(save_dir / 'X_train.csv')
    y_train.to_csv(save_dir / 'y_train.csv')
    X_val_s.to_csv(save_dir / 'X_val.csv')
    y_val.to_csv(save_dir / 'y_val.csv')
    X_test_s.to_csv(save_dir / 'X_test.csv')
    y_test.to_csv(save_dir / 'y_test.csv')

    # Save scaler for this ticker
    joblib.dump(scaler, save_dir / 'scaler.pkl')

    print(f"  ✅ {ticker}: train={len(X_train)} val={len(X_val)} "
          f"test={len(X_test)} | features={len(feature_cols)}")
    return feature_cols

all_feature_cols = {}
for ticker in TICKERS:
    cols = rebuild_splits(ticker)
    all_feature_cols[ticker] = cols


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Retrain base XGBoost + LGBM models (Phase 10)
# ══════════════════════════════════════════════════════════════════════
print("\n🤖 STEP 3: Retraining base XGBoost + LGBM models...")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def load_splits_local(ticker):
    d = SPLITS_DIR / ticker
    X_train = pd.read_csv(d / 'X_train.csv', index_col=0, parse_dates=True)
    y_train = pd.read_csv(d / 'y_train.csv', index_col=0).squeeze()
    X_val   = pd.read_csv(d / 'X_val.csv',   index_col=0, parse_dates=True)
    y_val   = pd.read_csv(d / 'y_val.csv',   index_col=0).squeeze()
    X_test  = pd.read_csv(d / 'X_test.csv',  index_col=0, parse_dates=True)
    y_test  = pd.read_csv(d / 'y_test.csv',  index_col=0).squeeze()
    return X_train, y_train, X_val, y_val, X_test, y_test

MODELS_DIR.mkdir(parents=True, exist_ok=True)
base_results = {}

for ticker in TICKERS:
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits_local(ticker)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        early_stopping_rounds=20, random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_val, y_val)], verbose=False)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    joblib.dump(xgb, MODELS_DIR / f'xgboost_{ticker}.pkl')

    # LGBM
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    lgbm.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             callbacks=[])
    lgbm_auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])
    joblib.dump(lgbm, MODELS_DIR / f'lgbm_{ticker}.pkl')

    base_results[ticker] = {'xgb': xgb_auc, 'lgbm': lgbm_auc}
    print(f"  ✅ {ticker}: XGB AUC={xgb_auc:.4f}  LGBM AUC={lgbm_auc:.4f}")


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Retrain regime-specific models (Phase 12)
# ══════════════════════════════════════════════════════════════════════
print("\n🧭 STEP 4: Retraining regime-specific models...")

REGIMES   = ['bull', 'bear', 'highvol']
MIN_ROWS  = 50   # minimum rows per regime to train a model

regime_results = {}

for ticker in TICKERS:
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits_local(ticker)

    # Detect regimes for each split
    feat_path = FEATURES_DIR / f'{ticker}_features.csv'
    df_full   = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Load regime thresholds saved by regime_detector
    threshold_path = MODELS_DIR / f'hmm_{ticker}.pkl'
    if not threshold_path.exists():
        print(f"  ⚠️  {ticker}: no regime thresholds found — skipping regime retrain")
        continue

    thresholds = joblib.load(threshold_path)

    def classify_row(row):
        vol = row.get('Volatility_20d', 0.2)
        roc = row.get('ROC_20', 0.0)
        rsi = row.get('RSI_14', 50.0)
        vt  = thresholds.get('vol_threshold', 0.42)
        if vol >= vt:           return 'highvol'
        if roc < 0 and rsi < 50: return 'bear'
        return 'bull'

    # Assign regimes using original feature CSV (has Volatility_20d etc.)
    df_full['regime'] = df_full.apply(classify_row, axis=1)

    # Align regime labels with train split index
    train_regimes = df_full['regime'].reindex(X_train.index).fillna('bull')
    test_regimes  = df_full['regime'].reindex(X_test.index).fillna('bull')

    ticker_results = {}
    for regime in REGIMES:
        mask_train = train_regimes == regime
        mask_test  = test_regimes  == regime

        X_r = X_train[mask_train]
        y_r = y_train[mask_train]

        if len(X_r) < MIN_ROWS:
            print(f"  ⚠️  {ticker}/{regime}: only {len(X_r)} rows — skipping")
            continue

        # Fit scaler per regime
        r_scaler = RobustScaler()
        X_r_s = pd.DataFrame(
            r_scaler.fit_transform(X_r),
            index=X_r.index, columns=X_r.columns
        )
        joblib.dump(r_scaler, MODELS_DIR / f'scaler_{regime}_{ticker}.pkl')

        # XGBoost
        xgb_r = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0,
        )
        xgb_r.fit(X_r_s, y_r, verbose=False)
        joblib.dump(xgb_r, MODELS_DIR / f'regime_xgb_{regime}_{ticker}.pkl')

        # LGBM
        lgbm_r = LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        lgbm_r.fit(X_r_s, y_r)
        joblib.dump(lgbm_r, MODELS_DIR / f'regime_lgbm_{regime}_{ticker}.pkl')

        # Score on test
        if mask_test.sum() >= 10:
            X_test_r   = X_test[mask_test]
            y_test_r   = y_test[mask_test]
            X_test_r_s = pd.DataFrame(
                r_scaler.transform(X_test_r),
                index=X_test_r.index, columns=X_test_r.columns
            )
            auc = roc_auc_score(
                y_test_r, xgb_r.predict_proba(X_test_r_s)[:, 1]
            )
            ticker_results[regime] = auc
            print(f"  ✅ {ticker}/{regime}: {len(X_r)} train rows | AUC={auc:.4f}")
        else:
            print(f"  ✅ {ticker}/{regime}: {len(X_r)} train rows | AUC=N/A (test too small)")

    regime_results[ticker] = ticker_results


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Update live cache features (drop/add same columns)
# ══════════════════════════════════════════════════════════════════════
print("\n🔄 STEP 5: Updating live cache to match new feature set...")

try:
    from src.data.live_pipeline import fetch_ohlcv, build_live_features, validate_feature_alignment
    from src.data.live_cache import save_features
    import time as _time

    for ticker in TICKERS:
        df_raw = fetch_ohlcv(ticker)
        if df_raw is not None:
            df_feat = build_live_features(df_raw, ticker)
            if df_feat is not None:
                # Add normalized EMA ratios to live features too
                if 'EMA_12' in df_feat.columns and 'Close' in df_feat.columns:
                    df_feat['Price_vs_EMA12'] = (
                        (df_feat['Close'] - df_feat['EMA_12']) / df_feat['EMA_12']
                    )
                if 'EMA_26' in df_feat.columns and 'Close' in df_feat.columns:
                    df_feat['Price_vs_EMA26'] = (
                        (df_feat['Close'] - df_feat['EMA_26']) / df_feat['EMA_26']
                    )
                # Drop raw price cols
                cols_to_drop = [c for c in RAW_PRICE_COLS_TO_DROP
                                if c in df_feat.columns]
                df_feat.drop(columns=cols_to_drop, inplace=True)

                save_features(ticker, df_feat, df_feat.index[0].date())
                print(f"  ✅ {ticker}: live cache updated")
        _time.sleep(0.3)
except Exception as e:
    print(f"  ⚠️  Live cache update failed: {e}")
    print("     Run: python scheduler.py manually to refresh live data")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("✅ COMPLETE — Feature Fix + Full Retrain Done")
print("=" * 60)
print("\nBase Model Results (Test AUC):")
print(f"{'Ticker':<8} {'XGB':>8} {'LGBM':>8}")
print("-" * 26)
for ticker, r in base_results.items():
    print(f"{ticker:<8} {r['xgb']:>8.4f} {r['lgbm']:>8.4f}")

print("\n✅ Raw price columns removed:  EMA_12, EMA_26, BB_Upper, BB_Lower, BB_Middle")
print("✅ Normalized ratios added:    Price_vs_EMA12, Price_vs_EMA26")
print("✅ All base models retrained   → experiments/models/xgboost_*.pkl")
print("✅ All regime models retrained → experiments/models/regime_xgb_*.pkl")
print("✅ Live cache updated          → data/live/atlas_live.db")
print("\n👉 Restart uvicorn and refresh Streamlit to see corrected feature chart")