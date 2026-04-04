"""
ATLAS Diagnostic Script
========================
Run this BEFORE backtest_report.py to find out why signals are all zero.

    conda activate fintech
    python diagnose_atlas.py
"""

import sys, warnings, joblib
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.config import TICKERS, PROCESSED_DIR
from src.backtesting.backtest_engine import load_price_and_features

MODEL_DIR      = ROOT / "experiments" / "models"
BACKTEST_START = "2023-07-01"
BACKTEST_END   = "2025-12-31"
TICKER         = "AAPL"   # just test one ticker

print("\n" + "=" * 65)
print("  ATLAS SIGNAL DIAGNOSTIC")
print("=" * 65)

# ── 1. Load feature data ──────────────────────────────────────────
print(f"\n[1] Loading feature data for {TICKER}...")
df = load_price_and_features(TICKER, start=BACKTEST_START, end=BACKTEST_END)
print(f"    Shape: {df.shape}")
print(f"    Columns ({len(df.columns)}): {list(df.columns)[:10]} ...")
print(f"    First 3 rows:\n{df.head(3)}")
print(f"    NaN counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── 2. Load model ─────────────────────────────────────────────────
print(f"\n[2] Loading XGBoost model for {TICKER}...")
model = joblib.load(MODEL_DIR / f"xgboost_{TICKER}.pkl")
print(f"    Model type: {type(model)}")

if hasattr(model, "feature_names_in_"):
    trained_features = list(model.feature_names_in_)
    print(f"    Model trained on {len(trained_features)} features:")
    print(f"    {trained_features[:10]} ...")
else:
    print("    ⚠️  Model has no feature_names_in_ — was trained without named columns")
    trained_features = None

# ── 3. Check feature overlap ──────────────────────────────────────
print(f"\n[3] Feature overlap check...")
exclude = {"target", "close", "open", "high", "low", "volume",
           "returns", "Close", "Open", "High", "Low", "Volume"}
available_features = [c for c in df.columns if c not in exclude]
print(f"    Features in data:  {len(available_features)}")

if trained_features:
    in_both    = [f for f in trained_features if f in df.columns]
    in_model_only = [f for f in trained_features if f not in df.columns]
    in_data_only  = [f for f in available_features if f not in trained_features]

    print(f"    Features in model: {len(trained_features)}")
    print(f"    ✅ Match (in both):       {len(in_both)}")
    print(f"    ❌ In model, NOT in data: {len(in_model_only)}")
    if in_model_only:
        print(f"       Missing: {in_model_only[:20]}")
    print(f"    ➕ In data, NOT in model: {len(in_data_only)}")
    feature_cols = in_both
else:
    feature_cols = available_features
    print(f"    Using all {len(feature_cols)} available features")

# ── 4. Try prediction ─────────────────────────────────────────────
print(f"\n[4] Running predict_proba on first 10 rows...")
try:
    X = df[feature_cols].copy().fillna(method="ffill").fillna(0)
    proba = model.predict_proba(X)[:, 1]
    print(f"    Probabilities (first 20): {np.round(proba[:20], 3)}")
    print(f"    Min prob: {proba.min():.4f}")
    print(f"    Max prob: {proba.max():.4f}")
    print(f"    Mean prob: {proba.mean():.4f}")
    print(f"    % above 0.55 threshold: {(proba >= 0.55).mean()*100:.1f}%")
    print(f"    % above 0.50 threshold: {(proba >= 0.50).mean()*100:.1f}%")
    print(f"    % above 0.45 threshold: {(proba >= 0.45).mean()*100:.1f}%")

    if proba.max() < 0.55:
        print(f"\n    ⚠️  Model NEVER reaches 0.55 threshold!")
        print(f"    ⚠️  Max confidence is only {proba.max():.4f}")
        print(f"    → Solution: lower threshold to {proba.max()*0.9:.2f} or retrain")
    else:
        print(f"\n    ✅ Model does produce signals above 0.55 threshold")
        print(f"    → The issue is a feature mismatch in backtest_report.py")

except Exception as e:
    print(f"    ❌ predict_proba failed: {e}")
    print(f"    → This confirms a feature mismatch")

# ── 5. Check what features the model actually expects ─────────────
print(f"\n[5] Checking data types...")
print(df[feature_cols[:5]].dtypes)
print(f"\nDone! Paste this output so we can fix the issue.\n")
