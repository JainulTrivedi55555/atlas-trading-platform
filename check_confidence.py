"""
Check live confidence values for all 25 tickers.
Run from project root: python check_confidence.py
"""
import sys
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')

from src.data.live_pipeline import fetch_ohlcv, build_live_features, get_model_feature_cols

ROOT       = Path('.')
MODELS_DIR = ROOT / 'experiments' / 'models'
DATA_PROC  = ROOT / 'data' / 'processed'

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'JPM',  'GS',   'BAC',   'NVDA', 'TSLA',
    'NFLX', 'ORCL', 'AMD',   'CRM',  'UBER',
    'WMT',  'JNJ',  'XOM',   'LLY',  'V',
    'SPY',  'QQQ',  'IWM',   'GLD',  'TLT',
]

print("=" * 55)
print("  ATLAS — Live Confidence Check (all 25 tickers)")
print("=" * 55)
print(f"{'Ticker':<6}  {'Prob Up':>8}  {'Confidence':>10}  {'Signal':>8}  {'Status'}")
print("-" * 55)

results = []

for ticker in TICKERS:
    try:
        # Fetch live data
        df_raw = fetch_ohlcv(ticker)
        if df_raw is None:
            print(f"{ticker:<6}  {'N/A':>8}  {'N/A':>10}  {'N/A':>8}  FETCH FAILED")
            continue

        # Build features
        df_feat = build_live_features(df_raw, ticker)
        if df_feat is None:
            print(f"{ticker:<6}  {'N/A':>8}  {'N/A':>10}  {'N/A':>8}  FEATURE FAILED")
            continue

        # Load model training columns
        train_cols = get_model_feature_cols(ticker)
        if train_cols is None:
            print(f"{ticker:<6}  {'N/A':>8}  {'N/A':>10}  {'N/A':>8}  NO SPLIT FILES")
            continue

        # Align live features to training columns
        live = df_feat.copy()
        for col in train_cols:
            if col not in live.columns:
                live[col] = 0.0
        live = live[train_cols]

        # Load model
        model_path = MODELS_DIR / f'xgboost_{ticker}.pkl'
        if not model_path.exists():
            print(f"{ticker:<6}  {'N/A':>8}  {'N/A':>10}  {'N/A':>8}  NO MODEL FILE")
            continue

        model   = joblib.load(model_path)
        prob_up = model.predict_proba(live.values)[0][1]
        conf    = abs(prob_up - 0.5) * 2
        signal  = 'BULLISH' if prob_up >= 0.5 else 'BEARISH'

        results.append({
            'ticker':     ticker,
            'prob_up':    prob_up,
            'confidence': conf,
            'signal':     signal,
        })

        # Flag if above common thresholds
        flag = ''
        if conf >= 0.55: flag = '<-- passes 0.55'
        elif conf >= 0.40: flag = '<-- passes 0.40'
        elif conf >= 0.30: flag = '<-- passes 0.30'

        print(f"{ticker:<6}  {prob_up:>8.3f}  {conf:>10.3f}  {signal:>8}  {flag}")

    except Exception as e:
        print(f"{ticker:<6}  ERROR: {e}")

print("=" * 55)

if results:
    confs = [r['confidence'] for r in results]
    print(f"\n  Min confidence : {min(confs):.3f}")
    print(f"  Max confidence : {max(confs):.3f}")
    print(f"  Avg confidence : {sum(confs)/len(confs):.3f}")
    print()
    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        passing = [r['ticker'] for r in results if r['confidence'] >= threshold]
        print(f"  Threshold {threshold:.2f} -> {len(passing):>2} tickers pass: {passing}")

print("=" * 55)
