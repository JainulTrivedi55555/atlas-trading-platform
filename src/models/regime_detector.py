"""
ATLAS Phase 12 — Market Regime Detector
Rule-based regime classification: BULL | BEAR | HIGHVOL

Uses actual column names from your dataset:
  Volatility_20d, RSI_14, ROC_20, Daily_Return
"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
from src.models.data_loader import load_splits
from src.utils.config import TICKERS

MODEL_DIR = Path('experiments/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'highvol'}


def _get_val(row, candidates, default=0.0):
    """
    Try multiple possible column names and return the first match.
    This makes the detector robust to column naming differences.
    """
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return default


def _classify_regime_row(row, vol_threshold: float) -> str:
    """
    Classify a single row into bull / bear / highvol.

    Rules:
      HIGHVOL : Volatility_20d > 75th percentile threshold
      BEAR    : (not HIGHVOL) AND ROC_20 < 0 AND RSI_14 < 50
      BULL    : everything else
    """
    vol = _get_val(row, ['Volatility_20d', 'hist_vol20', 'volatility_20d'], 0.2)
    roc = _get_val(row, ['ROC_20', 'roc20', 'ROC20'], 0.0)
    rsi = _get_val(row, ['RSI_14', 'rsi14', 'RSI14'], 50.0)

    if vol >= vol_threshold:
        return 'highvol'

    if roc < 0 and rsi < 50:
        return 'bear'

    return 'bull'


def fit_regime_detector(ticker: str) -> dict:
    """
    Compute regime thresholds from historical training data for a ticker.
    Saves thresholds to experiments/models/hmm_{ticker}.pkl
    Returns the thresholds dict.
    """
    print(f"  Fitting regime detector for {ticker}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(ticker)

    # Compute vol threshold from training data only (no data leakage)
    vol_col = None
    for c in ['Volatility_20d', 'hist_vol20', 'volatility_20d']:
        if c in X_train.columns:
            vol_col = c
            break

    if vol_col:
        vol_series = X_train[vol_col].dropna()
        vol_threshold = float(vol_series.quantile(0.75))
    else:
        vol_threshold = 0.20  # sensible default ~20% annualised vol
        print(f"    WARNING: No volatility column found, using default {vol_threshold}")

    thresholds = {
        'vol_threshold': vol_threshold,
        'ticker':        ticker,
        'method':        'rule_based',
        'vol_col':       vol_col,
    }

    save_path = MODEL_DIR / f'hmm_{ticker}.pkl'
    joblib.dump(thresholds, save_path)
    print(f"    Thresholds saved to {save_path}")
    print(f"    vol_threshold (75th pct of {vol_col}): {vol_threshold:.4f}")
    return thresholds


def _load_thresholds(ticker: str) -> dict:
    """Load thresholds, refitting if file is missing or old HMM format."""
    save_path = MODEL_DIR / f'hmm_{ticker}.pkl'
    if save_path.exists():
        payload = joblib.load(save_path)
        # Old HMM object or wrong dict format — refit
        if not isinstance(payload, dict) or 'vol_threshold' not in payload:
            print(f"  Old format detected for {ticker} — refitting...")
            return fit_regime_detector(ticker)
        return payload
    return fit_regime_detector(ticker)


def label_regimes(ticker: str, hmm=None) -> pd.Series:
    """
    Labels every row in the full historical data with a regime.
    Returns a pd.Series with values: 'bull', 'bear', 'highvol'

    hmm parameter kept for API compatibility — pass thresholds dict or None.
    """
    if hmm is not None and isinstance(hmm, dict) and 'vol_threshold' in hmm:
        thresholds = hmm
    else:
        thresholds = _load_thresholds(ticker)

    vol_threshold = thresholds['vol_threshold']

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(ticker)
    X_all = pd.concat([X_train, X_val, X_test])
    X_all.index = pd.to_datetime(X_all.index).normalize()
    X_all = X_all.sort_index()

    regimes = X_all.apply(
        lambda row: _classify_regime_row(row, vol_threshold), axis=1
    )
    regimes.index = X_all.index
    return regimes


def detect_current_regime(ticker: str,
                           live_features: pd.DataFrame = None) -> str:
    """
    Detects the current market regime for a ticker.
    Uses live features if provided, otherwise uses last row of historical data.
    Returns: 'bull', 'bear', or 'highvol'
    """
    thresholds = _load_thresholds(ticker)
    vol_threshold = thresholds['vol_threshold']

    if live_features is not None:
        X = live_features
    else:
        X_train, _, X_val, _, X_test, _ = load_splits(ticker)
        X_all = pd.concat([X_train, X_val, X_test])
        X_all.index = pd.to_datetime(X_all.index).normalize()
        X = X_all.sort_index()

    last_row = X.iloc[-1]
    regime = _classify_regime_row(last_row, vol_threshold)
    return regime