import sys, warnings, os
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import joblib
import pandas as pd
import numpy as np
from src.models.data_loader import load_splits
from src.risk.risk_engine import full_risk_report
from src.utils.config import TICKERS, PROCESSED_DIR

# ── Phase 12: Regime predictor ────────────────────────────────────────────────
try:
    from src.models.regime_predictor import (
        get_regime_signal, get_all_regime_signals
    )
    REGIME_MODELS_AVAILABLE = True
except ImportError:
    REGIME_MODELS_AVAILABLE = False
    print('WARNING: regime_predictor not found — regime endpoint will use base models')

# ── Phase 11: Live cache integration ─────────────────────────────────────────
# NOTE: load_features and LIVE_CACHE_AVAILABLE are exported so main.py can use them
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.data.live_cache import init_db, load_features, get_pipeline_status
    init_db()
    LIVE_CACHE_AVAILABLE = True
except ImportError:
    LIVE_CACHE_AVAILABLE = False
    load_features = None
    print('WARNING: live_cache not found — serving historical data only')

MODEL_DIR = Path(__file__).parent.parent / 'experiments/models'


def get_live_features_for_ticker(ticker: str) -> pd.DataFrame | None:
    """
    Returns live feature DataFrame (1x43) from cache if fresh, else None.
    Used by main.py regime endpoints to pass live data to regime predictor.
    """
    if not LIVE_CACHE_AVAILABLE or load_features is None:
        return None
    try:
        cached = load_features(ticker.upper())
        if cached and cached.get('is_fresh'):
            return cached['features']
    except Exception:
        pass
    return None


def get_features_for_ticker(ticker: str, feature_cols: list) -> tuple:
    """
    Returns (features_df, data_date_str, is_live).
    Tries live cache first; falls back to last row of historical data.
    """
    if LIVE_CACHE_AVAILABLE and load_features is not None:
        try:
            cached = load_features(ticker)
            if cached and cached.get('is_fresh'):
                live_features = cached['features'].copy()
                for col in feature_cols:
                    if col not in live_features.columns:
                        live_features[col] = 0
                return (
                    live_features[feature_cols],
                    cached['as_of_date'],
                    True
                )
        except Exception:
            pass

    # Fallback: last row of historical test split
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(ticker)
    X_test.index = pd.to_datetime(X_test.index).normalize()
    last_row = X_test.iloc[[-1]].copy()
    last_date = str(last_row.index[-1].date())
    for col in feature_cols:
        if col not in last_row.columns:
            last_row[col] = 0
    return (last_row[feature_cols], last_date, False)


def get_signal(ticker: str, model_type: str = 'xgboost') -> dict:
    """
    Get trading signal for a ticker.
    Returns signal, confidence, feature importances, and live/historical flag.
    """
    model_path = MODEL_DIR / f'{model_type}_{ticker}.pkl'
    if not model_path.exists():
        return {'error': f'No model for {ticker}'}

    model = joblib.load(model_path)

    try:
        feature_cols = model.get_booster().feature_names
    except Exception:
        X_train, _, _, _, X_test, _ = load_splits(ticker)
        feature_cols = list(X_train.columns)

    latest, data_date, is_live = get_features_for_ticker(ticker, feature_cols)

    proba   = model.predict_proba(latest)[0]
    prob_up = float(proba[1])
    signal  = 'BULLISH' if prob_up >= 0.5 else 'BEARISH'

    try:
        importances = model.feature_importances_
        feat_imp = dict(zip(
            feature_cols,
            [round(float(v), 4) for v in importances]
        ))
        top5 = dict(sorted(
            feat_imp.items(),
            key=lambda x: x[1], reverse=True
        )[:5])
    except Exception:
        top5 = {}

    return {
        'ticker':      ticker,
        'signal':      signal,
        'confidence':  round(prob_up if prob_up >= 0.5 else 1 - prob_up, 4),
        'prob_up':     round(prob_up, 4),
        'prob_down':   round(1 - prob_up, 4),
        'model':       model_type,
        'as_of_date':  data_date,
        'is_live':     is_live,
        'data_source': 'live_cache' if is_live else 'historical',
        'top_features': top5,
    }


def get_risk(ticker: str) -> dict:
    """Get risk metrics for a ticker."""
    try:
        return full_risk_report(ticker)
    except Exception as e:
        return {'error': str(e)}


def get_all_signals(model_type: str = 'xgboost') -> list:
    """Get signals for all 10 tickers at once."""
    results = []
    for ticker in TICKERS:
        sig = get_signal(ticker, model_type)
        results.append(sig)
    return results