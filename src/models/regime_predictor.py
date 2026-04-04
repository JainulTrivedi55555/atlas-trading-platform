"""
ATLAS Phase 12 — Regime-Aware Predictor
Selects the correct regime model at inference time.
Falls back gracefully to Phase 10 base model if regime model missing.
"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import joblib
from src.models.regime_detector import detect_current_regime
from src.models.data_loader import load_splits
from src.utils.config import TICKERS

MODEL_DIR = Path('experiments/models')
REGIMES = ['bull', 'bear', 'highvol']
ALGOS   = ['xgb', 'lgbm']

def regime_model_exists(algo: str, regime: str, ticker: str) -> bool:
    """Check if a regime model file exists."""
    path        = MODEL_DIR / f'regime_{algo}_{regime}_{ticker}.pkl'
    scaler_path = MODEL_DIR / f'scaler_{regime}_{ticker}.pkl'
    return path.exists() and scaler_path.exists()

def get_regime_signal(
    ticker: str,
    model_type: str = 'xgboost',
    live_features: pd.DataFrame = None
) -> dict:
    """
    Get a regime-aware trading signal for a ticker.
    """
    algo_key = 'xgb' if 'xgb' in model_type.lower() else 'lgbm'

    # 🟢 Step 1: Detect current regime
    try:
        regime = detect_current_regime(ticker, live_features)
    except Exception as e:
        print(f"  Regime detection failed for {ticker}: {e} — using bull")
        regime = 'bull'

    # 🔵 Step 2: Resolve feature source
    if live_features is not None:
        features = live_features
        is_live  = True
        # ── FIX 1: safely extract actual date from live cache index ──────────
        try:
            idx = features.index[-1]
            if hasattr(idx, 'date'):
                as_of = str(idx.date())
            elif hasattr(idx, 'strftime'):
                as_of = str(idx.strftime('%Y-%m-%d'))
            else:
                # Index is a plain integer — fetch date directly from live cache
                from src.data.live_cache import load_features as _lf
                _cached = _lf(ticker)
                as_of = _cached['as_of_date'] if _cached else str(pd.Timestamp.today().date())
        except Exception:
            as_of = str(pd.Timestamp.today().date())
        # ─────────────────────────────────────────────────────────────────────
    else:
        X_train, _, _, _, X_test, _ = load_splits(ticker)
        X_test.index = pd.to_datetime(X_test.index).normalize()
        features = X_test.iloc[[-1]]
        is_live  = False
        as_of    = str(features.index[-1].date())

    # 🟡 Step 3: Try regime-specific model
    fallback_used = False
    if regime_model_exists(algo_key, regime, ticker):
        scaler_path = MODEL_DIR / f'scaler_{regime}_{ticker}.pkl'
        model_path  = MODEL_DIR / f'regime_{algo_key}_{regime}_{ticker}.pkl'
        scaler     = joblib.load(scaler_path)
        model      = joblib.load(model_path)
        model_used = f'regime_{algo_key}_{regime}'

        # Get feature columns from model
        try:
            feature_cols = model.get_booster().feature_names
        except Exception:
            feature_cols = list(features.columns)

        # Align features
        feat_aligned = features.copy()
        for col in feature_cols:
            if col not in feat_aligned.columns:
                feat_aligned[col] = 0
        feat_aligned = feat_aligned[feature_cols]

        # Scale
        feat_scaled = pd.DataFrame(
            scaler.transform(feat_aligned),
            columns=feature_cols,
            index=feat_aligned.index
        )
    else:
        # Fallback to Phase 10 base model
        fallback_used = True
        base_path = MODEL_DIR / f'{"xgboost" if algo_key == "xgb" else "lgbm"}_{ticker}.pkl'
        if not base_path.exists():
            return {'error': f'No model found for {ticker}'}

        model      = joblib.load(base_path)
        model_used = f'base_{model_type}'

        try:
            feature_cols = model.get_booster().feature_names
        except Exception:
            _, _, _, _, X_test, _ = load_splits(ticker)
            feature_cols = list(X_test.columns)

        feat_aligned = features.copy()
        for col in feature_cols:
            if col not in feat_aligned.columns:
                feat_aligned[col] = 0
        feat_scaled = feat_aligned[feature_cols]

    # 🟠 Step 4: Predict
    proba   = model.predict_proba(feat_scaled)[0]
    prob_up = float(proba[1])
    signal  = 'BULLISH' if prob_up >= 0.5 else 'BEARISH'

    # 🟣 Step 5: Top 5 feature importances
    try:
        importances = model.feature_importances_
        feat_imp = dict(zip(feature_cols, [round(float(v), 4) for v in importances]))
        top5     = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5])
    except Exception:
        top5 = {}

    # 🔴 Regime label for display
    regime_display = {
        'bull':    '🟢 Bull Market',
        'bear':    '🔴 Bear Market',
        'highvol': '🟡 High Volatility',
    }.get(regime, regime)

    return {
        'ticker':         ticker,
        'signal':         signal,
        'confidence':     round(prob_up if prob_up >= 0.5 else 1 - prob_up, 4),
        'prob_up':        round(prob_up, 4),
        'prob_down':      round(1 - prob_up, 4),
        'regime':         regime,
        'regime_display': regime_display,
        'model_used':     model_used,
        'model_type':     model_type,
        'as_of_date':     as_of,
        'is_live':        is_live,
        'data_source':    'live_cache' if is_live else 'historical',
        'fallback_used':  fallback_used,
        'top_features':   top5,
    }


def get_all_regime_signals(model_type: str = 'xgboost') -> list:
    """Get regime-aware signals for all 10 tickers using live cache."""
    # ── FIX 2: fetch live features for each ticker, same as single-ticker endpoint
    try:
        from app.atlas_engine import get_live_features_for_ticker
        live_cache_ok = True
    except Exception:
        live_cache_ok = False

    results = []
    for ticker in TICKERS:
        try:
            live_feats = get_live_features_for_ticker(ticker) if live_cache_ok else None
            sig        = get_regime_signal(ticker, model_type, live_features=live_feats)
            results.append(sig)
        except Exception as e:
            results.append({'ticker': ticker, 'error': str(e)})
    return results