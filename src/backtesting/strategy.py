import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR
from src.models.data_loader import load_splits
from src.backtesting.backtest_engine import (
    load_price_and_features,
    generate_signals,
    run_backtest,
    calculate_metrics,
    benchmark_buy_hold,
    regime_performance,
)

MODEL_DIR = Path(__file__).parent.parent.parent / 'experiments/models'

def run_atlas_strategy(ticker:    str,
                        model_name: str = 'xgboost',
                        threshold:  float = 0.5) -> dict:
    """Run ATLAS model strategy for one ticker."""

    # Load model
    model_file = MODEL_DIR / f'{model_name}_{ticker}.pkl'
    if not model_file.exists():
        print(f'  Model not found: {model_file}')
        return None
    model = joblib.load(model_file)

    # Load 43-feature splits — correct for Phase 4 models
    X_train, y_train, X_val, y_val, X_test, y_test = (
        load_splits(ticker)
    )
    X_test.index = pd.to_datetime(X_test.index).normalize()
    y_test.index = pd.to_datetime(y_test.index).normalize()

    # Get feature names from model
    try:
        feature_cols = model.get_booster().feature_names
    except Exception:
        feature_cols = list(X_train.columns)

    # Fill any missing features with 0
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0

    # Generate signals directly from X_test
    # This guarantees index alignment
    X_input = X_test[feature_cols].copy().ffill().fillna(0)
    proba   = model.predict_proba(X_input)[:, 1]
    signals = pd.Series(
        (proba >= threshold).astype(int),
        index=X_test.index,
        name='signal'
    )

    # Get returns aligned to same index
    df      = load_price_and_features(ticker)
    df.index = pd.to_datetime(df.index).normalize()
    returns = df['Daily_Return'].dropna()
    returns.index = pd.to_datetime(returns.index).normalize()

    # Run backtest on overlapping dates only
    common  = returns.index.intersection(signals.index)
    print(f'  {ticker} {model_name}: '
          f'{len(common)} overlapping days, '
          f'{signals.loc[common].sum()} long signals')

    results = run_backtest(
        returns.loc[common],
        signals.loc[common]
    )

    metrics = calculate_metrics(
        results,
        label=f'ATLAS {model_name.upper()} ({ticker})'
    )

    try:
        regime_df = regime_performance(results, df)
    except Exception:
        regime_df = None

    return {
        'results':   results,
        'metrics':   metrics,
        'regime_df': regime_df,
        'signals':   signals,
        'ticker':    ticker,
        'model':     model_name,
    }

def run_buy_hold(ticker: str) -> dict:
    """Buy-and-hold benchmark for one ticker."""
    df      = load_price_and_features(ticker)
    returns = df['Daily_Return'].dropna()
    results = benchmark_buy_hold(returns)
    metrics = calculate_metrics(
        results, label=f'Buy-and-Hold ({ticker})'
    )
    return {'results': results, 'metrics': metrics, 'ticker': ticker}

def run_random_strategy(ticker: str,
                         seed:   int = 42) -> dict:
    """Random signal baseline — coin flip every day."""
    df      = load_price_and_features(ticker)
    returns = df['Daily_Return'].dropna()
    np.random.seed(seed)
    signals = pd.Series(
        np.random.randint(0, 2, len(returns)),
        index=returns.index
    )
    results = run_backtest(returns, signals)
    metrics = calculate_metrics(
        results, label=f'Random ({ticker})'
    )
    return {'results': results, 'metrics': metrics, 'ticker': ticker}

def compare_all_strategies(ticker: str) -> pd.DataFrame:
    """Run all 3 strategies and return comparison table."""
    print(f'\nBacktesting {ticker}...')
    all_metrics = []
    
    # ATLAS XGBoost
    xgb = run_atlas_strategy(ticker, 'xgboost')
    if xgb:
        all_metrics.append(xgb['metrics'])
        
    # ATLAS LightGBM
    lgbm = run_atlas_strategy(ticker, 'lgbm')
    if lgbm:
        all_metrics.append(lgbm['metrics'])
        
    # Buy and Hold
    bh = run_buy_hold(ticker)
    all_metrics.append(bh['metrics'])
    
    # Random baseline
    rand = run_random_strategy(ticker)
    all_metrics.append(rand['metrics'])
    
    return pd.DataFrame(all_metrics).set_index('label')