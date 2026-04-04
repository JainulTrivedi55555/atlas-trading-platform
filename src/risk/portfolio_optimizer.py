import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR, TICKERS

def load_price_matrix(tickers=None,
                       start_date='2015-01-01',
                       end_date='2022-12-31') -> pd.DataFrame:
    """
    Load closing prices for all tickers into one DataFrame.
    Uses training period only to avoid data leakage.
    """
    if tickers is None:
        tickers = TICKERS
    prices = {}
    for ticker in tickers:
        try:
            path = PROCESSED_DIR / f'features/{ticker}_features.csv'
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).normalize()
            mask = (df.index >= start_date) & (df.index <= end_date)
            prices[ticker] = df.loc[mask, 'Close']
        except Exception as e:
            print(f'  Skip {ticker}: {e}')
    
    price_df = pd.DataFrame(prices).dropna()
    print(f'Price matrix: {price_df.shape} '
          f'({price_df.index.min().date()} to '
          f'{price_df.index.max().date()})')
    return price_df

def optimize_max_sharpe(prices: pd.DataFrame) -> dict:
    """
    Maximum Sharpe Ratio portfolio.
    Finds weights that maximise return per unit of risk.
    """
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models, expected_returns
    
    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, cov)
    ef.add_constraint(lambda w: w >= 0)       # No short selling
    ef.add_constraint(lambda w: w <= 0.30)    # Max 30% per stock
    
    weights = ef.max_sharpe(risk_free_rate=0.05)
    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(
        verbose=False, risk_free_rate=0.05
    )
    
    return {
        'strategy':   'Max Sharpe',
        'weights':    cleaned,
        'exp_return': round(perf[0] * 100, 2),
        'volatility': round(perf[1] * 100, 2),
        'sharpe':     round(perf[2], 3),
    }

def optimize_min_volatility(prices: pd.DataFrame) -> dict:
    """
    Minimum Volatility portfolio.
    Finds weights that minimise portfolio variance.
    """
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models, expected_returns
    
    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, cov)
    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= 0.30)
    
    weights = ef.min_volatility()
    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(
        verbose=False, risk_free_rate=0.05
    )
    
    return {
        'strategy':   'Min Volatility',
        'weights':    cleaned,
        'exp_return': round(perf[0] * 100, 2),
        'volatility': round(perf[1] * 100, 2),
        'sharpe':     round(perf[2], 3),
    }

def optimize_equal_weight(tickers=None) -> dict:
    """Equal weight baseline — 1/N for each ticker."""
    if tickers is None:
        tickers = TICKERS
    n = len(tickers)
    weights = {t: round(1/n, 4) for t in tickers}
    return {
        'strategy': 'Equal Weight',
        'weights':  weights,
        'exp_return': None,
        'volatility': None,
        'sharpe':     None,
    }

def optimize_model_weighted(prices: pd.DataFrame,
                             model_signals: dict) -> dict:
    """
    Model-aware portfolio: tilt weights toward tickers
    where ATLAS predicts up days.
    """
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models, expected_returns
    
    mu = expected_returns.mean_historical_return(prices)
    cov = risk_models.sample_cov(prices)
    
    # Blend historical returns with model signal
    blended_mu = mu.copy()
    for ticker, signal in model_signals.items():
        if ticker in blended_mu.index:
            # Scale: signal=0.7 adds 20% boost to expected return
            boost = (signal - 0.5) * 0.4
            blended_mu[ticker] = mu[ticker] * (1 + boost)
            
    ef = EfficientFrontier(blended_mu, cov)
    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= 0.30)
    
    weights = ef.max_sharpe(risk_free_rate=0.05)
    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(
        verbose=False, risk_free_rate=0.05
    )
    
    return {
        'strategy':   'ATLAS Model-Weighted',
        'weights':    cleaned,
        'exp_return': round(perf[0] * 100, 2),
        'volatility': round(perf[1] * 100, 2),
        'sharpe':     round(perf[2], 3),
    }

if __name__ == '__main__':
    prices = load_price_matrix()
    result = optimize_max_sharpe(prices)
    print(f'Max Sharpe: {result["sharpe"]}')
    print(f'Weights: {result["weights"]}')