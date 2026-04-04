import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR, TICKERS

def load_returns(ticker: str) -> pd.Series:
    """Load daily returns for a ticker from features CSV."""
    path = PROCESSED_DIR / f'features/{ticker}_features.csv'
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    returns = df['Daily_Return'].dropna()
    return returns

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk.
    Returns the loss at the given confidence level.
    """
    return float(np.percentile(returns, (1 - confidence) * 100))

def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall).
    Average return on the worst (1-confidence)% of days.
    """
    var = calculate_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var

def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown.
    Returns negative number — e.g. -0.34 = 34% max loss.
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())

def calculate_sharpe(returns: pd.Series, risk_free: float = 0.05) -> float:
    """
    Annualised Sharpe Ratio.
    risk_free: annual risk-free rate (default 5% = 2024 T-bill)
    """
    daily_rf = risk_free / 252
    excess = returns - daily_rf
    if returns.std() == 0:
        return 0.0
    return float((excess.mean() / returns.std()) * np.sqrt(252))

def calculate_sortino(returns: pd.Series, risk_free: float = 0.05) -> float:
    """
    Annualised Sortino Ratio.
    Only penalises downside volatility unlike Sharpe.
    """
    daily_rf = risk_free / 252
    excess = returns - daily_rf
    downside = returns[returns < daily_rf]
    if len(downside) < 2 or downside.std() == 0:
        return 0.0
    downside_std = downside.std() * np.sqrt(252)
    ann_return = returns.mean() * 252
    return float(ann_return / downside_std)

def calculate_calmar(returns: pd.Series) -> float:
    """
    Calmar Ratio = Annualised Return / abs(Max Drawdown).
    """
    ann_return = returns.mean() * 252
    max_drawdown = calculate_max_drawdown(returns)
    if max_drawdown == 0:
        return 0.0
    return float(ann_return / abs(max_drawdown))

def full_risk_report(ticker: str, start_date: str = None, end_date: str = None) -> dict:
    """
    Calculate all risk metrics for one ticker.
    """
    returns = load_returns(ticker)
    if start_date:
        returns = returns[returns.index >= start_date]
    if end_date:
        returns = returns[returns.index <= end_date]
    
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    
    report = {
        'ticker':          ticker,
        'n_days':          len(returns),
        'ann_return':      round(ann_return * 100, 2),
        'ann_volatility':  round(ann_vol * 100, 2),
        'var_95':          round(calculate_var(returns, 0.95) * 100, 3),
        'var_99':          round(calculate_var(returns, 0.99) * 100, 3),
        'cvar_95':         round(calculate_cvar(returns, 0.95) * 100, 3),
        'max_drawdown':    round(calculate_max_drawdown(returns) * 100, 2),
        'sharpe_ratio':    round(calculate_sharpe(returns), 3),
        'sortino_ratio':   round(calculate_sortino(returns), 3),
        'calmar_ratio':    round(calculate_calmar(returns), 3),
    }
    return report

def risk_report_all_tickers(tickers=None, start_date=None, end_date=None) -> pd.DataFrame:
    """Calculate risk metrics for all tickers."""
    if tickers is None:
        tickers = TICKERS
    reports = []
    for ticker in tickers:
        try:
            r = full_risk_report(ticker, start_date, end_date)
            reports.append(r)
            print(f'  {ticker}: Sharpe={r["sharpe_ratio"]:.3f} '
                  f'MaxDD={r["max_drawdown"]:.1f}%')
        except Exception as e:
            print(f'  ERROR {ticker}: {e}')
    return pd.DataFrame(reports).set_index('ticker')

if __name__ == '__main__':
    df = risk_report_all_tickers()
    print('\n' + '='*60)
    print('ATLAS RISK REPORT — ALL TICKERS')
    print('='*60)
    print(df.to_string())