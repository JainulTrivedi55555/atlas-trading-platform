import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR, TICKERS

TRANSACTION_COST = 0.001  # 0.1% per trade
INITIAL_CAPITAL  = 10000  # Starting portfolio value

def load_price_and_features(ticker: str,
                             start: str = '2023-07-01',
                             end:   str = '2025-12-31') -> pd.DataFrame:
    """Load price and feature data for backtesting period."""
    path = PROCESSED_DIR / f'features/{ticker}_features.csv'
    df   = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).normalize()
    mask = (df.index >= start) & (df.index <= end)
    df   = df.loc[mask].copy()
    print(f'  {ticker}: {len(df)} trading days '
          f'({df.index.min().date()} to {df.index.max().date()})')
    return df

def generate_signals(df: pd.DataFrame,
                      model,
                      feature_cols: list,
                      threshold: float = 0.5) -> pd.Series:
    """
    Generate daily trading signals from model.
    Returns Series of 1 (buy/hold) or 0 (sell/cash).
    """
    X = df[feature_cols].copy()
    X = X.fillna(method='ffill').fillna(0)
    proba = model.predict_proba(X)[:, 1]
    signals = pd.Series(
        (proba >= threshold).astype(int),
        index=df.index,
        name='signal'
    )
    return signals

def run_backtest(returns:  pd.Series,
                  signals:  pd.Series,
                  cost:     float = TRANSACTION_COST,
                  capital:  float = INITIAL_CAPITAL) -> pd.DataFrame:
    """
    Core vectorised backtest.
    """
    # Align signals and returns 
    common = returns.index.intersection(signals.index)
    ret = returns.loc[common]
    sig = signals.loc[common]
    
    # Position: shift signal by 1 day
    position = sig.shift(1).fillna(0)
    
    # Calculate transaction costs
    trades      = position.diff().abs().fillna(0)
    trade_costs = trades * cost
    
    # Strategy daily returns
    strategy_ret = position * ret - trade_costs
    
    # Build results DataFrame
    results = pd.DataFrame({
        'asset_return':    ret,
        'signal':          sig,
        'position':        position,
        'trade':           trades,
        'cost':            trade_costs,
        'strategy_return': strategy_ret,
    })
    
    # Equity curves
    results['strategy_equity'] = capital * (1 + results['strategy_return']).cumprod()
    results['asset_equity']    = capital * (1 + results['asset_return']).cumprod()
    
    return results

def calculate_metrics(results: pd.DataFrame,
                        label:   str = 'Strategy') -> dict:
    """Calculate all performance metrics from backtest results."""
    ret = results['strategy_return'].dropna()
    
    # Annualised return
    n_days     = len(ret)
    total_ret  = (1 + ret).prod() - 1
    ann_return = (1 + total_ret) ** (252 / n_days) - 1
    
    # Volatility
    ann_vol = ret.std() * np.sqrt(252)
    
    # Sharpe
    daily_rf = 0.05 / 252
    excess   = ret - daily_rf
    sharpe   = (excess.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
    
    # Sortino
    downside     = ret[ret < daily_rf]
    downside_std = downside.std() * np.sqrt(252)
    sortino      = ann_return / downside_std if (len(downside) > 0 and downside_std > 0) else 0
    
    # Max drawdown 
    equity       = results['strategy_equity']
    rolling_max  = equity.cummax()
    drawdown     = (equity - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calmar
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Trade statistics
    n_trades    = int(results['trade'].sum())
    wins        = (ret[results['position'] > 0] > 0).sum()
    losses      = (ret[results['position'] > 0] <= 0).sum()
    win_rate    = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    gross_profit = ret[ret > 0].sum()
    gross_loss   = abs(ret[ret < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    final_equity = results['strategy_equity'].iloc[-1]
    
    return {
        'label':         label,
        'ann_return':    round(ann_return * 100, 2),
        'ann_vol':       round(ann_vol * 100, 2),
        'sharpe':        round(sharpe, 3),
        'sortino':       round(sortino, 3),
        'calmar':        round(calmar, 3),
        'max_drawdown':  round(max_drawdown * 100, 2),
        'win_rate':      round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 3),
        'n_trades':      n_trades,
        'total_return':  round(total_ret * 100, 2),
        'final_equity':  round(final_equity, 2),
    }

def benchmark_buy_hold(returns: pd.Series,
                         capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """Buy and hold benchmark."""
    equity = capital * (1 + returns).cumprod()
    results = pd.DataFrame({
        'asset_return':    returns,
        'signal':          pd.Series(1, index=returns.index),
        'position':        pd.Series(1, index=returns.index),
        'trade':           pd.Series(0, index=returns.index),
        'cost':            pd.Series(0, index=returns.index),
        'strategy_return': returns,
        'strategy_equity': equity,
        'asset_equity':    equity,
    })
    return results

def classify_regime(row, vix_threshold=25):
    """Classify a single day into one of 4 market regimes."""
    vix_high  = row.get('vix', 20) > vix_threshold
    above_sma = row.get('Price_vs_SMA200', 0) > 0
    
    if not vix_high and above_sma:     
        return 'Low Vol Bull'
    elif vix_high and above_sma:       
        return 'High Vol Bull'
    elif not vix_high and not above_sma: 
        return 'Low Vol Bear'
    else:                              
        return 'High Vol Bear'

def regime_performance(results:  pd.DataFrame,
                         features: pd.DataFrame) -> pd.DataFrame:
    """Break down strategy performance by market regime."""
    macro_path = PROCESSED_DIR / 'features/macro_clean.csv'
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    macro.index = pd.to_datetime(macro.index).normalize()
    
    combined = features.copy()
    if 'vix' not in combined.columns and 'vix' in macro.columns:
        combined['vix'] = macro['vix']
    
    combined['regime'] = combined.apply(classify_regime, axis=1)
    common = results.index.intersection(combined.index)
    results_r = results.loc[common].copy()
    results_r['regime'] = combined.loc[common, 'regime']
    
    regime_stats = []
    for regime in ['Low Vol Bull','High Vol Bull', 'Low Vol Bear','High Vol Bear']:
        mask = results_r['regime'] == regime
        if mask.sum() < 10:
            continue
        r   = results_r.loc[mask, 'strategy_return']
        bh  = results_r.loc[mask, 'asset_return']
        regime_stats.append({
            'Regime':       regime,
            'Days':         mask.sum(),
            'Strategy Ret': round(r.mean() * 252 * 100, 1),
            'BuyHold Ret':  round(bh.mean() * 252 * 100, 1),
            'Win Rate':     round((r > 0).mean() * 100, 1),
            'Sharpe':       round((r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0, 3),
        })
    return pd.DataFrame(regime_stats).set_index('Regime')