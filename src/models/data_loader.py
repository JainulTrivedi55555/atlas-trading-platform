import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import TICKERS, PROCESSED_DIR

def load_splits(ticker: str, target: str = 'Target_Direction'):
    """
    Load train/val/test splits for one ticker.
    Returns X_train, y_train, X_val, y_val, X_test, y_test
    """
    split_dir = PROCESSED_DIR / f'splits/{ticker}'
    X_train = pd.read_csv(split_dir / 'X_train.csv',
                          index_col=0, parse_dates=True)
    y_train = pd.read_csv(split_dir / 'y_train.csv',
                          index_col=0, parse_dates=True).squeeze()
    X_val   = pd.read_csv(split_dir / 'X_val.csv',
                          index_col=0, parse_dates=True)
    y_val   = pd.read_csv(split_dir / 'y_val.csv',
                          index_col=0, parse_dates=True).squeeze()
    X_test  = pd.read_csv(split_dir / 'X_test.csv',
                          index_col=0, parse_dates=True) 
    y_test  = pd.read_csv(split_dir / 'y_test.csv',
                          index_col=0, parse_dates=True).squeeze()

    # Fix index types
    for y in [y_train, y_val, y_test]:
        y.index = pd.to_datetime(y.index)
        
    print(f'{ticker} — Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}')
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_class_weight(y_train):
    """Compute scale_pos_weight for XGBoost/LightGBM."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return float(n_neg / n_pos)

def load_all_tickers(target: str = 'Target_Direction'):
    """Load splits for all 10 tickers. Returns dict."""
    all_data = {}
    for ticker in TICKERS:
        try:
            splits = load_splits(ticker, target)
            all_data[ticker] = splits
        except Exception as e:
            print(f'ERROR loading {ticker}: {e}')
            
    print(f'Loaded {len(all_data)} tickers')
    return all_data

if __name__ == '__main__':
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_splits('AAPL')
    print(f'X_train shape: {X_tr.shape}')
    print(f'Class weight:  {get_class_weight(y_tr):.3f}')