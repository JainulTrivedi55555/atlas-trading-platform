import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure the project root is in the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import TICKERS, RAW_DIR, PROCESSED_DIR

def clean_price_data(ticker: str) -> pd.DataFrame:
    """
    Clean a single ticker CSV.
    Returns a clean DataFrame ready for feature engineering.
    """
    # Load raw data
    fp = RAW_DIR / f'price/{ticker}_daily.csv'
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df.index.name = 'Date'
    
    # Step 1: Sort by date (always ascending) 
    df = df.sort_index(ascending=True)
    
    # Step 2: Remove duplicate dates 
    df = df[~df.index.duplicated(keep='first')]
    
    # Step 3: Handle missing values 
    # Forward fill price columns — if market was closed,
    # the last known price is the correct representation
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[price_cols] = df[price_cols].ffill()
    
    # Drop any remaining rows with NaN (very start of series)
    df = df.dropna(subset=price_cols)
    
    # Step 4: Remove outliers using domain knowledge 
    # A daily price change of more than 50% is almost certainly
    # a data error — not a real market move
    daily_return = df['Close'].pct_change()
    extreme_moves = daily_return.abs() > 0.50
    if extreme_moves.sum() > 0:
        print(f'  {ticker}: Removed {extreme_moves.sum()} extreme outliers')
        df = df[~extreme_moves]
        
    # Step 5: Add basic derived columns 
    df['Daily_Return'] = df['Close'].pct_change()                       # % daily return
    df['Log_Return']   = np.log(df['Close'] / df['Close'].shift(1))      # Log return
    df['Price_Range']  = df['High'] - df['Low']                          # Intraday range
    df['Gap']          = df['Open'] - df['Close'].shift(1)               # Overnight gap 
    
    # Step 6: Drop first row (NaN from pct_change) 
    df = df.dropna(subset=['Daily_Return'])
    
    print(f'  {ticker}: {len(df)} rows after cleaning')
    return df

def clean_all_tickers() -> dict:
    save_dir = PROCESSED_DIR / 'features'
    save_dir.mkdir(parents=True, exist_ok=True)
    cleaned = {}
    
    for ticker in TICKERS:
        df = clean_price_data(ticker)
        fp = save_dir / f'{ticker}_clean.csv'
        df.to_csv(fp)
        cleaned[ticker] = df
        
    return cleaned

if __name__ == '__main__':
    print('Cleaning price data...')
    data = clean_all_tickers()
    print(f'\nCleaned {len(data)} tickers')
    
    # Show sample
    if data:
        sample = list(data.values())[0]
        print(f'Columns: {list(sample.columns)}')
        print(sample[['Close', 'Daily_Return', 'Log_Return']].tail(5))