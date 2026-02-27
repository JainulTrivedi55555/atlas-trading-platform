import yfinance as yf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Ensure the project root is in the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import TICKERS, TRAIN_START, TEST_END, RAW_DIR

def download_price_data(tickers, start, end):
    data = {}
    save_dir = RAW_DIR / 'price'
    save_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tqdm(tickers, desc='Downloading prices'):
        try:
            # Updated method — more reliable in newer yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=start,
                end=end,
                auto_adjust=True
            )

            if df.empty:
                print(f'WARNING: No data for {ticker}')
                continue

            # Clean up column names
            df.index = df.index.tz_localize(None)  # Remove timezone
            df.to_csv(save_dir / f'{ticker}_daily.csv')
            data[ticker] = df
            print(f'  {ticker}: {len(df)} rows saved')

        except Exception as e:
            print(f'ERROR {ticker}: {e}')

    return data

def load_price_data(ticker):
    """Load a previously downloaded ticker CSV."""
    fp = RAW_DIR / 'price' / f'{ticker}_daily.csv'
    if not fp.exists():
        raise FileNotFoundError(f'Run download first for {ticker}')
    return pd.read_csv(fp, index_col=0, parse_dates=True)

if __name__ == '__main__':
    # Execute the download process
    data = download_price_data(TICKERS, TRAIN_START, TEST_END)

    print(f'\nDownloaded {len(data)} tickers')
    
    if data:
        # Sanity check: show the last 3 rows of the first ticker in the dictionary
        first_ticker = list(data.keys())[0]
        print(f"\nLast 3 rows for {first_ticker}:")
        print(data[first_ticker].tail(3))