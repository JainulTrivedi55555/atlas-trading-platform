import pandas as pd
from fredapi import Fred
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import FRED_API_KEY, TRAIN_START, TEST_END, RAW_DIR

MACRO_SERIES = {
    'fed_funds_rate': 'FEDFUNDS',        # Federal Funds Rate
    'cpi_inflation':  'CPIAUCSL',         # Consumer Price Index
    'unemployment':   'UNRATE',            # Unemployment Rate
    'gdp_growth':     'A191RL1Q225SBEA',  # Real GDP Growth %
    'treasury_10yr':  'DGS10',             # 10-Year Treasury Yield
    'treasury_2yr':   'DGS2',              # 2-Year Treasury Yield
    'yield_curve':    'T10Y2Y',            # 10Y minus 2Y (recession indicator)
    'vix':            'VIXCLS',            # Volatility Index
    'sp500':          'SP500',             # S&P 500 Index Level
}

def download_macro_data():
    fred = Fred(api_key=FRED_API_KEY)
    save_dir = RAW_DIR / 'macro'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    all_series = {}
    for name, code in MACRO_SERIES.items():
        try:
            s = fred.get_series(
                code,
                observation_start=TRAIN_START,
                observation_end=TEST_END
            )
            all_series[name] = s
            print(f'  {name}: {len(s)} points')
        except Exception as e:
            print(f'  ERROR {name}: {e}')
            
    # Macro data is monthly/quarterly — resample to daily with forward fill
    macro_df = pd.DataFrame(all_series).resample('D').ffill()
    macro_df.to_csv(RAW_DIR / 'macro/macro_indicators.csv')
    print(f'Saved. Shape: {macro_df.shape}')
    return macro_df

if __name__ == '__main__':
    macro = download_macro_data()
    print(macro.tail())