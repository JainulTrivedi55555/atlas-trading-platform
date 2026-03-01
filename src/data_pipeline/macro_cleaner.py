import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure the project root is in the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import RAW_DIR, PROCESSED_DIR

def clean_macro_data() -> pd.DataFrame:
    """Clean macro indicators and engineer macro features."""
    fp = RAW_DIR / 'macro/macro_indicators.csv'
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    df = df.sort_index()
    
    print(f'Raw shape: {df.shape}')
    print(f'Missing values before cleaning:')
    print(df.isnull().sum())

    # Step 1: Forward fill all macro indicators 
    # This is correct for macro data - if the Fed hasn't changed
    # rates this month, last month's rate is still the current rate
    df = df.ffill()

    # Step 2: Backward fill for any remaining NaN at start
    df = df.bfill()

    # Step 3: Engineer macro features 
    # Rate of change - how fast is inflation accelerating?
    df['cpi_mom_change']  = df['cpi_inflation'].pct_change(30)   # Month over month
    df['rate_3m_change']  = df['fed_funds_rate'].diff(90)        # 3-month change in rates
    df['unemp_3m_change'] = df['unemployment'].diff(90)          # 3-month change in unemployment

    # Step 4: Recession indicator 
    # Inverted yield curve (2yr > 10yr) historically predicts recession
    df['yield_inverted'] = (df['yield_curve'] < 0).astype(int)

    # Step 5: VIX regime 
    # VIX > 30 = high fear/volatility regime
    # VIX < 20 = calm market regime
    df['vix_high_regime'] = (df['vix'] > 30).astype(int)
    df['vix_low_regime']  = (df['vix'] < 20).astype(int)

    # Step 6: Drop NaN from engineered features 
    df = df.dropna()

    print(f'\nClean shape: {df.shape}')
    print(f'Missing after cleaning: {df.isnull().sum().sum()}')

    save_dir = PROCESSED_DIR / 'features'
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / 'macro_clean.csv')
    
    print('Saved to data/processed/features/macro_clean.csv')
    return df

if __name__ == '__main__':
    macro = clean_macro_data()
    print('\nFinal columns:')
    print(list(macro.columns))
    print(macro.tail(3))