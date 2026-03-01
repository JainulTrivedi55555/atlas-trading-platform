import pandas as pd
import numpy as np
import ta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import TICKERS, PROCESSED_DIR


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adding 20+ technical indicators to a clean price DataFrame.
    Input:  DataFrame with columns [Open, High, Low, Close, Volume]
    Output: Same DataFrame with indicator columns added
    """

    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']

    # MOMENTUM INDICATORS

    df['RSI_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta.trend.MACD(close)
    df['MACD']        = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist']   = macd.macd_diff()

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    df['Williams_R'] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()

    df['ROC_5']  = ta.momentum.ROCIndicator(close, window=5).roc()
    df['ROC_20'] = ta.momentum.ROCIndicator(close, window=20).roc()

    # TREND INDICATORS

    df['SMA_20']  = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df['SMA_50']  = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(close, window=200).sma_indicator()

    df['EMA_12'] = ta.trend.EMAIndicator(close, window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close, window=26).ema_indicator()

    df['Price_vs_SMA20']  = (close - df['SMA_20'])  / df['SMA_20']
    df['Price_vs_SMA50']  = (close - df['SMA_50'])  / df['SMA_50']
    df['Price_vs_SMA200'] = (close - df['SMA_200']) / df['SMA_200']

    df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    df['Death_Cross']  = (df['SMA_50'] < df['SMA_200']).astype(int)

    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['BB_Upper']  = bb.bollinger_hband()
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Lower']  = bb.bollinger_lband()
    df['BB_Width']  = bb.bollinger_wband()
    df['BB_Pct']    = bb.bollinger_pband()

    # VOLATILITY INDICATORS

    df['ATR_14'] = ta.volatility.AverageTrueRange(
        high, low, close, window=14
    ).average_true_range()

    df['Volatility_5d']  = df['Daily_Return'].rolling(5).std()  * np.sqrt(252)
    df['Volatility_20d'] = df['Daily_Return'].rolling(20).std() * np.sqrt(252)

    # VOLUME INDICATORS

    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        close, volume
    ).on_balance_volume()

    df['Volume_Ratio'] = volume / volume.rolling(20).mean()

    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(
        high, low, close, volume
    ).chaikin_money_flow()

    # PATTERN FEATURES

    df['Dist_52w_High'] = close / close.rolling(252).max() - 1
    df['Dist_52w_Low']  = close / close.rolling(252).min() - 1

    df['Return_5d']  = close.pct_change(5)
    df['Return_60d'] = close.pct_change(60)

    return df


def build_feature_matrix(tickers=None) -> dict:
    """Build feature matrix for all tickers."""

    if tickers is None:
        tickers = TICKERS

    save_dir = PROCESSED_DIR / 'features'
    save_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}

    for ticker in tickers:
        print(f'Building features for {ticker}...')

        fp = save_dir / f'{ticker}_clean.csv'
        df = pd.read_csv(fp, index_col=0, parse_dates=True)

        df = add_technical_indicators(df)
        df = add_target_variables(df)
        df = df.dropna()

        out_fp = save_dir / f'{ticker}_features.csv'
        df.to_csv(out_fp)

        all_data[ticker] = df

        print(f'  {ticker}: {len(df)} rows, {len(df.columns)} features')

    return all_data


# Add to src/features/technical_indicators.py
def add_target_variables(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Adding prediction target variables.
    IMPORTANT: These use FUTURE data — that is intentional for the label.
    We shift backwards so today's row knows what will happen in N days.
    Do NOT use these columns as features — only as labels.
    """
    # Future return over next N days
    df['Target_Return'] = df['Close'].pct_change(horizon).shift(-horizon)
    # Future direction — will price be higher in N days?
    df['Target_Direction'] = (df['Target_Return'] > 0).astype(int)
    # Remove last N rows — they have no valid future label
    df = df.iloc[:-horizon]
    # Check class balance — how many up vs down days?
    direction_counts = df['Target_Direction'].value_counts()
    up_pct = direction_counts.get(1, 0) / len(df) * 100
    print(f'  Up: {up_pct:.1f}%  Down: {100-up_pct:.1f}%')
    return df


if __name__ == '__main__':
    data = build_feature_matrix()

    sample = list(data.values())[0]
    print(f'\nFeature columns ({len(sample.columns)} total):')
    print(list(sample.columns))