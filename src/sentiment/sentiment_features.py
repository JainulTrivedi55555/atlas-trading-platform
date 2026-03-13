import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR, TICKERS


def build_sentiment_features(ticker: str) -> pd.DataFrame:
    """
    Build sentiment proxy features from price action.
    Used because historical news data is unavailable for 2015-2025.
    Price momentum is a validated sentiment proxy (Zhang et al. 2011).
    """
    price_df = pd.read_csv(
        PROCESSED_DIR / f'features/{ticker}_features.csv',
        index_col=0, parse_dates=True
    )

    df = pd.DataFrame(index=price_df.index)
    returns = price_df['Close'].pct_change().fillna(0)

    # Rolling sentiment proxies from returns
    df['Sentiment_3d']  = returns.rolling(3).mean().fillna(0)
    df['Sentiment_7d']  = returns.rolling(7).mean().fillna(0)
    df['Sentiment_14d'] = returns.rolling(14).mean().fillna(0)
    df['Sentiment_30d'] = returns.rolling(30).mean().fillna(0)

    # Momentum — is sentiment improving or worsening?
    df['Sentiment_Momentum'] = df['Sentiment_7d'] - df['Sentiment_14d']

    # Volatility proxy — high vol = uncertain sentiment
    df['Sentiment_Vol'] = returns.rolling(7).std().fillna(0)

    # Volume proxy — high volume = high news activity
    if 'Volume' in price_df.columns:
        vol = price_df['Volume']
        df['News_Count_7d'] = (
            (vol / vol.rolling(30).mean())
            .fillna(1).rolling(7).mean().fillna(1)
        )
    else:
        df['News_Count_7d'] = 1.0

    # Price divergence — short vs long term momentum
    df['Sentiment_Price_Divergence'] = (
        df['Sentiment_7d'] - df['Sentiment_30d']
    )

    # Positive and negative ratio proxies
    df['Positive_Ratio_7d'] = (
        (returns > 0).rolling(7).mean().fillna(0.5)
    )
    df['Negative_Ratio_7d'] = (
        (returns < 0).rolling(7).mean().fillna(0.5)
    )

    # Regime — is overall trend positive?
    df['Sentiment_Regime'] = (df['Sentiment_30d'] > 0).astype(int)

    print(f'{ticker}: {len(df.columns)} sentiment proxy features built')
    return df


def add_sentiment_to_splits(ticker: str):
    """
    Add sentiment features to existing train/val/test splits.
    """
    from src.models.data_loader import load_splits

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(ticker)

    # Normalize all indexes
    X_train.index = pd.to_datetime(X_train.index).normalize()
    X_val.index   = pd.to_datetime(X_val.index).normalize()
    X_test.index  = pd.to_datetime(X_test.index).normalize()
    y_train.index = pd.to_datetime(y_train.index).normalize()
    y_val.index   = pd.to_datetime(y_val.index).normalize()
    y_test.index  = pd.to_datetime(y_test.index).normalize()

    # Build and join sentiment features
    sent_feat       = build_sentiment_features(ticker)
    sent_feat.index = pd.to_datetime(sent_feat.index).normalize()

    X_train_s = X_train.join(sent_feat, how='left').fillna(0)
    X_val_s   = X_val.join(sent_feat,   how='left').fillna(0)
    X_test_s  = X_test.join(sent_feat,  how='left').fillna(0)

    print(f'{ticker}: {X_train.shape[1]} → {X_train_s.shape[1]} features')
    return X_train_s, y_train, X_val_s, y_val, X_test_s, y_test


if __name__ == '__main__':
    X_tr, y_tr, X_v, y_v, X_te, y_te = add_sentiment_to_splits('AAPL')
    print(f'Feature count: {X_tr.shape[1]}')