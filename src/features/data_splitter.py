import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import (
    TICKERS, PROCESSED_DIR,
    TRAIN_START, TRAIN_END,
    VAL_START, VAL_END,
    TEST_START, TEST_END
)

# Columns that are labels — never goes in X
TARGET_COLS = ['Target_Direction', 'Target_Return']

# Columns to exclude from features
EXCLUDE_COLS = TARGET_COLS + ['Dividends', 'Stock Splits']


def split_ticker_data(ticker: str):
    """
    Load feature CSV for one ticker and return
    properly time-split X/y arrays for train, val, test.
    """
    fp = PROCESSED_DIR / f'features/{ticker}_features.csv'
    df = pd.read_csv(fp, index_col=0, parse_dates=True)

    df = df.sort_index()

    # Split by date — NEVER randomly
    train = df.loc[TRAIN_START:TRAIN_END]
    val   = df.loc[VAL_START:VAL_END]
    test  = df.loc[TEST_START:TEST_END]

    print(f'  {ticker} splits:')
    print(f'    Train: {len(train)} rows ({TRAIN_START} to {TRAIN_END})')
    print(f'    Val:   {len(val)} rows ({VAL_START} to {VAL_END})')
    print(f'    Test:  {len(test)} rows ({TEST_START} to {TEST_END})')

    # Feature columns = everything except targets and excluded cols
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Separate features (X) and labels (y)
    X_train = train[feature_cols]
    y_train = train['Target_Direction']

    X_val   = val[feature_cols]
    y_val   = val['Target_Direction']

    X_test  = test[feature_cols]
    y_test  = test['Target_Direction']

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def normalize_features(X_train, X_val, X_test):
    """
    Normalizing features using StandardScaler.
    CRITICAL: Fit ONLY on training data.
    Applying the same transformation to val and test.
    Never fit on val or test — that would be data leakage.
    """
    from sklearn.preprocessing import StandardScaler
    import joblib

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    # Transform val and test using TRAIN statistics
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    # Save scaler — needed at inference time
    save_dir = PROCESSED_DIR / 'splits'
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, save_dir / 'scaler.pkl')

    print('  Scaler saved to data/processed/splits/scaler.pkl')

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_splits(ticker, X_train, y_train, X_val, y_val, X_test, y_test):
    """Saving all splits to CSV for easy loading later."""

    save_dir = PROCESSED_DIR / f'splits/{ticker}'
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(save_dir / 'X_train.csv')
    y_train.to_csv(save_dir / 'y_train.csv')

    X_val.to_csv(save_dir / 'X_val.csv')
    y_val.to_csv(save_dir / 'y_val.csv')

    X_test.to_csv(save_dir / 'X_test.csv')
    y_test.to_csv(save_dir / 'y_test.csv')

    print(f'  Splits saved to data/processed/splits/{ticker}/')


if __name__ == '__main__':

    # Run for all tickers
    for ticker in TICKERS:
        print(f'\nProcessing {ticker}...')

        X_tr, y_tr, X_v, y_v, X_te, y_te, feat_cols = split_ticker_data(ticker)

        X_tr_s, X_v_s, X_te_s, scaler = normalize_features(X_tr, X_v, X_te)

        save_splits(ticker, X_tr_s, y_tr, X_v_s, y_v, X_te_s, y_te)

    print('\nAll splits created and saved!')