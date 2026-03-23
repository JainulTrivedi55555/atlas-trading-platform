import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR

def build_macro_features(ticker: str) -> pd.DataFrame:
    """
    Build macro features aligned to price dates for one ticker.
    Uses macro_clean.csv created in Phase 2.
    Returns DataFrame with macro features indexed by trading days.
    """
    # Load macro clean data (Phase 2 output)
    macro_path = PROCESSED_DIR / 'features' / 'macro_clean.csv'
    if not macro_path.exists():
        raise FileNotFoundError(
            f'macro_clean.csv not found. Run macro_cleaner.py first.'
        )
    
    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    macro.index = pd.to_datetime(macro.index).normalize() 

    # Load price data to align dates
    price_path = PROCESSED_DIR / f'features/{ticker}_features.csv'
    price_df   = pd.read_csv(price_path, index_col=0, parse_dates=True)
    price_df.index = pd.to_datetime(price_df.index).normalize()

    # Reindex macro to trading days (forward fill gaps)
    macro_aligned = macro.reindex(price_df.index, method='ffill')

    # Additional Macro Feature Engineering 
    # 1. Fed rate change momentum — is Fed tightening or easing?
    macro_aligned['Fed_Rate_Momentum'] = (
        macro_aligned['fed_funds_rate'].diff(30).fillna(0)
    )
    # 2. VIX momentum — is fear increasing or decreasing?
    macro_aligned['VIX_Momentum'] = (
        macro_aligned['vix'].diff(5).fillna(0)
    )
    # 3. VIX normalized — relative to its own history
    macro_aligned['VIX_Normalized'] = (
        (macro_aligned['vix'] - macro_aligned['vix'].rolling(252).mean()) /
        (macro_aligned['vix'].rolling(252).std() + 1e-8)
    ).fillna(0)

    # 4. Yield curve momentum — is curve steepening or flattening?
    macro_aligned['Yield_Curve_Momentum'] = (
        macro_aligned['yield_curve'].diff(30).fillna(0)
    )
    # 5. Real rate = nominal - inflation
    macro_aligned['Real_Rate'] = (
        macro_aligned['treasury_10yr'] - macro_aligned['cpi_inflation']
    ).fillna(0)
    # 6. Macro stress index
    # High VIX + inverted yield + high unemployment = stress
    macro_aligned['Macro_Stress'] = (
        macro_aligned['vix_high_regime'] +
        macro_aligned['yield_inverted'] +
        (macro_aligned['unemployment'] > 6).astype(int)
    )
    # 7. Rate environment: 0=low, 1=medium, 2=high
    macro_aligned['Rate_Regime'] = pd.cut(
        macro_aligned['fed_funds_rate'],
        bins=[-1, 1, 3, 20],
        labels=[0, 1, 2]
    ).astype(float).fillna(0)
    
    # 8. Inflation regime 
    macro_aligned['Inflation_High'] = (
        macro_aligned['cpi_inflation'] > macro_aligned['cpi_inflation']
        .rolling(252).mean()
    ).astype(int).fillna(0)

    # Select final feature columns
    macro_feature_cols = [
        # Raw macro indicators
        'fed_funds_rate', 'cpi_inflation', 'unemployment',
        'treasury_10yr', 'yield_curve', 'vix',
        # Engineered macro from Phase 2
        'yield_inverted', 'vix_high_regime', 'vix_low_regime',
        'cpi_mom_change', 'rate_3m_change',
        # New engineered features
        'Fed_Rate_Momentum', 'VIX_Momentum', 'VIX_Normalized',
        'Yield_Curve_Momentum', 'Real_Rate',
        'Macro_Stress', 'Rate_Regime', 'Inflation_High',
    ]

    # Keep only columns that exist
    available = [c for c in macro_feature_cols if c in macro_aligned.columns]
    result = macro_aligned[available].fillna(0)
    print(f'{ticker}: {len(available)} macro features aligned')
    print(f'  Macro date range: {result.index.min()} to {result.index.max()}')
    
    return result

def add_macro_to_splits(ticker: str):
    """
    Add macro features to existing train/val/test splits.
    Stacks on top of sentiment features from Phase 5.
    """
    from src.models.data_loader import load_splits
    from src.sentiment.sentiment_features import add_sentiment_to_splits
    
    # Load splits WITH sentiment features (Phase 5)
    X_train, y_train, X_val, y_val, X_test, y_test = add_sentiment_to_splits(ticker)

    # Normalize indexes — Phase 4 fix
    for df in [X_train, X_val, X_test]:
        df.index = pd.to_datetime(df.index).normalize()
    y_train.index = pd.to_datetime(y_train.index).normalize()
    y_val.index   = pd.to_datetime(y_val.index).normalize()
    y_test.index  = pd.to_datetime(y_test.index).normalize()

    # Build and join macro features
    macro_feat       = build_macro_features(ticker)
    macro_feat.index = pd.to_datetime(macro_feat.index).normalize()

    X_train_m = X_train.join(macro_feat, how='left').fillna(0)
    X_val_m   = X_val.join(macro_feat,   how='left').fillna(0)
    X_test_m  = X_test.join(macro_feat,  how='left').fillna(0)

    print(f'\n{ticker} feature expansion:')
    print(f'  Technical+Sentiment: {X_train.shape[1]} features')
    print(f'  After adding macro:  {X_train_m.shape[1]} features')
    print(f'  Macro features added: {X_train_m.shape[1]-X_train.shape[1]}')
    
    return X_train_m, y_train, X_val_m, y_val, X_test_m, y_test

if __name__ == '__main__':
    X_tr, y_tr, X_v, y_v, X_te, y_te = add_macro_to_splits('AAPL')
    print(f'\nTotal features: {X_tr.shape[1]}')
    macro_cols = [c for c in X_tr.columns if c not in ['Sentiment_3d']]
    print(f'Feature matrix shape: {X_tr.shape}')