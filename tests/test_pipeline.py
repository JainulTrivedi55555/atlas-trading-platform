"""
ATLAS  — Integration Tests
Tests live data pipeline end-to-end.
"""
import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.live_pipeline import (
    fetch_ohlcv, build_live_features, validate_feature_alignment, TICKERS
)
from src.data.live_cache import (
    init_db, save_features, load_features, get_pipeline_status
)
def test_fetch_returns_dataframe():
    """Test that yfinance returns valid OHLCV data."""
    df = fetch_ohlcv('AAPL')
    assert df is not None, 'fetch_ohlcv returned None for AAPL'
    assert not df.empty, 'Returned DataFrame is empty'
    assert 'close' in df.columns, 'Missing close column'
    assert len(df) >= 60, f'Too few rows: {len(df)}'
    print(f'fetch test passed: {len(df)} rows, last={df.index[-1].date()}')
def test_features_shape_is_43():
    """Test that feature engineering produces exactly 43 features."""
    df_raw  = fetch_ohlcv('MSFT')
    df_feat = build_live_features(df_raw, 'MSFT')
    assert df_feat is not None, 'build_live_features returned None'
    assert df_feat.shape == (1, 43), f'Wrong shape: {df_feat.shape}, expected (1,43)'
    print(f'feature shape test passed: {df_feat.shape}')
def test_validation_passes():
    """Test that valid features pass the validator."""
    df_raw  = fetch_ohlcv('NVDA')
    df_feat = build_live_features(df_raw, 'NVDA')
    result  = validate_feature_alignment(df_feat)
    assert result is True, 'Validation failed for NVDA'
    print('validation test passed')

def test_cache_save_and_load():
    """Test SQLite cache write → read round-trip."""
    init_db()
    df_raw  = fetch_ohlcv('GOOGL')
    df_feat = build_live_features(df_raw, 'GOOGL')
    as_of   = df_feat.index[0].date()
    # Save to cache
    save_features('GOOGL', df_feat, as_of)
    # Load back
    cached = load_features('GOOGL')
    assert cached is not None, 'load_features returned None after save'
    assert cached['features'].shape == (1, 43), 'Cached features wrong shape'
    assert cached['as_of_date'] == str(as_of), 'as_of_date mismatch'
    print(f'cache round-trip test passed: as_of={as_of}')
def test_pipeline_status_returns_all_tickers():
    """Test that get_pipeline_status returns entries for all 10 tickers."""
    statuses = get_pipeline_status()
    assert len(statuses) == 10, f'Expected 10 statuses, got {len(statuses)}'
    tickers_returned = [s['ticker'] for s in statuses]
    for t in TICKERS:
        assert t in tickers_returned, f'{t} missing from pipeline status'
    print('pipeline status test passed')
if __name__ == '__main__':
    print('Running ATLAS Phase 11 Integration Tests...')
    test_fetch_returns_dataframe()
    test_features_shape_is_43()
    test_validation_passes()
    test_cache_save_and_load()
    test_pipeline_status_returns_all_tickers()
    print()
    print('=' * 50)
    print('ALL PHASE 11 TESTS PASSED ')
    print('=' * 50)