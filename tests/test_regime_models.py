"""
Integration Tests
Tests regime detector, trainer, and predictor end-to-end.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.regime_detector import (
    fit_regime_detector, label_regimes, detect_current_regime
)
from src.models.regime_predictor import get_regime_signal

def test_regime_detector_fits():
    """HMM fits and saves successfully."""
    hmm = fit_regime_detector('AAPL')
    assert hmm is not None
    assert Path('experiments/models/hmm_AAPL.pkl').exists()
    print("test_regime_detector_fits: PASSED")

def test_regime_labels_distribution():
    """All 3 regimes appear in labels."""
    regimes = label_regimes('AAPL')
    values = set(regimes.unique())
    assert 'bull'    in values, f"'bull' not in {values}"
    assert 'bear'    in values, f"'bear' not in {values}"
    assert 'highvol' in values, f"'highvol' not in {values}"
    print(f"test_regime_labels_distribution: PASSED — {dict(regimes.value_counts())}")

def test_current_regime_returns_valid():
    """detect_current_regime returns a valid regime name."""
    regime = detect_current_regime('AAPL')
    assert regime in ('bull', 'bear', 'highvol'), f"Invalid regime: {regime}"
    print(f"test_current_regime_returns_valid: PASSED — current regime: {regime}")

def test_regime_signal_structure():
    """get_regime_signal returns correct dict structure."""
    result = get_regime_signal('AAPL', 'xgboost')
    required = ['ticker', 'signal', 'confidence', 'prob_up',
                'prob_down', 'regime', 'model_used', 'as_of_date']
    
    for key in required:
        assert key in result, f"Missing key: {key}"
        
    assert result['signal'] in ('BULLISH', 'BEARISH')
    assert result['regime'] in ('bull', 'bear', 'highvol')
    assert 0 <= result['prob_up'] <= 1
    
    print(f"test_regime_signal_structure: PASSED — "
          f"signal={result['signal']}, regime={result['regime']}, "
          f"conf={result['confidence']:.4f}")

if __name__ == '__main__':
    print("Running ATLAS Phase 12 Integration Tests...")
    print("=" * 50)
    test_regime_detector_fits()
    test_regime_labels_distribution()
    test_current_regime_returns_valid()
    test_regime_signal_structure()
    print()
    print("=" * 50)
    print("ALL PHASE 12 TESTS PASSED")
    print("=" * 50)