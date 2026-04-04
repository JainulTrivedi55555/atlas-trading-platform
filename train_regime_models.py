"""
Train All Regime Models
Run once from project root:
    conda activate fintech
    python train_regime_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import TICKERS
from src.models.regime_trainer import train_regime_models

if __name__ == '__main__':
    print("ATLAS Phase 12 — Training Regime-Specific Models")
    print("=" * 55)
    print(f"Tickers: {TICKERS}")
    print("This will take 5-15 minutes depending on your CPU.")
    print("=" * 55)
    
    all_results = {}
    failed = []
    
    for ticker in TICKERS:
        try:
            results = train_regime_models(ticker)
            all_results[ticker] = results
        except Exception as e:
            print(f"ERROR training {ticker}: {e}")
            failed.append(ticker)
            
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE — Summary")
    print("=" * 55)
    
    for ticker, results in all_results.items():
        print(f"\n{ticker}:")
        for regime, r in results.items():
            if not r.get('skipped'):
                print(f"  {regime:8s} XGB: {r['xgb_auc']:.4f} "
                      f"LGBM: {r['lgbm_auc']:.4f} "
                      f"(n={r['n_train']})")
            else:
                print(f"  {regime:8s} SKIPPED (n={r['n_train']} rows)")
                
    if failed:
        print(f"\nFAILED tickers: {failed}")
    else:
        print("\nAll tickers trained successfully!")
        
    print("\nNext: run python -m uvicorn app.main:app --reload --port 8000")