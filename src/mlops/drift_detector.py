import sys, warnings, os
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.models.data_loader import load_splits
from src.utils.config import TICKERS

os.makedirs('docs/drift', exist_ok=True)


def run_drift_report(ticker: str = 'AAPL') -> str:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    X_train,_,_,_,X_test,_ = load_splits(ticker)
    print(f'{ticker} — Train:{len(X_train)} Test:{len(X_test)}')

    feature_cols = list(X_train.columns[:43])
    ref  = X_train[feature_cols].copy()
    curr = X_test[feature_cols].copy()

    # New evidently 0.5+ API
    report = Report([DataDriftPreset()])
    my_eval = report.run(ref, curr)

    out_path = f'docs/drift/{ticker}_drift_report.html'
    my_eval.save_html(out_path)

    print(f'Drift report saved: {out_path}')
    return out_path


if __name__ == '__main__':
    print('Running drift detection for AAPL...')
    path = run_drift_report('AAPL')
    print(f'Open in browser: {path}')