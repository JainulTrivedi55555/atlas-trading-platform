import sys
sys.path.append('.')
import joblib
import pandas as pd
from src.models.data_loader import load_splits

tickers = ['AAPL','MSFT','GOOGL','NVDA','TSLA']

print('Direct model verification')
print('='*55)
print(f'{"Ticker":<8} {"Prob Up":<12} {"Prob Down":<12} {"Signal"}')
print('='*55)

for ticker in tickers:
    model = joblib.load(
        f'experiments/models/xgboost_{ticker}.pkl'
    )
    X_train,_,_,_,X_test,_ = load_splits(ticker)
    X_test.index = pd.to_datetime(
        X_test.index
    ).normalize()

    try:
        feat_cols = model.get_booster().feature_names
    except:
        feat_cols = list(X_train.columns)

    for col in feat_cols:
        if col not in X_test.columns:
            X_test[col] = 0

    latest   = X_test[feat_cols].iloc[[-1]].ffill().fillna(0)
    proba    = model.predict_proba(latest)[0]
    prob_up  = round(proba[1] * 100, 1)
    prob_down = round(proba[0] * 100, 1)
    signal   = 'BULLISH' if prob_up >= 50 else 'BEARISH'

    print(f'{ticker:<8} {prob_up:<12} {prob_down:<12} {signal}')

print('='*55)
print('These numbers must match Streamlit exactly')