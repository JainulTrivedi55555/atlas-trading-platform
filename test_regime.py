import requests, json

# Test 1: Single ticker regime signal
r = requests.get("http://localhost:8000/signal/regime/AAPL")
print("Single ticker:")
print(json.dumps(r.json(), indent=2))

# Test 2: All tickers regime signals  
r2 = requests.get("http://localhost:8000/signals/regime/all")
data = r2.json()
print(f"\nAll tickers ({len(data)} results):")
for s in data:
    print(f"  {s['ticker']:6} | {s['signal']:8} | regime: {s.get('regime','N/A'):8} | conf: {s.get('confidence',0):.4f}")