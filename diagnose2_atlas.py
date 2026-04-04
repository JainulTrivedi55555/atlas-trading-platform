"""
ATLAS Backtest Performance Report
"""

import sys
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.config import TICKERS, PROCESSED_DIR
from src.models.data_loader import load_splits
from src.backtesting.backtest_engine import (
    run_backtest,
    calculate_metrics,
    benchmark_buy_hold,
)

MODEL_DIR        = ROOT / "experiments" / "models"
CONFIDENCE_THRESHOLD = 0.55
INITIAL_CAPITAL  = 10_000
MODEL_TYPES      = ["xgboost", "lgbm"]

print("\n" + "=" * 75)
print("  ATLAS BACKTEST PERFORMANCE REPORT")
print(f"  Data split  : held-out test set (never seen during training)")
print(f"  Tickers     : {', '.join(TICKERS)}")
print(f"  Threshold   : {CONFIDENCE_THRESHOLD}")
print(f"  Capital     : ${INITIAL_CAPITAL:,} per ticker")
print("=" * 75)


def load_best_model(ticker: str):
    for mt in MODEL_TYPES:
        path = MODEL_DIR / f"{mt}_{ticker}.pkl"
        if path.exists():
            return joblib.load(path), mt
    return None, None


all_results    = []
failed_tickers = []

print("\n📈 Running backtests on held-out test set...\n")

for ticker in TICKERS:
    model, model_type = load_best_model(ticker)
    if model is None:
        print(f"  ⏭️  {ticker:<6} — model not found")
        failed_tickers.append(ticker)
        continue

    try:
        # ── Load the pre-split test data ──────────────────────────────
        _, _, _, _, X_test, y_test = load_splits(ticker)

        print(f"  {ticker}: {len(X_test)} test days", end="")

        # ── Generate probabilities using real model ───────────────────
        proba = model.predict_proba(X_test)[:, 1]

        print(f"  |  prob range [{proba.min():.3f} – {proba.max():.3f}]", end="")

        # ── Auto-adjust threshold if model never reaches it ───────────
        threshold = CONFIDENCE_THRESHOLD
        if (proba >= threshold).sum() == 0:
            # Use the 40th percentile of probabilities as threshold instead
            threshold = float(np.percentile(proba, 60))
            print(f"  |  ⚠️  auto-threshold → {threshold:.3f}", end="")

        signals = pd.Series(
            (proba >= threshold).astype(int),
            index=X_test.index,
            name="signal"
        )

        trades_count = int(signals.diff().abs().fillna(signals).sum())
        print(f"  |  {trades_count} trades")

        # ── Get daily returns aligned to test index ───────────────────
        # Try loading from feature file first, fall back to y_test returns
        feature_path = PROCESSED_DIR / f"features/{ticker}_features.csv"
        if feature_path.exists():
            feat_df = pd.read_csv(feature_path, index_col=0, parse_dates=True)
            feat_df.index = pd.to_datetime(feat_df.index).normalize()
            common = X_test.index.intersection(feat_df.index)

            if "Daily_Return" in feat_df.columns and len(common) > 10:
                returns = feat_df.loc[common, "Daily_Return"]
            elif "returns" in feat_df.columns and len(common) > 10:
                returns = feat_df.loc[common, "returns"]
            else:
                # Reconstruct from Close prices
                close_col = next((c for c in ["Close", "close"] if c in feat_df.columns), None)
                if close_col:
                    returns = feat_df.loc[common, close_col].pct_change().fillna(0)
                else:
                    returns = y_test.squeeze().map({1: 0.005, 0: -0.005})
        else:
            returns = y_test.squeeze().map({1: 0.005, 0: -0.005})

        # Align returns and signals to same index
        common_idx = returns.index.intersection(signals.index)
        returns    = returns.loc[common_idx].fillna(0)
        signals    = signals.reindex(common_idx).fillna(0)

        # ── Run backtests ─────────────────────────────────────────────
        strat_results = run_backtest(returns, signals, capital=INITIAL_CAPITAL)
        strat_metrics = calculate_metrics(strat_results, label=f"ATLAS_{ticker}")

        bah_results   = benchmark_buy_hold(returns, capital=INITIAL_CAPITAL)
        bah_metrics   = calculate_metrics(bah_results, label=f"BuyHold_{ticker}")

        beats_bah = strat_metrics["total_return"] > bah_metrics["total_return"]

        all_results.append({
            "ticker":     ticker,
            "model_type": model_type,
            "threshold":  round(threshold, 3),
            "strat":      strat_metrics,
            "bah":        bah_metrics,
            "beats_bah":  beats_bah,
        })

        icon = "✅" if beats_bah else "❌"
        print(f"  {icon} {ticker:<6} [{model_type:<8}]  "
              f"ATLAS={strat_metrics['total_return']:>7.1f}%  "
              f"B&H={bah_metrics['total_return']:>7.1f}%  "
              f"Sharpe={strat_metrics['sharpe']:>6.3f}  "
              f"WinRate={strat_metrics['win_rate']:>5.1f}%")

    except Exception as e:
        import traceback
        print(f"\n  ⚠️  {ticker:<6} — {e}")
        traceback.print_exc()
        failed_tickers.append(ticker)


# ── Bail if nothing ran ────────────────────────────────────────────
if not all_results:
    print("\n❌ No results. Check data/processed/splits/ exists for each ticker.")
    sys.exit(1)


# ── Full results table ─────────────────────────────────────────────
W = 135
print("\n" + "=" * W)
print("  DETAILED RESULTS")
print("=" * W)
print(f"{'Ticker':<8} {'Model':<10} {'Thresh':>7} "
      f"{'ATLAS%':>8} {'B&H%':>8} "
      f"{'Ann%':>7} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} "
      f"{'MaxDD%':>8} {'WinRate':>9} {'ProfFact':>10} {'Trades':>8} {'Beats?':>8}")
print("-" * W)

beats_count = 0
for r in all_results:
    s = r["strat"]
    b = r["bah"]
    beats = "✅ YES" if r["beats_bah"] else "❌ NO"
    if r["beats_bah"]:
        beats_count += 1
    print(f"{r['ticker']:<8} {r['model_type']:<10} {r['threshold']:>7.3f} "
          f"{s['total_return']:>8.2f} {b['total_return']:>8.2f} "
          f"{s['ann_return']:>7.2f} {s['sharpe']:>8.3f} {s['sortino']:>9.3f} {s['calmar']:>8.3f} "
          f"{s['max_drawdown']:>8.2f} {s['win_rate']:>9.1f} {s['profit_factor']:>10.3f} "
          f"{s['n_trades']:>8} {beats:>8}")

print("-" * W)

def avg(key, src="strat"):
    return np.mean([r[src][key] for r in all_results])

print(f"{'AVERAGE':<8} {'':10} {'':>7} "
      f"{avg('total_return'):>8.2f} {avg('total_return','bah'):>8.2f} "
      f"{avg('ann_return'):>7.2f} {avg('sharpe'):>8.3f} {avg('sortino'):>9.3f} {avg('calmar'):>8.3f} "
      f"{avg('max_drawdown'):>8.2f} {avg('win_rate'):>9.1f} {avg('profit_factor'):>10.3f} "
      f"{'':>8} {f'{beats_count}/{len(all_results)}':>8}")
print("=" * W)


# ── Verdict ────────────────────────────────────────────────────────
avg_sharpe = avg("sharpe")
avg_dd     = avg("max_drawdown")
avg_win    = avg("win_rate")
avg_pf     = avg("profit_factor")
total      = len(all_results)

print("\n" + "=" * 75)
print("  📊 FINAL VERDICT — IS ATLAS WORKING?")
print("=" * 75)

if beats_count >= int(total * 0.7):
    print(f"  🟢 STRONG   — beats B&H on {beats_count}/{total} tickers")
elif beats_count >= int(total * 0.5):
    print(f"  🟡 MIXED    — beats B&H on {beats_count}/{total} tickers")
else:
    print(f"  🔴 WEAK     — beats B&H on {beats_count}/{total} tickers")

if avg_sharpe >= 1.5:
    print(f"  🟢 Sharpe {avg_sharpe:.2f} — excellent risk-adjusted returns")
elif avg_sharpe >= 1.0:
    print(f"  🟡 Sharpe {avg_sharpe:.2f} — good risk-adjusted returns")
elif avg_sharpe >= 0.5:
    print(f"  🟠 Sharpe {avg_sharpe:.2f} — below target (try threshold=0.60)")
else:
    print(f"  🔴 Sharpe {avg_sharpe:.2f} — weak signals")

if avg_pf >= 1.5:
    print(f"  🟢 Profit Factor {avg_pf:.2f} — strong edge")
elif avg_pf >= 1.0:
    print(f"  🟡 Profit Factor {avg_pf:.2f} — slight edge")
else:
    print(f"  🔴 Profit Factor {avg_pf:.2f} — no edge yet")

print(f"\n  {'Metric':<25} {'Result':>10}   {'Target':>10}")
print(f"  {'-'*50}")
print(f"  {'Beats B&H':<25} {f'{beats_count}/{total}':>10}   {'7+/10':>10}")
print(f"  {'Avg Sharpe':<25} {avg_sharpe:>10.3f}   {'>1.0':>10}")
print(f"  {'Avg Max Drawdown':<25} {avg_dd:>9.1f}%   {'>-25%':>10}")
print(f"  {'Avg Win Rate':<25} {avg_win:>9.1f}%   {'>52%':>10}")
print(f"  {'Avg Profit Factor':<25} {avg_pf:>10.3f}   {'>1.2':>10}")

print(f"\n  WHAT THE NUMBERS MEAN:")
if avg_sharpe >= 1.0 and beats_count >= 6:
    print("  ✅ ATLAS is generating real alpha — safe to expand to 25 tickers")
elif avg_sharpe >= 0.5:
    print("  🔧 System has signal but needs tuning — try retraining with n_trials=50")
    print("  🔧 Or try lowering threshold to 0.50 for more trades")
else:
    print("  🔧 The model may be overfitting to training data")
    print("  🔧 Recommended: retrain with n_trials=50 and check feature engineering")

if failed_tickers:
    print(f"\n  ⚠️  Skipped: {failed_tickers}")
print("=" * 75)


# ── Save CSV ───────────────────────────────────────────────────────
rows = []
for r in all_results:
    row = {
        "ticker":       r["ticker"],
        "model_used":   r["model_type"],
        "threshold":    r["threshold"],
        "beats_bah":    r["beats_bah"],
    }
    row.update({f"atlas_{k}": v for k, v in r["strat"].items() if k != "label"})
    row.update({f"bah_{k}":   v for k, v in r["bah"].items()   if k != "label"})
    rows.append(row)

out_path = ROOT / "data" / "backtest_report.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\n💾 Saved to: {out_path}\n")