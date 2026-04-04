"""
ATLAS Ticker Expansion Script
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

# ── New tickers to add ────────────────────────────────────────────
NEW_TICKERS = [
    # Group 2 — Large Cap (different sectors)
    "NFLX",   # Entertainment
    "ORCL",   # Enterprise Software
    "AMD",    # Semiconductors
    "CRM",    # SaaS/Cloud
    "UBER",   # Consumer Tech
    "WMT",    # Retail (defensive)
    "JNJ",    # Healthcare (defensive)
    "XOM",    # Energy
    "LLY",    # Pharma
    "V",      # Payments
    # Group 3 — ETFs (market context signals)
    "SPY",    # S&P 500 benchmark
    "QQQ",    # Nasdaq 100
    "IWM",    # Small Cap
    "GLD",    # Gold (safe haven)
    "TLT",    # Long-term bonds (risk-off)
]

EXISTING_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "GS", "BAC", "NVDA", "TSLA"
]

ALL_TICKERS = EXISTING_TICKERS + NEW_TICKERS

# ── Import config values ──────────────────────────────────────────
from src.utils.config import (
    RAW_DIR, PROCESSED_DIR,
    TRAIN_START, TRAIN_END,
    VAL_START, VAL_END,
    TEST_START, TEST_END,
)

# ── Import pipeline functions ─────────────────────────────────────
from src.data_pipeline.price_collector import download_price_data
from src.data_pipeline.price_cleaner import clean_price_data
from src.features.technical_indicators import add_technical_indicators, add_target_variables
from src.features.data_splitter import split_ticker_data, normalize_features, save_splits
from src.models.xgboost_model import train_xgboost
from src.models.lgbm_model import train_lgbm

# ── Tracking ──────────────────────────────────────────────────────
succeeded = []
failed    = []


def print_header(msg):
    print(f"\n{'='*65}")
    print(f"  {msg}")
    print(f"{'='*65}")


def print_step(step, ticker, msg=""):
    print(f"  [{step}] {ticker:<6} {msg}")


# ══════════════════════════════════════════════════════════════════
print_header("ATLAS TICKER EXPANSION — 10 → 25 TICKERS")
print(f"  Adding: {', '.join(NEW_TICKERS)}")
print(f"  Steps per ticker: download → clean → features → splits → train")
print(f"  Estimated time: ~90 minutes")
# ══════════════════════════════════════════════════════════════════


# ── STEP 1: Download raw price data ──────────────────────────────
print_header("STEP 1/6 — Downloading price data")

download_price_data(NEW_TICKERS, TRAIN_START, TEST_END)

# Verify downloads
raw_dir = RAW_DIR / "price"
downloaded = []
for ticker in NEW_TICKERS:
    fp = raw_dir / f"{ticker}_daily.csv"
    if fp.exists():
        df = pd.read_csv(fp, index_col=0)
        print_step("✅", ticker, f"{len(df)} rows downloaded")
        downloaded.append(ticker)
    else:
        print_step("❌", ticker, "FAILED — no file created")
        failed.append(ticker)

to_process = [t for t in NEW_TICKERS if t in downloaded]
print(f"\n  Downloaded: {len(downloaded)}/{len(NEW_TICKERS)} tickers")


# ── STEP 2: Clean price data ──────────────────────────────────────
print_header("STEP 2/6 — Cleaning price data")

cleaned = []
for ticker in to_process:
    try:
        df = clean_price_data(ticker)
        save_path = PROCESSED_DIR / f"features/{ticker}_clean.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        print_step("✅", ticker, f"{len(df)} rows cleaned → {save_path.name}")
        cleaned.append(ticker)
    except Exception as e:
        print_step("❌", ticker, f"clean failed: {e}")
        failed.append(ticker)

to_process = [t for t in to_process if t in cleaned]


# ── STEP 3: Build technical indicator features ────────────────────
print_header("STEP 3/6 — Building technical indicator features")

featured = []
for ticker in to_process:
    try:
        clean_path = PROCESSED_DIR / f"features/{ticker}_clean.csv"
        df = pd.read_csv(clean_path, index_col=0, parse_dates=True)

        df = add_technical_indicators(df)
        df = add_target_variables(df)
        df = df.dropna()

        feat_path = PROCESSED_DIR / f"features/{ticker}_features.csv"
        df.to_csv(feat_path)
        print_step("✅", ticker, f"{len(df)} rows, {len(df.columns)} features")
        featured.append(ticker)
    except Exception as e:
        print_step("❌", ticker, f"features failed: {e}")
        failed.append(ticker)

to_process = [t for t in to_process if t in featured]


# ── STEP 4: Create train/val/test splits ──────────────────────────
print_header("STEP 4/6 — Creating train/val/test splits")

split_done = []
for ticker in to_process:
    try:
        X_tr, y_tr, X_v, y_v, X_te, y_te, feat_cols = split_ticker_data(ticker)

        # Check we have enough data in each split
        if len(X_tr) < 100:
            print_step("⚠️ ", ticker, f"train set too small ({len(X_tr)} rows) — skipping")
            failed.append(ticker)
            continue
        if len(X_te) < 30:
            print_step("⚠️ ", ticker, f"test set too small ({len(X_te)} rows) — skipping")
            failed.append(ticker)
            continue

        X_tr_s, X_v_s, X_te_s, scaler = normalize_features(X_tr, X_v, X_te)
        save_splits(ticker, X_tr_s, y_tr, X_v_s, y_v, X_te_s, y_te)

        # Save ticker-specific scaler
        scaler_path = PROCESSED_DIR / f"splits/{ticker}/scaler.pkl"
        joblib.dump(scaler, scaler_path)

        print_step("✅", ticker,
                   f"train={len(X_tr)} val={len(X_v)} test={len(X_te)}")
        split_done.append(ticker)
    except Exception as e:
        print_step("❌", ticker, f"split failed: {e}")
        failed.append(ticker)

to_process = [t for t in to_process if t in split_done]


# ── STEP 5: Train XGBoost models ─────────────────────────────────
print_header("STEP 5/6 — Training XGBoost models (n_trials=30)")
print("  (30 trials balances speed vs quality for new tickers)")

xgb_done = []
for ticker in to_process:
    try:
        print(f"\n  Training XGBoost for {ticker}...")
        train_xgboost(ticker, n_trials=30)
        model_path = ROOT / "experiments" / "models" / f"xgboost_{ticker}.pkl"
        if model_path.exists():
            print_step("✅", ticker, "XGBoost model saved")
            xgb_done.append(ticker)
        else:
            print_step("❌", ticker, "model file not found after training")
            failed.append(ticker)
    except Exception as e:
        print_step("❌", ticker, f"XGBoost training failed: {e}")
        failed.append(ticker)


# ── STEP 6: Train LightGBM models ────────────────────────────────
print_header("STEP 6/6 — Training LightGBM models (n_trials=30)")

lgbm_done = []
for ticker in to_process:
    try:
        print(f"\n  Training LightGBM for {ticker}...")
        train_lgbm(ticker, n_trials=30)
        model_path = ROOT / "experiments" / "models" / f"lgbm_{ticker}.pkl"
        if model_path.exists():
            print_step("✅", ticker, "LightGBM model saved")
            lgbm_done.append(ticker)
        else:
            print_step("⚠️ ", ticker, "LightGBM model file not found (XGBoost still usable)")
    except Exception as e:
        print_step("⚠️ ", ticker, f"LightGBM failed: {e} (XGBoost still usable)")


# ── STEP 7: Update config.py ──────────────────────────────────────
print_header("STEP 7/7 — Updating config.py with new ticker list")

# Only add tickers that fully succeeded (have splits + at least XGBoost model)
fully_succeeded = [
    t for t in NEW_TICKERS
    if t in split_done and
    (ROOT / "experiments" / "models" / f"xgboost_{t}.pkl").exists()
]

new_full_list = EXISTING_TICKERS + fully_succeeded

config_path = ROOT / "src" / "utils" / "config.py"

try:
    config_text = config_path.read_text()

    # Replace the TICKERS line
    old_line = next(
        line for line in config_text.splitlines()
        if line.strip().startswith("TICKERS")
    )
    new_line = (
        "TICKERS = [\n"
        "    # Group 1 — Original large cap tech + finance\n"
        "    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',\n"
        "    'JPM', 'GS', 'BAC', 'NVDA', 'TSLA',\n"
        "    # Group 2 — Additional large caps (different sectors)\n"
    )
    for t in [t for t in fully_succeeded if t in [
        "NFLX","ORCL","AMD","CRM","UBER","WMT","JNJ","XOM","LLY","V"
    ]]:
        new_line += f"    '{t}',\n"
    new_line += (
        "    # Group 3 — ETFs (market context signals)\n"
    )
    for t in [t for t in fully_succeeded if t in [
        "SPY","QQQ","IWM","GLD","TLT"
    ]]:
        new_line += f"    '{t}',\n"
    new_line += "]"

    config_text = config_text.replace(old_line, new_line)
    config_path.write_text(config_text)

    print(f"  ✅ config.py updated")
    print(f"  New TICKERS list ({len(new_full_list)} tickers):")
    print(f"  {new_full_list}")

except Exception as e:
    print(f"  ⚠️  Could not auto-update config.py: {e}")
    print(f"  Manually update TICKERS in src/utils/config.py to:")
    print(f"  {new_full_list}")


# ── Final summary ─────────────────────────────────────────────────
print_header("EXPANSION COMPLETE — SUMMARY")

failed_unique = list(set(failed))
print(f"\n  ✅ Successfully added: {len(fully_succeeded)} new tickers")
if fully_succeeded:
    print(f"     {fully_succeeded}")

print(f"\n  📊 Total tickers now: {len(new_full_list)}")
print(f"     {new_full_list}")

if failed_unique:
    print(f"\n  ❌ Failed tickers: {failed_unique}")
    print(f"     These were NOT added to config.py")
    print(f"     Re-run this script to retry them")

print(f"""
  NEXT STEPS:
  1. Run the backtest to see all 25 tickers:
       python backtest_report.py

  2. Check the Streamlit dashboard:
       streamlit run streamlit_app.py

  3. If any tickers failed, just re-run:
       python expand_tickers.py
""")
