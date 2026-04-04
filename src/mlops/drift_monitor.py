"""
Evidently AI Drift Monitor
Compares live market features against training data distribution.
Generates HTML report + returns drift score for scheduler alerts.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger("atlas.drift_monitor")

REPORTS_DIR = Path("reports/drift")
REFERENCE_DIR = Path("data/processed/features")

# Drift threshold — above this triggers a retrain alert
DRIFT_THRESHOLD = 0.30  # 30% of features drifted

# Key features to monitor (most predictive from Phase 4)
MONITOR_FEATURES = [
    "RSI_14", "MACD", "MACD_Signal", "BB_Pct", "BB_Width",
    "ATR_14", "Volatility_20d", "Volume_Ratio", "Price_vs_SMA20",
    "Price_vs_SMA50", "Price_vs_SMA200", "Return_5d", "Return_60d",
    "Stoch_K", "Williams_R", "CMF", "OBV",
]


def load_reference_data(ticker: str) -> pd.DataFrame:
    """
    Load training reference data for a ticker.
    Uses the processed CSV files created in Phase 2.
    """
    ref_path = REFERENCE_DIR / f"{ticker}_features.csv"

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference data not found: {ref_path}")

    df = pd.read_csv(ref_path, index_col=0, parse_dates=True)

    # Use training period only (2015-2022)
    df = df[df.index <= "2022-12-31"]

    return df


def load_live_data(ticker: str) -> pd.DataFrame:
    """
    Load recent live data for a ticker from the live cache.
    Uses last 60 trading days of live data.
    """
    try:
        from src.data.live_cache import load_features

        df = load_features(ticker)

        if df is None or len(df) < 20:
            raise ValueError(f"Insufficient live data for {ticker}")

        return df.tail(60)

    except Exception as e:
        logger.error(f"Could not load live data for {ticker}: {e}")
        return None


def compute_drift_scores(ref_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
    """
    Compute drift scores for each monitored feature.
    Uses Population Stability Index (PSI) for numerical features.

    PSI < 0.1 → No drift
    PSI 0.1-0.2 → Moderate drift
    PSI > 0.2 → Significant drift (retrain candidate)
    """

    drift_scores = {}
    drifted_features = []

    for feature in MONITOR_FEATURES:

        if feature not in ref_df.columns or feature not in live_df.columns:
            continue

        try:
            ref_vals = ref_df[feature].dropna().values
            live_vals = live_df[feature].dropna().values

            if len(ref_vals) < 10 or len(live_vals) < 10:
                continue

            # PSI calculation
            bins = np.percentile(ref_vals, np.linspace(0, 100, 11))
            bins = np.unique(bins)

            if len(bins) < 3:
                continue

            ref_hist = np.histogram(ref_vals, bins=bins)[0]
            live_hist = np.histogram(live_vals, bins=bins)[0]

            ref_pct = (ref_hist + 1e-6) / len(ref_vals)
            live_pct = (live_hist + 1e-6) / len(live_vals)

            psi = float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))

            drift_scores[feature] = round(psi, 4)

            if psi > 0.20:
                drifted_features.append(feature)

        except Exception as e:
            logger.debug(f"PSI failed for {feature}: {e}")
            continue

    n_monitored = len(drift_scores)
    n_drifted = len(drifted_features)

    drift_ratio = n_drifted / n_monitored if n_monitored > 0 else 0

    return {
        "drift_scores": drift_scores,
        "drifted_features": drifted_features,
        "n_monitored": n_monitored,
        "n_drifted": n_drifted,
        "drift_ratio": round(drift_ratio, 4),
        "retrain_alert": drift_ratio >= DRIFT_THRESHOLD,
    }


def generate_drift_report(ticker: str, report_date: date = None) -> dict:
    """
    Full drift pipeline: load data, compute PSI, save JSON summary.
    Also generates an Evidently HTML report if evidently is available.
    """

    if report_date is None:
        report_date = date.today()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Drift check: {ticker} — {report_date}")

    # Load datasets
    try:
        ref_df = load_reference_data(ticker)
        live_df = load_live_data(ticker)

    except Exception as e:
        logger.error(f"Data load failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}

    if live_df is None:
        return {"ticker": ticker, "error": "No live data"}

    # Compute drift
    result = compute_drift_scores(ref_df, live_df)

    result["ticker"] = ticker
    result["report_date"] = str(report_date)

    # Evidently HTML report
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        common_features = [
            f for f in MONITOR_FEATURES
            if f in ref_df.columns and f in live_df.columns
        ]

        ref_sample = ref_df[common_features].tail(200)
        live_sample = live_df[common_features]

        report = Report(metrics=[DataDriftPreset()])

        report.run(
            reference_data=ref_sample,
            current_data=live_sample
        )

        html_path = REPORTS_DIR / f"{ticker}_drift_{report_date}.html"

        report.save_html(str(html_path))

        result["html_report"] = str(html_path)

        logger.info(f"Evidently HTML report: {html_path}")

    except Exception as e:
        logger.warning(f"Evidently HTML report skipped: {e}")
        result["html_report"] = None

    # Save JSON summary
    json_path = REPORTS_DIR / f"{ticker}_drift_{report_date}.json"

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    alert = result.get("retrain_alert", False)

    logger.info(
        f"{ticker}: drift_ratio={result['drift_ratio']:.2%} | "
        f"drifted={result['n_drifted']}/{result['n_monitored']} features | "
        f"RETRAIN_ALERT={alert}"
    )

    return result


def run_all_tickers_drift(report_date: date = None) -> list:
    """Run drift check for all 10 ATLAS tickers."""

    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'JPM', 'GS', 'BAC', 'NVDA', 'TSLA'
    ]

    results = []
    alerts = []

    for ticker in TICKERS:

        result = generate_drift_report(ticker, report_date)

        results.append(result)

        if result.get("retrain_alert"):
            alerts.append(ticker)

    if alerts:
        logger.warning(
            f"RETRAIN ALERT: {len(alerts)} tickers have significant drift: "
            f"{alerts} — consider retraining models."
        )
    else:
        logger.info("Drift check complete — no retrain alert triggered.")

    return results


def load_latest_drift_summary() -> list:
    """Load the most recent JSON drift results for the dashboard."""

    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'JPM', 'GS', 'BAC', 'NVDA', 'TSLA'
    ]

    summaries = []

    for ticker in TICKERS:

        files = sorted(
            REPORTS_DIR.glob(f"{ticker}_drift_*.json"),
            reverse=True
        )

        if files:
            with open(files[0]) as f:
                summaries.append(json.load(f))

    return summaries