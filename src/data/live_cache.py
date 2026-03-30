import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH      = PROJECT_ROOT / 'data' / 'live' / 'atlas_live.db'
logger       = logging.getLogger('atlas.cache')

# Maximum age before data is considered stale (market-hours-aware)
STALE_HOURS = 26   # more than 1 trading day = stale

def init_db():
    """Create tables if they do not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS live_features (
            ticker        TEXT PRIMARY KEY,
            as_of_date    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            features_json TEXT NOT NULL,
            fetch_status  TEXT NOT NULL DEFAULT 'ok'
        )
    ''')

    cur.execute('''
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at       TEXT NOT NULL,
            tickers_ok   TEXT,
            tickers_fail TEXT,
            duration_sec REAL
        )
    ''')

    con.commit()
    con.close()
    logger.info(f'Database initialised at {DB_PATH}')

def save_features(ticker: str, features: pd.DataFrame, as_of_date: date):
    """Persist a 1-row feature DataFrame to the cache."""
    feat_dict = features.iloc[0].to_dict()
    # Convert numpy types to plain Python for JSON serialisation
    feat_dict = {k: float(v) for k, v in feat_dict.items()}
    con = sqlite3.connect(DB_PATH)
    con.execute('''
        INSERT OR REPLACE INTO live_features
        (ticker, as_of_date, updated_at, features_json, fetch_status)
        VALUES (?, ?, ?, ?, 'ok')
    ''', (
        ticker,
        str(as_of_date),
        datetime.utcnow().isoformat(),
        json.dumps(feat_dict),
    ))
    con.commit()
    con.close()
    logger.info(f'Saved live features for {ticker} as of {as_of_date}')

def load_features(ticker: str) -> dict | None:
    """
    Load the latest cached features for a ticker.
    Returns dict with keys: features (pd.DataFrame), as_of_date, updated_at, is_fresh
    Returns None if ticker not in cache.
    """
    con = sqlite3.connect(DB_PATH)
    row = con.execute(
        'SELECT as_of_date, updated_at, features_json FROM live_features WHERE ticker=?',
        (ticker,)
    ).fetchone()
    con.close()

    if row is None:
        return None

    as_of_date, updated_at, feat_json = row
    feat_dict  = json.loads(feat_json)
    features   = pd.DataFrame([feat_dict])

    # Freshness check
    updated_dt = datetime.fromisoformat(updated_at)
    age_hours  = (datetime.utcnow() - updated_dt).total_seconds() / 3600
    is_fresh   = age_hours <= STALE_HOURS

    return {
        'features'  : features,
        'as_of_date': as_of_date,
        'updated_at': updated_at,
        'age_hours' : round(age_hours, 1),
        'is_fresh'  : is_fresh,
    }

def get_pipeline_status() -> list[dict]:
    """
    Returns freshness status for ALL tickers currently in the database.
    Dynamically reads all tickers from DB — no hardcoded list.
    """
    init_db()  # Ensure DB exists
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        'SELECT ticker, as_of_date, updated_at, fetch_status FROM live_features'
    ).fetchall()
    con.close()

    # Import all 25 tickers from live_pipeline to ensure we report on all of them
    try:
        from src.data.live_pipeline import TICKERS
    except Exception:
        # Fallback: use whatever is in the DB
        TICKERS = [r[0] for r in rows]

    status_map = {r[0]: r for r in rows}
    result = []

    for t in TICKERS:
        if t in status_map:
            _, as_of, updated_at, fetch_status = status_map[t]
            updated_dt = datetime.fromisoformat(updated_at)
            age_h = (datetime.utcnow() - updated_dt).total_seconds() / 3600
            result.append({
                'ticker'    : t,
                'as_of_date': as_of,
                'updated_at': updated_at,
                'age_hours' : round(age_h, 1),
                'is_fresh'  : age_h <= STALE_HOURS,
                'status'    : fetch_status,
            })
        else:
            result.append({
                'ticker'  : t,
                'status'  : 'never_fetched',
                'is_fresh': False,
            })
    return result

def mark_fetch_failed(ticker: str, error: str):
    """Record a failed fetch attempt in the DB."""
    con = sqlite3.connect(DB_PATH)
    con.execute('''
        UPDATE live_features SET fetch_status=? WHERE ticker=?
    ''', (f'error: {error[:200]}', ticker))
    con.commit()
    con.close()