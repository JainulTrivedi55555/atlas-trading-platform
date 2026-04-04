"""
Position Tracker
Reads live positions from Alpaca and saves daily snapshots to SQLite.
"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import sqlite3
import json
from datetime import datetime, timezone
from src.broker.alpaca_broker import get_account, get_positions, get_order_history

DB_PATH = Path('data/broker/atlas_orders.db')
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    
    # Portfolio snapshots (one row per day)
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date   TEXT NOT NULL,
            portfolio_value REAL,
            cash            REAL,
            equity          REAL,
            buying_power    REAL,
            n_positions     INTEGER,
            positions_json  TEXT,
            created_at      TEXT
        )
    """)
    
    # Individual order records
    c.execute("""
        CREATE TABLE IF NOT EXISTS order_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id     TEXT UNIQUE,
            ticker       TEXT,
            side         TEXT,
            qty          REAL,
            filled_qty   REAL,
            filled_price REAL,
            status       TEXT,
            regime       TEXT,
            signal       TEXT,
            created_at   TEXT,
            logged_at    TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_portfolio_snapshot():
    """
    Save current portfolio state to DB.
    Call this once per day after order execution.
    """
    init_db()
    account   = get_account()
    positions = get_positions()
    today     = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    snapshot = {
        'snapshot_date':   today,
        'portfolio_value': account['portfolio_value'],
        'cash':            account['cash'],
        'equity':          account['equity'],
        'buying_power':    account['buying_power'],
        'n_positions':     len(positions),
        'positions_json':  json.dumps(positions),
        'created_at':      datetime.now(timezone.utc).isoformat(),
    }
    
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    
    # Upsert — one snapshot per day
    c.execute("""
        INSERT OR REPLACE INTO portfolio_snapshots
        (snapshot_date, portfolio_value, cash, equity,
         buying_power, n_positions, positions_json, created_at)
        VALUES (:snapshot_date, :portfolio_value, :cash, :equity,
                :buying_power, :n_positions, :positions_json, :created_at)
    """, snapshot)
    
    conn.commit()
    conn.close()
    
    print(f"Portfolio snapshot saved: ${snapshot['portfolio_value']:,.2f} | "
          f"{snapshot['n_positions']} positions")
    return snapshot

def log_order(order_dict: dict, regime: str = '', signal: str = ''):
    """Log an executed order to the DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    try:
        c.execute("""
            INSERT OR IGNORE INTO order_log
            (order_id, ticker, side, qty, filled_qty, filled_price,
             status, regime, signal, created_at, logged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order_dict.get('id', ''),
            order_dict.get('ticker', ''),
            order_dict.get('side', ''),
            order_dict.get('qty', 0),
            order_dict.get('filled_qty', 0),
            order_dict.get('filled_price', 0),
            order_dict.get('status', ''),
            regime,
            signal,
            order_dict.get('created_at', ''),
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
    finally:
        conn.close()

def get_portfolio_history() -> list:
    """Returns all portfolio snapshots ordered by date."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT snapshot_date, portfolio_value, cash, n_positions
        FROM portfolio_snapshots
        ORDER BY snapshot_date ASC
    """)
    rows = c.fetchall()
    conn.close()
    
    return [
        {'date': r[0], 'portfolio_value': r[1],
         'cash': r[2], 'n_positions': r[3]}
        for r in rows
    ]

def get_logged_orders(limit: int = 50) -> list:
    """Returns recent orders from local DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        SELECT ticker, side, qty, filled_price, status,
               regime, signal, created_at
        FROM order_log
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    
    return [
        {'ticker': r[0], 'side': r[1], 'qty': r[2],
         'filled_price': r[3], 'status': r[4],
         'regime': r[5], 'signal': r[6], 'created_at': r[7]}
        for r in rows
    ]

