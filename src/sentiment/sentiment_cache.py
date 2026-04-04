"""
Sentiment Cache
Stores FinBERT sentiment scores in SQLite.
Prevents re-running FinBERT on every API call.
"""
import sqlite3
import json
import logging
from datetime import datetime, date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger('atlas.sentiment_cache')

DB_PATH = Path('data/sentiment/atlas_sentiment.db')


def init_db():
    """Create sentiment database and tables if they don't exist."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker        TEXT    NOT NULL,
            score_date    TEXT    NOT NULL,
            sentiment_score REAL  NOT NULL,
            sentiment_label TEXT  NOT NULL,
            n_headlines   INTEGER NOT NULL,
            positive_pct  REAL    NOT NULL,
            negative_pct  REAL    NOT NULL,
            headlines_json TEXT,
            created_at    TEXT    DEFAULT (datetime('now')),
            UNIQUE(ticker, score_date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_headlines (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker     TEXT NOT NULL,
            fetch_date TEXT NOT NULL,
            headline   TEXT NOT NULL,
            score      REAL,
            label      TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()

    logger.info(f'Sentiment DB initialised at {DB_PATH}')


def save_sentiment(ticker: str, sentiment: dict,
                   headlines: list = None,
                   score_date: date = None) -> bool:
    """
    Save a sentiment result to the cache.

    Args:
        ticker:     Stock ticker
        sentiment:  Dict from aggregate_sentiment()
        headlines:  Optional list of raw headline strings
        score_date: Date for this score (default: today)
    """

    if score_date is None:
        score_date = date.today()

    score_date_str = str(score_date)

    try:
        conn = sqlite3.connect(DB_PATH)

        conn.execute("""
            INSERT OR REPLACE INTO sentiment_scores
            (ticker, score_date, sentiment_score, sentiment_label,
             n_headlines, positive_pct, negative_pct, headlines_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker.upper(),
            score_date_str,
            sentiment['sentiment_score'],
            sentiment['sentiment_label'],
            sentiment['n_headlines'],
            sentiment['positive_pct'],
            sentiment['negative_pct'],
            json.dumps(headlines or []),
        ))

        conn.commit()
        conn.close()

        logger.info(
            f'{ticker}: Sentiment cached — '
            f'score={sentiment["sentiment_score"]:.3f} '
            f'label={sentiment["sentiment_label"]} '
            f'headlines={sentiment["n_headlines"]}'
        )

        return True

    except Exception as e:
        logger.error(f'{ticker}: Cache save failed — {e}')
        return False


def load_sentiment(ticker: str, score_date: date = None) -> dict | None:
    """
    Load cached sentiment for a ticker.
    Returns None if no cached score exists for this date.

    Args:
        ticker:     Stock ticker
        score_date: Date to load (default: today)

    Returns:
        Sentiment dict or None
    """

    if score_date is None:
        score_date = date.today()

    score_date_str = str(score_date)

    try:
        conn = sqlite3.connect(DB_PATH)

        row = conn.execute("""
            SELECT ticker, score_date, sentiment_score, sentiment_label,
                   n_headlines, positive_pct, negative_pct, headlines_json
            FROM sentiment_scores
            WHERE ticker = ? AND score_date = ?
        """, (ticker.upper(), score_date_str)).fetchone()

        conn.close()

        if row is None:
            return None

        return {
            'ticker':           row[0],
            'score_date':       row[1],
            'sentiment_score':  row[2],
            'sentiment_label':  row[3],
            'n_headlines':      row[4],
            'positive_pct':     row[5],
            'negative_pct':     row[6],
            'headlines':        json.loads(row[7] or '[]'),
            'is_cached':        True,
        }

    except Exception as e:
        logger.error(f'{ticker}: Cache load failed — {e}')
        return None


def load_all_sentiment(score_date: date = None) -> list:
    """Load cached sentiment for all tickers for a given date."""

    if score_date is None:
        score_date = date.today()

    score_date_str = str(score_date)

    try:
        conn = sqlite3.connect(DB_PATH)

        rows = conn.execute("""
            SELECT ticker, score_date, sentiment_score, sentiment_label,
                   n_headlines, positive_pct, negative_pct
            FROM sentiment_scores
            WHERE score_date = ?
            ORDER BY sentiment_score DESC
        """, (score_date_str,)).fetchall()

        conn.close()

        return [{
            'ticker':          r[0],
            'score_date':      r[1],
            'sentiment_score': r[2],
            'sentiment_label': r[3],
            'n_headlines':     r[4],
            'positive_pct':    r[5],
            'negative_pct':    r[6],
        } for r in rows]

    except Exception as e:
        logger.error(f'Load all sentiment failed: {e}')
        return []


def get_sentiment_history(ticker: str, days: int = 30) -> list:
    """Load sentiment history for a ticker for the last N days."""

    try:
        conn = sqlite3.connect(DB_PATH)

        rows = conn.execute("""
            SELECT score_date, sentiment_score, sentiment_label, n_headlines
            FROM sentiment_scores
            WHERE ticker = ?
            ORDER BY score_date DESC
            LIMIT ?
        """, (ticker.upper(), days)).fetchall()

        conn.close()

        return [{
            'date': r[0],
            'score': r[1],
            'label': r[2],
            'n_headlines': r[3]
        } for r in rows]

    except Exception as e:
        logger.error(f'{ticker}: History load failed — {e}')
        return []


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)

    init_db()

    print(f'Sentiment DB created at: {DB_PATH}')
    print(f'Absolute path: {DB_PATH.resolve()}')