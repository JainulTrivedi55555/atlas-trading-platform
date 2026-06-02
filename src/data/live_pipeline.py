import time
import logging
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_LIVE    = PROJECT_ROOT / 'data' / 'live'
DATA_PROC    = PROJECT_ROOT / 'data' / 'processed'
LOG_DIR      = PROJECT_ROOT / 'logs'
DB_PATH      = DATA_LIVE / 'atlas_live.db'
DATA_LIVE.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger('atlas.pipeline')

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'JPM', 'GS', 'BAC', 'NVDA', 'TSLA',
    'NFLX', 'ORCL', 'AMD', 'CRM', 'UBER',
    'WMT', 'JNJ', 'XOM', 'LLY', 'V',
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT',
]

FETCH_PERIOD   = '1y'
FETCH_DELAY    = 1.2
# Minimum model edge to cache live features: abs(prob_up - 0.5) * 2
# 0.12 ≈ 6% directional edge — tuned for live deployment (0.30 skipped all prod tickers)
MIN_CONFIDENCE = 0.05


def fetch_ohlcv(ticker: str, period: str = FETCH_PERIOD) -> pd.DataFrame | None:
    try:
        logger.info(f'Fetching {ticker} from yfinance...')
        raw = yf.download(ticker, period=period, interval='1d',
                          auto_adjust=True, progress=False)

        if raw is None or raw.empty:
            logger.warning(f'{ticker}: yfinance returned empty DataFrame')
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [c.lower() for c in raw.columns]
        raw.index   = pd.to_datetime(raw.index).normalize()

        required = ['open', 'high', 'low', 'close', 'volume']
        missing  = [c for c in required if c not in raw.columns]
        if missing:
            logger.error(f'{ticker}: Missing columns {missing}')
            return None

        raw = raw.dropna(subset=['close']).ffill(limit=3)

        if len(raw) < 60:
            logger.error(f'{ticker}: Insufficient data ({len(raw)} rows)')
            return None

        logger.info(f'{ticker}: Fetched {len(raw)} rows, last {raw.index[-1].date()}')
        return raw

    except Exception as e:
        logger.error(f'{ticker}: Fetch failed — {e}')
        return None


# Columns that must exist after live feature build (training schema from technical_indicators.py)
REQUIRED_TRAINING_COLS = [
    'RSI_14', 'Volatility_20d', 'MACD', 'Daily_Return', 'CMF', 'BB_Pct',
]


def build_live_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    Build the latest feature row using the same schema as model training.

    Uses add_technical_indicators() so column names and formulas match
    data/processed/splits/{ticker}/X_train.csv (e.g. RSI_14, Volatility_20d).
    """
    try:
        from src.features.technical_indicators import add_technical_indicators

        df = df.copy().sort_index()

        # yfinance OHLCV (lowercase) -> training PascalCase
        train_df = pd.DataFrame(index=df.index)
        train_df['Open']   = df['open']
        train_df['High']   = df['high']
        train_df['Low']    = df['low']
        train_df['Close']  = df['close']
        train_df['Volume'] = df['volume']

        # Same derived columns as src/data_pipeline/price_cleaner.py
        train_df['Daily_Return'] = train_df['Close'].pct_change()
        train_df['Log_Return']   = np.log(train_df['Close'] / train_df['Close'].shift(1))
        train_df['Price_Range']  = train_df['High'] - train_df['Low']
        train_df['Gap']          = train_df['Open'] - train_df['Close'].shift(1)

        # Placeholders for tickers whose splits include these columns
        train_df['Dividends']      = 0.0
        train_df['Stock Splits']   = 0.0
        train_df['Capital Gains'] = 0.0

        train_df = add_technical_indicators(train_df)

        # Same EMA ratio features as fix_features_and_retrain.py (AAPL-style models)
        if 'EMA_12' in train_df.columns:
            train_df['Price_vs_EMA12'] = (
                (train_df['Close'] - train_df['EMA_12']) / train_df['EMA_12']
            )
        if 'EMA_26' in train_df.columns:
            train_df['Price_vs_EMA26'] = (
                (train_df['Close'] - train_df['EMA_26']) / train_df['EMA_26']
            )

        # Latest row only — do not dropna() on full history (long-window cols
        # leave early rows NaN but the last row is usable for live inference).
        last_row = train_df.ffill().iloc[[-1]].fillna(0)
        if last_row.empty:
            logger.error(f'{ticker}: No rows available after feature build')
            return None
        if last_row[REQUIRED_TRAINING_COLS].isnull().any().any():
            logger.error(f'{ticker}: Required prediction columns still NaN on latest row')
            return None
        logger.info(
            f'{ticker}: Features built for {last_row.index[0].date()} '
            f'({last_row.shape[1]} cols, training schema)'
        )
        return last_row

    except Exception as e:
        logger.error(f'{ticker}: Feature engineering failed — {e}')
        return None


def validate_feature_alignment(live_features: pd.DataFrame) -> bool:
    missing = [c for c in REQUIRED_TRAINING_COLS if c not in live_features.columns]
    if missing:
        logger.error(f'Missing training-schema columns: {missing}')
        return False
    cols = [c for c in REQUIRED_TRAINING_COLS if c in live_features.columns]
    if live_features[cols].isnull().any().any():
        logger.error(f'NaN in required live feature columns: {cols}')
        return False
    if np.isinf(live_features[cols].values).any():
        logger.error('Inf values in required live feature columns')
        return False
    logger.info(f'Feature validation passed: shape={live_features.shape}')
    return True


def get_model_feature_cols(ticker: str) -> list | None:
    """
    Read the exact feature column names from the saved training splits.
    This is the ONLY reliable way to know what features the model expects —
    it reads directly from the CSV files created during training.
    """
    try:
        split_dir = DATA_PROC / f'splits/{ticker}'
        x_train   = pd.read_csv(split_dir / 'X_train.csv', index_col=0, nrows=1)
        return list(x_train.columns)
    except Exception as e:
        logger.warning(f'{ticker}: Could not load training feature cols — {e}')
        return None


def check_confidence(ticker: str, df_feat: pd.DataFrame) -> tuple[bool, float]:
    """
    Run the XGBoost model on live features and check confidence.

    FIX: The model was trained on the feature columns from load_splits()
    (e.g. 37 columns for original tickers, 44 for new tickers).
    We align live features to those EXACT columns before predict_proba.
    This eliminates the 'Feature shape mismatch' error entirely.

    Confidence = abs(prob_up - 0.5) * 2
      prob_up=0.9  -> confidence=0.80  (very confident BULLISH)
      prob_up=0.5  -> confidence=0.00  (no edge)
      prob_up=0.2  -> confidence=0.60  (confident BEARISH)
    """
    try:
        model_path = PROJECT_ROOT / 'experiments' / 'models' / f'xgboost_{ticker}.pkl'
        if not model_path.exists():
            logger.warning(f'{ticker}: No XGBoost model — skipping confidence check')
            return True, 0.0

        # Get exact columns the model was trained on
        train_cols = get_model_feature_cols(ticker)
        if train_cols is None:
            logger.warning(f'{ticker}: Cannot verify feature cols — allowing through')
            return True, 0.0

        # Align live features to training columns exactly
        live_aligned = df_feat.copy()
        for col in train_cols:
            if col not in live_aligned.columns:
                live_aligned[col] = 0.0          # fill missing with 0
        live_aligned = live_aligned[train_cols]  # reorder + drop extras

        model      = joblib.load(model_path)
        prob_up    = model.predict_proba(live_aligned.values)[0][1]
        confidence = abs(prob_up - 0.5) * 2
        passes     = confidence >= MIN_CONFIDENCE

        logger.info(
            f'{ticker}: prob_up={prob_up:.3f}  confidence={confidence:.3f}  '
            f'threshold={MIN_CONFIDENCE}  -> {"PASS" if passes else "SKIP"}'
        )
        return passes, confidence

    except Exception as e:
        logger.warning(f'{ticker}: Confidence check error ({e}) — allowing through')
        return True, 0.0


def run_daily_pipeline():
    """
    Main pipeline — runs at 8:30 AM ET Mon-Fri via scheduler.py

    Steps:
      1. Fetch OHLCV from yfinance (25 tickers)
      2. Build technical features (training schema via technical_indicators.py)
      3. Validate feature alignment
      4. Confidence filter: skip tickers below MIN_CONFIDENCE (0.55)
      5. Save passing tickers to live cache (broker only sees these)
    """
    from src.data.live_cache import save_features, mark_fetch_failed

    start_time = datetime.utcnow()
    logger.info('=' * 60)
    logger.info(f'ATLAS Daily Pipeline — {start_time.isoformat()}')
    logger.info(f'Tickers: {len(TICKERS)} | Min confidence: {MIN_CONFIDENCE}')
    logger.info('=' * 60)

    ok_tickers      = []
    skipped_tickers = []
    fail_tickers    = []

    for ticker in TICKERS:
        try:
            # Step 1: Fetch
            df_raw = fetch_ohlcv(ticker)
            if df_raw is None:
                raise ValueError('fetch_ohlcv returned None')

            # Step 2: Features
            df_feat = build_live_features(df_raw, ticker)
            if df_feat is None:
                raise ValueError('build_live_features returned None')

            # Step 3: Validate
            if not validate_feature_alignment(df_feat):
                raise ValueError('Feature alignment validation failed')

            # Step 4: Confidence filter
            passes, confidence = check_confidence(ticker, df_feat)
            if not passes:
                logger.info(
                    f'{ticker}: SKIPPED — confidence={confidence:.3f} '
                    f'< {MIN_CONFIDENCE} — no trade today'
                )
                skipped_tickers.append({
                    'ticker': ticker,
                    'confidence': round(confidence, 3),
                })
                time.sleep(FETCH_DELAY)
                continue

            # Step 5: Save to cache
            as_of_date = df_feat.index[0].date()
            save_features(ticker, df_feat, as_of_date)
            ok_tickers.append(ticker)
            logger.info(
                f'{ticker}: SAVED — confidence={confidence:.3f} '
                f'for {as_of_date}'
            )
            time.sleep(FETCH_DELAY)

        except Exception as e:
            logger.error(f'{ticker}: FAILED — {e}')
            mark_fetch_failed(ticker, str(e))
            fail_tickers.append(ticker)

    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info('=' * 60)
    logger.info(f'Done in {duration:.1f}s')
    logger.info(f'Acting on  ({len(ok_tickers)}): {ok_tickers}')
    logger.info(f'Skipped    ({len(skipped_tickers)}): {[s["ticker"] for s in skipped_tickers]}')
    logger.info(f'Failed     ({len(fail_tickers)}): {fail_tickers}')
    logger.info('=' * 60)

    if fail_tickers:
        try:
            from scheduler import send_failure_alert
            send_failure_alert(fail_tickers, 'See logs/pipeline.log')
        except Exception:
            pass

    return {
        'ok': ok_tickers,
        'skipped': skipped_tickers,
        'fail': fail_tickers,
        'duration_sec': duration,
    }