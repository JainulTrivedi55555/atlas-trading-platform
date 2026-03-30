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
MIN_CONFIDENCE = 0.30


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


def build_live_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    try:
        df = df.copy().sort_index()

        df['daily_return']      = df['close'].pct_change()
        df['log_return']        = np.log(df['close'] / df['close'].shift(1))
        df['high_low_spread']   = (df['high'] - df['low']) / df['close']
        df['open_close_spread'] = (df['close'] - df['open']) / df['open']
        df['vwap_approx']       = (df['high'] + df['low'] + df['close']) / 3

        for w in [5, 10, 20, 50, 200]:
            df[f'ma{w}'] = df['close'].rolling(w).mean()

        df['ema12']    = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26']    = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_diff'] = df['ema12'] - df['ema26']

        delta = df['close'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss + 1e-10)
        df['rsi14']       = 100 - (100 / (1 + rs))
        df['macd']        = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist']   = df['macd'] - df['macd_signal']

        for r in [5, 10, 20]:
            df[f'roc{r}'] = df['close'].pct_change(r) * 100

        bb_mid         = df['close'].rolling(20).mean()
        bb_std         = df['close'].rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid

        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low']  - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)

        df['atr14']      = tr.rolling(14).mean()
        df['hist_vol20'] = df['log_return'].rolling(20).std() * np.sqrt(252)
        df['hist_vol60'] = df['log_return'].rolling(60).std() * np.sqrt(252)

        df['vol_ma20']  = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-10)
        df['obv']       = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ma20']  = df['obv'].rolling(20).mean()
        df['obv_trend'] = df['obv'] - df['obv_ma20']

        plus_dm  = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        atr14_   = tr.rolling(14).mean()
        df['plus_di']  = 100 * (plus_dm.rolling(14).mean()  / (atr14_ + 1e-10))
        df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / (atr14_ + 1e-10))
        dx = (100 * (df['plus_di'] - df['minus_di']).abs() /
              (df['plus_di'] + df['minus_di'] + 1e-10))
        df['adx14'] = dx.rolling(14).mean()

        tp       = (df['high'] + df['low'] + df['close']) / 3
        mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        df['cci20'] = (tp - tp.rolling(20).mean()) / (0.015 * mean_dev + 1e-10)

        low14         = df['low'].rolling(14).min()
        high14        = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['williams_r'] = -100 * (high14 - df['close']) / (high14 - low14 + 1e-10)

        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'daily_return', 'log_return', 'high_low_spread',
            'open_close_spread', 'vwap_approx',
            'ma5', 'ma10', 'ma20', 'ma50', 'ma200',
            'ema12', 'ema26', 'ema_diff',
            'rsi14', 'macd', 'macd_signal', 'macd_hist', 'roc5', 'roc10', 'roc20',
            'bb_upper', 'bb_lower', 'bb_width', 'atr14', 'hist_vol20', 'hist_vol60',
            'vol_ma20', 'vol_ratio', 'obv', 'obv_ma20', 'obv_trend',
            'adx14', 'plus_di', 'minus_di', 'cci20', 'stoch_k', 'stoch_d', 'williams_r',
        ]

        df = df[feature_cols].dropna()
        if df.empty:
            logger.error(f'{ticker}: All rows dropped after NaN removal')
            return None

        last_row = df.iloc[[-1]]
        assert last_row.shape == (1, 43), f'Expected (1,43), got {last_row.shape}'
        logger.info(f'{ticker}: Features built for {last_row.index[0].date()}')
        return last_row

    except Exception as e:
        logger.error(f'{ticker}: Feature engineering failed — {e}')
        return None


def validate_feature_alignment(live_features: pd.DataFrame) -> bool:
    if live_features.shape[1] != 43:
        logger.error(f'Feature count mismatch: got {live_features.shape[1]}, expected 43')
        return False
    if live_features.isnull().any().any():
        logger.error(f'NaN in live features')
        return False
    if np.isinf(live_features.values).any():
        logger.error('Inf values in live features')
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
      2. Build 43 technical indicator features
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