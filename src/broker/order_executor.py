"""
Order Executor
Converts ATLAS signals into Alpaca paper trading orders.

KEY FIX: Broker now reads signals ONLY from the live cache (tickers that
passed the confidence filter in live_pipeline.py). It no longer calls
get_regime_signal() directly, which was bypassing the confidence filter
and using a different threshold (0.55) than the pipeline (0.30).

Flow:
  live_pipeline.py  → confidence filter (0.30) → saves to live cache
  order_executor.py → reads live cache → executes only cached tickers
"""
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import joblib
import numpy as np
from datetime import datetime

from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums    import OrderSide, TimeInForce

from src.broker.alpaca_broker import (
    get_trading_client, get_account,
    get_position, get_positions, get_latest_price,
)

# Import all 25 tickers from live_pipeline (not config.py which has only 10)
from src.data.live_pipeline import TICKERS, MIN_CONFIDENCE
from src.data.live_cache    import load_features, get_pipeline_status

# ── Config ────────────────────────────────────────────────────────────────────
POSITION_PCT  = 0.05    # 5% of portfolio per position
MIN_ORDER_USD = 10.0    # minimum order value in dollars
LOG_PATH      = Path('logs/broker.log')
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Logging — APPEND mode so history is never wiped ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='a', encoding='utf-8'),  # ← APPEND
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('atlas.broker')


def is_market_open() -> bool:
    """Check if US stock market is currently open via Alpaca clock."""
    try:
        client = get_trading_client()
        clock  = client.get_clock()
        return clock.is_open
    except Exception as e:
        logger.warning(f'Could not check market clock: {e}')
        return False


def get_signal_from_cache(ticker: str, model_type: str = 'xgboost') -> dict | None:
    """
    Get signal for a ticker FROM THE LIVE CACHE ONLY.

    Returns None if ticker is not in cache (i.e. it was skipped by the
    confidence filter in live_pipeline.py). This ensures the broker only
    acts on tickers that passed the confidence filter.
    """
    try:
        cached = load_features(ticker)
        if cached is None or not cached.get('is_fresh'):
            return None  # Not in cache = was filtered out or stale

        df_feat = cached['features']

        # Load model and get signal
        ROOT       = Path('.')
        model_path = ROOT / 'experiments' / 'models' / f'{model_type}_{ticker}.pkl'
        if not model_path.exists():
            logger.warning(f'{ticker}: Model not found at {model_path}')
            return None

        model = joblib.load(model_path)

        # ── Column alignment fix ───────────────────────────────────────────
        # Models were trained on different feature sets (37 vs 44 columns).
        # Align live features to exactly the columns the model was trained on.
        try:
            from src.data.live_pipeline import get_model_feature_cols
            train_cols = get_model_feature_cols(ticker)
            if train_cols is not None:
                live = df_feat.copy()
                for col in train_cols:
                    if col not in live.columns:
                        live[col] = 0.0
                df_feat = live[train_cols]
        except Exception as align_err:
            logger.warning(f'{ticker}: Column alignment skipped — {align_err}')

        prob_up  = model.predict_proba(df_feat.values)[0][1]
        signal   = 'BULLISH' if prob_up >= 0.5 else 'BEARISH'
        conf     = abs(prob_up - 0.5) * 2

        # Also get regime if available
        regime = 'unknown'
        try:
            from src.models.regime_predictor import get_regime_signal
            regime_result = get_regime_signal(ticker, model_type)
            regime = regime_result.get('regime', 'unknown')
            # Use regime model signal if available (more context-aware)
            signal = regime_result.get('signal', signal)
            conf   = regime_result.get('confidence', conf)
        except Exception:
            pass  # Regime model optional — fall back to base signal

        return {
            'ticker':     ticker,
            'signal':     signal,
            'confidence': conf,
            'prob_up':    prob_up,
            'regime':     regime,
        }

    except Exception as e:
        logger.error(f'{ticker}: Failed to get signal from cache — {e}')
        return None


def calculate_order_qty(portfolio_value: float, price: float) -> int:
    """Calculate number of shares to buy using POSITION_PCT."""
    order_value = portfolio_value * POSITION_PCT
    if order_value < MIN_ORDER_USD:
        return 0
    qty = int(order_value / price)
    return max(qty, 0)


def execute_signal(ticker: str, model_type: str = 'xgboost') -> dict:
    """
    Execute the ATLAS signal for one ticker as a paper trade.

    Only acts if the ticker is in the live cache (passed confidence filter).
    """
    result = {
        'ticker':   ticker,
        'action':   'none',
        'order_id': None,
        'qty':      0,
        'signal':   None,
        'confidence': None,
        'reason':   '',
        'success':  False,
    }

    # ── Step 1: Get signal from cache (None = filtered out) ────────────────
    sig = get_signal_from_cache(ticker, model_type)

    if sig is None:
        result['reason'] = f'Not in live cache — skipped by confidence filter'
        return result

    signal     = sig.get('signal', 'BEARISH')
    confidence = sig.get('confidence', 0.0)
    regime     = sig.get('regime', 'unknown')

    result['signal']     = signal
    result['confidence'] = confidence

    logger.info(f'{ticker}: signal={signal} regime={regime} conf={confidence:.3f}')

    # ── Step 2: Get current position ───────────────────────────────────────
    current_pos = get_position(ticker)
    is_long     = current_pos is not None and float(current_pos.get('qty', 0)) > 0
    client      = get_trading_client()

    # ── Step 3: Execute based on signal + position ─────────────────────────
    if signal == 'BULLISH' and not is_long:
        try:
            account       = get_account()
            portfolio_val = account['portfolio_value']
            price         = get_latest_price(ticker)

            if price is None or price <= 0:
                result['reason'] = 'Could not get current price'
                return result

            qty = calculate_order_qty(portfolio_val, price)
            if qty == 0:
                result['reason'] = 'Order quantity too small'
                return result

            order_req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            order = client.submit_order(order_req)
            result.update({
                'action':   'BUY',
                'order_id': str(order.id),
                'qty':      qty,
                'price':    price,
                'reason':   f'BULLISH (regime={regime}, conf={confidence:.1%})',
                'success':  True,
            })
            logger.info(f'{ticker}: BUY {qty} shares @ ~${price:.2f} | order_id={order.id}')

        except Exception as e:
            result['reason'] = f'BUY order failed: {e}'
            logger.error(f'{ticker}: {result["reason"]}')

    elif signal == 'BEARISH' and is_long:
        try:
            client.close_position(ticker)
            result.update({
                'action':  'SELL',
                'qty':     float(current_pos.get('qty', 0)),
                'reason':  f'BEARISH — closing position (regime={regime})',
                'success': True,
            })
            logger.info(f'{ticker}: SELL (close) {current_pos.get("qty")} shares')

        except Exception as e:
            result['reason'] = f'SELL order failed: {e}'
            logger.error(f'{ticker}: {result["reason"]}')

    elif signal == 'BULLISH' and is_long:
        result['reason'] = 'Already long — holding'
    elif signal == 'BEARISH' and not is_long:
        result['reason'] = 'BEARISH — no position, no action'

    return result


def execute_all_signals(model_type: str = 'xgboost') -> list:
    """
    Execute signals for all tickers that passed the confidence filter.

    Only tickers present in the live cache (saved by live_pipeline.py after
    passing the confidence filter) will have orders placed.
    """
    logger.info('=' * 55)
    logger.info(f'ATLAS Order Execution — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Tickers: {len(TICKERS)} total | Confidence threshold: {MIN_CONFIDENCE:.0%}')
    logger.info('=' * 55)

    # Check which tickers are actually in cache (passed confidence filter)
    statuses     = get_pipeline_status()
    cached_today = {s['ticker'] for s in statuses if s.get('is_fresh')}
    logger.info(f'Tickers in live cache today: {sorted(cached_today)}')

    market_open = is_market_open()
    if not market_open:
        logger.warning('Market is closed — logging signals only, no orders submitted.')

    results = []
    for ticker in TICKERS:
        try:
            if market_open:
                res = execute_signal(ticker, model_type)
            else:
                # Market closed — still get signal info for logging, but don't trade
                sig = get_signal_from_cache(ticker, model_type)
                if sig:
                    res = {
                        'ticker':     ticker,
                        'action':     'market_closed',
                        'signal':     sig.get('signal', 'N/A'),
                        'confidence': sig.get('confidence', 0),
                        'regime':     sig.get('regime', 'N/A'),
                        'reason':     'Market closed — no order submitted',
                        'success':    False,
                    }
                    logger.info(
                        f'{ticker}: {sig["signal"]} conf={sig["confidence"]:.3f} '
                        f'— market closed, no order'
                    )
                else:
                    res = {
                        'ticker':  ticker,
                        'action':  'skipped',
                        'signal':  'N/A',
                        'reason':  'Not in cache — filtered by confidence',
                        'success': False,
                    }
                    logger.info(f'{ticker}: SKIPPED — not in live cache (low confidence)')
            results.append(res)

        except Exception as e:
            logger.error(f'{ticker}: Unexpected error — {e}')
            results.append({
                'ticker': ticker, 'action': 'error',
                'reason': str(e), 'success': False,
            })

    # Summary
    buys    = sum(1 for r in results if r.get('action') == 'BUY')
    sells   = sum(1 for r in results if r.get('action') == 'SELL')
    skipped = sum(1 for r in results if r.get('action') in ('skipped', 'none'))
    closed  = sum(1 for r in results if r.get('action') == 'market_closed')

    logger.info(f'Execution complete: {buys} BUY | {sells} SELL | {skipped} skipped | {closed} market_closed')
    logger.info('=' * 55)
    return results