from datetime import datetime
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ── Engine imports ────────────────────────────────────────────────────────────
from app.atlas_engine import (
    get_signal, get_risk, get_all_signals,
    LIVE_CACHE_AVAILABLE, REGIME_MODELS_AVAILABLE,
    get_live_features_for_ticker,
)

# ── Pipeline status ───────────────────────────────────────────────────────────
try:
    from src.data.live_cache import get_pipeline_status
    PIPELINE_STATUS_AVAILABLE = True
except ImportError:
    PIPELINE_STATUS_AVAILABLE = False

# ── Portfolio optimizer (Phase 8) ─────────────────────────────────────────────
try:
    from src.risk.portfolio_optimizer import (
        load_price_matrix,
        optimize_max_sharpe,
        optimize_min_volatility,
        optimize_model_weighted,
    )
    PORTFOLIO_OPTIMIZER_AVAILABLE = True
except Exception:
    PORTFOLIO_OPTIMIZER_AVAILABLE = False

# ── Phase 13 Section 7: Broker imports ───────────────────────────────────────
try:
    from src.broker.alpaca_broker import get_account, get_positions, get_order_history
    from src.broker.order_executor import execute_all_signals
    from src.broker.position_tracker import (
        save_portfolio_snapshot, get_portfolio_history, get_logged_orders
    )
    BROKER_AVAILABLE = True
except Exception as _broker_err:
    BROKER_AVAILABLE = False
    print(f'WARNING: Broker not available — {_broker_err}')

# ── Phase 14 Section 7: Sentiment imports ────────────────────────────────────
try:
    from src.sentiment.sentiment_cache import (
        load_sentiment, load_all_sentiment, get_sentiment_history
    )
    from src.sentiment.signal_fusion import get_fused_signal, get_all_fused_signals
    SENTIMENT_AVAILABLE = True
except Exception as _sent_err:
    SENTIMENT_AVAILABLE = False
    print(f'WARNING: Sentiment not available — {_sent_err}')

# ── Phase 16: MLOps imports ───────────────────────────────────────────────────
try:
    from src.mlops.mlflow_logger import get_all_experiments_summary
    from src.mlops.drift_monitor import load_latest_drift_summary
    MLOPS_AVAILABLE = True
except Exception as _mlops_err:
    MLOPS_AVAILABLE = False
    print(f'WARNING: MLOps not available — {_mlops_err}')

app = FastAPI(
    title='ATLAS Trading Signal API',
    description='Algorithmic Trading with LLM-Augmented Signal Synthesis',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

class SignalRequest(BaseModel):
    ticker: str
    model_type: Optional[str] = 'xgboost'

TICKERS = ['AAPL','MSFT','GOOGL','AMZN','META',
           'JPM','GS','BAC','NVDA','TSLA']


# ── Root & Health ─────────────────────────────────────────────────────────────
@app.get('/')
def root():
    return {
        'name':    'ATLAS Trading Signal API',
        'version': '1.0.0',
        'status':  'running',
        'live_cache_available':          LIVE_CACHE_AVAILABLE,
        'regime_models_available':       REGIME_MODELS_AVAILABLE,
        'portfolio_optimizer_available': PORTFOLIO_OPTIMIZER_AVAILABLE,
        'broker_available':              BROKER_AVAILABLE,
        'sentiment_available':           SENTIMENT_AVAILABLE,
        'mlops_available':               MLOPS_AVAILABLE,
        'endpoints': [
            'GET  /health',
            'POST /signal',
            'GET  /signal/regime/{ticker}',
            'GET  /signal/{ticker}',
            'GET  /signals/regime/all',
            'GET  /signals/all',
            'GET  /risk/{ticker}',
            'GET  /portfolio/weights',
            'GET  /portfolio',
            'GET  /portfolio/positions',
            'GET  /orders/history',
            'POST /orders/execute',
            'GET  /sentiment/{ticker}',
            'GET  /sentiment/all',
            'GET  /sentiment/history/{ticker}',
            'GET  /signal/fused/{ticker}',
            'GET  /signals/fused/all',
            'GET  /mlops/experiments',
            'GET  /mlops/drift/latest',
            'GET  /status/pipeline',
            'GET  /docs',
        ]
    }


@app.get('/health')
def health():
    return {
        'status':                        'healthy',
        'models':                        'loaded',
        'live_cache_available':          LIVE_CACHE_AVAILABLE,
        'regime_models_available':       REGIME_MODELS_AVAILABLE,
        'portfolio_optimizer_available': PORTFOLIO_OPTIMIZER_AVAILABLE,
        'broker_available':              BROKER_AVAILABLE,
        'sentiment_available':           SENTIMENT_AVAILABLE,
        'mlops_available':               MLOPS_AVAILABLE,
    }


# ── CRITICAL: ALL specific routes BEFORE any generic /{ticker} routes ─────────
# Route order matters in FastAPI — first match wins.
# Order: sentiment/* → signal/regime/* → signal/fused/* → signals/fused/* → signal/{ticker}

# ── Phase 14 Section 7: Sentiment endpoints (MUST be before /signal/{ticker}) ─

@app.get('/sentiment/all')
def sentiment_all():
    """
    Returns today's FinBERT sentiment scores for all 10 tickers.
    Ranked by sentiment_score descending (most bullish first).
    """
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='Sentiment not available — run scheduler first'
        )
    results = load_all_sentiment()
    if not results:
        return {
            'message': 'No sentiment data for today — run scheduler first',
            'scores':  [],
        }
    return {'scores': results, 'count': len(results)}


@app.get('/sentiment/history/{ticker}')
def sentiment_history(ticker: str, days: int = 30):
    """
    Returns sentiment history for a ticker for the last N days.
    Query param: ?days=N  (default 30)
    """
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='Sentiment not available'
        )
    history = get_sentiment_history(ticker.upper(), days=days)
    return {'ticker': ticker.upper(), 'history': history, 'days': days}


@app.get('/sentiment/{ticker}')
def sentiment_ticker(ticker: str):
    """
    Returns today's FinBERT sentiment score for a ticker.
    Loaded from atlas_sentiment.db (populated by 8:30 AM scheduler).
    """
    if not SENTIMENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='Sentiment not available — run scheduler first'
        )
    result = load_sentiment(ticker.upper())
    if result is None:
        return {
            'ticker':          ticker.upper(),
            'sentiment_score': None,
            'sentiment_label': 'unavailable',
            'n_headlines':     0,
            'message':         'No sentiment data for today — scheduler may not have run yet',
        }
    result['ticker'] = ticker.upper()
    return result


# ── Phase 14: Fused signal endpoints (MUST be before /signal/{ticker}) ────────

@app.get('/signals/fused/all')
def fused_signals_all(model_type: str = 'xgboost'):
    """
    Returns fused signals for all 10 tickers.
    Each signal combines price model + FinBERT sentiment (70/30).
    Falls back to price-only signals if sentiment unavailable.
    """
    if not SENTIMENT_AVAILABLE or not REGIME_MODELS_AVAILABLE:
        return get_all_signals(model_type)
    return get_all_fused_signals(model_type)


@app.get('/signal/regime/{ticker}')
def regime_signal_get(ticker: str, model_type: str = 'xgboost'):
    """Returns regime-aware signal using live cache features."""
    if not REGIME_MODELS_AVAILABLE:
        result = get_signal(ticker.upper(), model_type)
        result['regime']         = 'unknown'
        result['regime_display'] = 'Base Model (Phase 10)'
        result['fallback_used']  = True
        return result
    from src.models.regime_predictor import get_regime_signal
    live_feats = get_live_features_for_ticker(ticker.upper())
    result = get_regime_signal(ticker.upper(), model_type, live_features=live_feats)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result


@app.get('/signal/fused/{ticker}')
def fused_signal_get(ticker: str, model_type: str = 'xgboost'):
    """
    Returns fused signal for a ticker: 70% price model + 30% FinBERT sentiment.
    Falls back to price-only signal if no sentiment data available.
    Query param: ?model_type=xgboost  or  ?model_type=lgbm
    """
    live_feats = get_live_features_for_ticker(ticker.upper())
    if REGIME_MODELS_AVAILABLE:
        from src.models.regime_predictor import get_regime_signal
        price_sig = get_regime_signal(
            ticker.upper(), model_type, live_features=live_feats
        )
    else:
        price_sig = get_signal(ticker.upper(), model_type)

    if 'error' in price_sig:
        raise HTTPException(status_code=404, detail=price_sig['error'])

    if not SENTIMENT_AVAILABLE:
        price_sig['fused']         = False
        price_sig['fusion_reason'] = 'Sentiment module not available'
        return price_sig

    return get_fused_signal(ticker.upper(), price_sig)


@app.get('/signal/{ticker}')
def signal_get(ticker: str, model_type: str = 'xgboost'):
    result = get_signal(ticker.upper(), model_type)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result


@app.post('/signal')
def signal_post(request: SignalRequest):
    result = get_signal(request.ticker.upper(), request.model_type)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result


# ── CRITICAL: /signals/regime/all BEFORE /signals/all ────────────────────────

@app.get('/signals/regime/all')
def regime_signals_all(model_type: str = 'xgboost'):
    """Returns regime-aware signals for all 10 tickers using live cache."""
    if not REGIME_MODELS_AVAILABLE:
        return get_all_signals(model_type)
    from src.models.regime_predictor import get_all_regime_signals
    return get_all_regime_signals(model_type)


@app.get('/signals/all')
def signals_all(model_type: str = 'xgboost'):
    return get_all_signals(model_type)


# ── Risk ──────────────────────────────────────────────────────────────────────
@app.get('/risk/{ticker}')
def risk(ticker: str):
    result = get_risk(ticker.upper())
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return result


# ── Portfolio weights (Phase 8 PyPortfolioOpt) ────────────────────────────────
@app.get('/portfolio/weights')
def portfolio_weights():
    """
    Returns 3 optimal portfolio weight allocations:
      - max_sharpe:  Maximises return per unit of risk (Markowitz)
      - min_vol:     Most defensive — lowest possible volatility
      - atlas_model: Tilts toward tickers with highest BULLISH probability today
    Falls back to equal weights (10% each) if optimizer unavailable.
    """
    equal = {t: round(1 / 10, 4) for t in TICKERS}

    if not PORTFOLIO_OPTIMIZER_AVAILABLE:
        fallback_result = {
            'weights': equal, 'sharpe': None,
            'exp_return': None, 'volatility': None,
            'strategy': 'Equal Weight (fallback)',
        }
        return {
            'max_sharpe':  fallback_result,
            'min_vol':     fallback_result,
            'atlas_model': fallback_result,
            'fallback':    True,
            'as_of':       datetime.utcnow().strftime('%Y-%m-%d'),
        }

    try:
        prices     = load_price_matrix()
        max_sharpe = optimize_max_sharpe(prices)
        min_vol    = optimize_min_volatility(prices)

        all_sigs   = get_all_signals('xgboost')
        signal_map = {
            s['ticker']: s['prob_up']
            for s in all_sigs
            if 'error' not in s
        }
        atlas_port = optimize_model_weighted(prices, signal_map)

        return {
            'max_sharpe':  max_sharpe,
            'min_vol':     min_vol,
            'atlas_model': atlas_port,
            'fallback':    False,
            'as_of':       datetime.utcnow().strftime('%Y-%m-%d'),
        }
    except Exception as e:
        return {
            'error':       str(e),
            'fallback':    True,
            'max_sharpe':  {'weights': equal},
            'min_vol':     {'weights': equal},
            'atlas_model': {'weights': equal},
        }


# ── Phase 13 Section 7: Broker endpoints ─────────────────────────────────────

@app.get('/portfolio')
def portfolio():
    """
    Returns current paper portfolio: account summary + open positions + history.
    Combines live Alpaca account data with local SQLite snapshot history.
    """
    if not BROKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='Broker not configured — add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env'
        )
    account   = get_account()
    positions = get_positions()
    history   = get_portfolio_history()
    return {
        'account':   account,
        'positions': positions,
        'history':   history,
    }


@app.get('/portfolio/positions')
def portfolio_positions():
    """Returns only current open positions from Alpaca paper account."""
    if not BROKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='Broker not configured — add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env'
        )
    return get_positions()


@app.get('/orders/history')
def orders_history(limit: int = 50):
    """
    Returns recent order history from Alpaca paper account.
    Query param: ?limit=N  (default 50, max recommended 200)
    """
    if not BROKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='Broker not configured — add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env'
        )
    return get_order_history(limit=limit)


@app.post('/orders/execute')
def orders_execute(model_type: str = 'xgboost'):
    """
    Manually trigger order execution for all 10 tickers.
    Reads current regime-aware signals → places BUY/SELL paper orders via Alpaca.
    Saves a portfolio snapshot to atlas_orders.db after execution.
    Safe to call anytime — checks market hours before submitting any order.
    Query param: ?model_type=xgboost  or  ?model_type=lgbm
    """
    if not BROKER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='Broker not configured — add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env'
        )
    results = execute_all_signals(model_type=model_type)
    save_portfolio_snapshot()
    return {
        'executed': len(results),
        'results':  results,
    }


# ── Phase 16 Section 8: MLOps endpoints (BEFORE /status/pipeline) ────────────

@app.get('/mlops/experiments')
def mlops_experiments():
    """
    Returns summary of all MLflow experiment runs.
    Shows latest accuracy and f1 per ticker+model combination.
    Browse the full UI at http://localhost:5000
    """
    if not MLOPS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='MLOps not available — install mlflow'
        )
    try:
        summary = get_all_experiments_summary()
        return {
            'experiments': summary,
            'count':       len(summary),
            'mlflow_ui':   'http://localhost:5000',
        }
    except Exception as e:
        return {'error': str(e), 'experiments': []}


@app.get('/mlops/drift/latest')
def mlops_drift_latest():
    """
    Returns the most recent drift report for all 10 tickers.
    Shows drift_ratio, drifted features, and retrain_alert flag.
    Drift check runs every Sunday at 9:00 AM ET via scheduler.
    """
    if not MLOPS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail='MLOps not available — install evidently'
        )
    try:
        summaries = load_latest_drift_summary()
        alerts    = [s['ticker'] for s in summaries if s.get('retrain_alert')]
        return {
            'drift_reports':  summaries,
            'retrain_alerts': alerts,
            'n_alerts':       len(alerts),
            'checked_at':     datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {'error': str(e), 'drift_reports': []}


# ── Pipeline status ───────────────────────────────────────────────────────────
@app.get('/status/pipeline')
def pipeline_status():
    """Returns live data pipeline status for all 10 tickers."""
    if not PIPELINE_STATUS_AVAILABLE:
        return {'error': 'live_cache not available', 'pipeline_ok': False}
    try:
        statuses    = get_pipeline_status()
        fresh_count = sum(1 for s in statuses if s.get('is_fresh', False))
        return {
            'pipeline_ok':   fresh_count == 10,
            'fresh_count':   fresh_count,
            'total_tickers': 10,
            'tickers':       statuses,
            'checked_at':    datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {'error': str(e), 'pipeline_ok': False}


if __name__ == '__main__':
    uvicorn.run(
        'app.main:app',
        host='0.0.0.0',
        port=8000,
        reload=True
    )