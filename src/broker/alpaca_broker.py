"""
Alpaca Broker Module
Single point of contact for all Alpaca API interactions.
Uses PAPER trading only — base URL is paper-api.alpaca.markets
"""
import sys, warnings, os
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import pandas as pd
from datetime import datetime, timezone

# 🟢 Load keys from .env 
API_KEY    = os.getenv('ALPACA_API_KEY',    '')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')

if not API_KEY or not SECRET_KEY:
    raise EnvironmentError(
        "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in your .env file. "
     
    )

# 🔵 Paper trading client (paper=True is critical) 
_trading_client = None
_data_client    = None

def get_trading_client() -> TradingClient:
    """Returns a cached paper trading client."""
    global _trading_client
    if _trading_client is None:
        _trading_client = TradingClient(
            api_key=API_KEY,
            secret_key=SECRET_KEY,
            paper=True   # ← ALWAYS True — never change this
        )
    return _trading_client

def get_data_client() -> StockHistoricalDataClient:
    """Returns a cached data client for live quotes."""
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(
            api_key=API_KEY,
            secret_key=SECRET_KEY,
        )
    return _data_client

def get_account() -> dict:
    """
    Returns current paper account summary.
    Key fields: portfolio_value, cash, buying_power, equity
    """
    client  = get_trading_client()
    account = client.get_account()
    return {
        'portfolio_value': float(account.portfolio_value),
        'cash':            float(account.cash),
        'buying_power':    float(account.buying_power),
        'equity':          float(account.equity),
        'currency':        account.currency,
        'status':          account.status,
        'paper':           True,
        'as_of':           datetime.now(timezone.utc).isoformat(),
    }

def get_positions() -> list:
    """
    Returns all currently open positions.
    Each item: ticker, qty, market_value, avg_entry, unrealized_pl, unrealized_plpc
    """
    client    = get_trading_client()
    positions = client.get_all_positions()
    result    = []
    for p in positions:
        result.append({
            'ticker':          p.symbol,
            'qty':             float(p.qty),
            'side':            p.side.value if hasattr(p.side, 'value') else str(p.side),
            'market_value':    float(p.market_value),
            'avg_entry_price': float(p.avg_entry_price),
            'current_price':   float(p.current_price),
            'unrealized_pl':   float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc) * 100,  # convert to %
        })
    return result

def get_position(ticker: str) -> dict | None:
    """Returns open position for a specific ticker, or None if flat."""
    client = get_trading_client()
    try:
        p = client.get_open_position(ticker.upper())
        return {
            'ticker':          p.symbol,
            'qty':             float(p.qty),
            'side':            p.side.value if hasattr(p.side, 'value') else str(p.side),
            'market_value':    float(p.market_value),
            'avg_entry_price': float(p.avg_entry_price),
            'current_price':   float(p.current_price),
            'unrealized_pl':   float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc) * 100,
        }
    except Exception:
        return None

def get_order_history(limit: int = 50) -> list:
    """Returns the last N filled/cancelled orders."""
    client = get_trading_client()
    req    = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        limit=limit,
    )
    orders = client.get_orders(filter=req)
    result = []
    for o in orders:
        result.append({
            'id':          str(o.id),
            'ticker':      o.symbol,
            'side':        o.side.value if hasattr(o.side, 'value') else str(o.side),
            'qty':         float(o.qty) if o.qty else 0,
            'filled_qty':  float(o.filled_qty) if o.filled_qty else 0,
            'filled_price':float(o.filled_avg_price) if o.filled_avg_price else 0,
            'status':      o.status.value if hasattr(o.status, 'value') else str(o.status),
            'created_at':  str(o.created_at),
            'type':        o.type.value if hasattr(o.type, 'value') else str(o.type),
        })
    return result

def get_latest_price(ticker: str) -> float | None:
    """Returns the latest ask price for a ticker."""
    try:
        client = get_data_client()
        req    = StockLatestQuoteRequest(symbol_or_symbols=ticker.upper())
        quote  = client.get_stock_latest_quote(req)
        return float(quote[ticker.upper()].ask_price)
    except Exception:
        return None