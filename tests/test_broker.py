"""
Broker Integration Tests
Tests Alpaca connection, account, positions, and order tracking.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.broker.alpaca_broker import get_account, get_positions, get_order_history
from src.broker.order_executor import is_market_open, get_signal_for_ticker
from src.broker.position_tracker import init_db, save_portfolio_snapshot

def test_alpaca_connection():
    """Can connect to Alpaca paper account."""
    account = get_account()
    assert account['paper'] == True, "Must be paper trading!"
    assert account['portfolio_value'] > 0, "Portfolio value must be > 0"
    assert 'cash' in account
    print(f"test_alpaca_connection: PASSED — "
          f"portfolio=${account['portfolio_value']:,.2f}")

def test_paper_mode():
    """Confirm paper=True is enforced."""
    account = get_account()
    assert account['paper'] == True, "CRITICAL: paper must be True!"
    print("test_paper_mode: PASSED — paper trading confirmed")

def test_positions_returns_list():
    """get_positions returns a list."""
    positions = get_positions()
    assert isinstance(positions, list)
    print(f"test_positions_returns_list: PASSED — {len(positions)} open positions")

def test_order_history_returns_list():
    """get_order_history returns a list."""
    orders = get_order_history(limit=10)
    assert isinstance(orders, list)
    print(f"test_order_history_returns_list: PASSED — {len(orders)} orders")

def test_market_clock():
    """is_market_open returns a bool."""
    result = is_market_open()
    assert isinstance(result, bool)
    print(f"test_market_clock: PASSED — market_open={result}")

def test_signal_fetch():
    """Can fetch a regime signal for AAPL."""
    sig = get_signal_for_ticker('AAPL')
    assert 'signal' in sig
    assert sig['signal'] in ('BULLISH', 'BEARISH')
    print(f"test_signal_fetch: PASSED — "
          f"AAPL signal={sig['signal']} regime={sig.get('regime','N/A')}")

def test_db_snapshot():
    """Can save a portfolio snapshot to SQLite."""
    init_db()
    snapshot = save_portfolio_snapshot()
    assert snapshot['portfolio_value'] > 0
    assert Path('data/broker/atlas_orders.db').exists()
    print(f"test_db_snapshot: PASSED — snapshot saved to atlas_orders.db")

if __name__ == '__main__':
    print("ATLAS Phase 13 — Broker Integration Tests")
    print("=" * 50)
    test_alpaca_connection()
    test_paper_mode()
    test_positions_returns_list()
    test_order_history_returns_list()
    test_market_clock()
    test_signal_fetch()
    test_db_snapshot()
    print()
    print("=" * 50)
    print("ALL PHASE 13 TESTS PASSED")
    print("=" * 50)