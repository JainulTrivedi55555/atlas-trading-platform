import time
import logging
import sys
import smtplib
from email.mime.text import MIMEText
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron         import CronTrigger

# ── Import the REAL pipeline from live_pipeline (has confidence filter) ────────
from src.data.live_pipeline import run_daily_pipeline, TICKERS
from src.data.live_cache    import init_db

# ── Phase 13: Broker execution ─────────────────────────────────────────────────
from src.broker.order_executor   import execute_all_signals
from src.broker.position_tracker import save_portfolio_snapshot

# ── Phase 14: Sentiment pipeline ──────────────────────────────────────────────
from src.sentiment.news_fetcher    import fetch_all_tickers
from src.sentiment.finbert_scorer  import aggregate_sentiment
from src.sentiment.sentiment_cache import init_db as init_sentiment_db
from src.sentiment.sentiment_cache import save_sentiment

# ── Phase 16: MLOps drift monitor ─────────────────────────────────────────────
try:
    from src.mlops.drift_monitor import run_all_tickers_drift
    DRIFT_MONITOR_AVAILABLE = True
except ImportError:
    DRIFT_MONITOR_AVAILABLE = False


# ── Logging Setup ──────────────────────────────────────────────────────────────
# IMPORTANT: mode='a' means APPEND — logs are never wiped on restart
log_path = Path('logs') / 'pipeline.log'
log_path.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a', encoding='utf-8'),  # ← APPEND mode
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('atlas.scheduler')

if not DRIFT_MONITOR_AVAILABLE:
    logger.warning('Drift monitor not available — install evidently')


# ── Optional email alert on pipeline failure ───────────────────────────────────
def send_failure_alert(fail_tickers: list, error_summary: str):
    """Send email alert when pipeline fails. Optional — configure or skip."""
    gmail_user = os.getenv('GMAIL_USER')
    gmail_pw   = os.getenv('GMAIL_APP_PW')
    if not gmail_user or not gmail_pw:
        logger.warning('Email alerts not configured (no GMAIL_USER/GMAIL_APP_PW in .env)')
        return
    subject = f'ATLAS Pipeline Alert: {len(fail_tickers)} ticker(s) failed'
    body    = f'Failed tickers: {fail_tickers}\n\nError summary:\n{error_summary}'
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From']    = gmail_user
    msg['To']      = gmail_user
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_user, gmail_pw)
            server.sendmail(gmail_user, [gmail_user], msg.as_string())
        logger.info('Failure alert email sent')
    except Exception as e:
        logger.error(f'Failed to send alert email: {e}')


# ── Sentiment pipeline ─────────────────────────────────────────────────────────
def run_sentiment_pipeline():
    """
    Fetch news + run FinBERT for all tickers.
    Runs in a subprocess to give PyTorch a clean memory state.
    (PyTorch's c10.dll fails when called after heavy XGBoost/yfinance work
    in the same process — subprocess gives it a fresh start every time.)
    """
    logger.info('Phase 14: Starting sentiment pipeline (subprocess)...')
    import subprocess
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "from src.sentiment.news_fetcher import fetch_all_tickers; "
        "from src.sentiment.finbert_scorer import aggregate_sentiment; "
        "from src.sentiment.sentiment_cache import init_db, save_sentiment; "
        "import logging; logging.basicConfig(level=logging.INFO, "
        "format='%(asctime)s  %(levelname)s  %(message)s'); "
        "logger = logging.getLogger('atlas.sentiment'); "
        "init_db(); "
        "all_headlines = fetch_all_tickers(); "
        "results = []; "
        "[results.append({'ticker': t, 'label': aggregate_sentiment(h)['sentiment_label'], "
        "'score': aggregate_sentiment(h)['sentiment_score'], 'n': aggregate_sentiment(h)['n_headlines']}) "
        "or save_sentiment(t, aggregate_sentiment(h), h) "
        "for t, h in all_headlines.items()]; "
        "b=[r for r in results if r['label']=='bullish']; "
        "n=[r for r in results if r['label']=='neutral']; "
        "be=[r for r in results if r['label']=='bearish']; "
        "logger.info(f'Sentiment complete: {len(b)} bullish | {len(be)} bearish | {len(n)} neutral')"
    )
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=False,
            timeout=300,
            cwd=str(Path(__file__).parent),
        )
        if result.returncode == 0:
            logger.info('Sentiment pipeline subprocess completed successfully.')
        else:
            logger.error(f'Sentiment subprocess exited with code {result.returncode}')
    except subprocess.TimeoutExpired:
        logger.error('Sentiment pipeline timed out after 300s')
    except Exception as e:
        logger.error(f'Sentiment pipeline subprocess failed: {e}')


# ── Phase 16: Drift monitor ────────────────────────────────────────────────────
def run_drift_monitor():
    """
    Weekly drift check — runs every Sunday at 9:00 AM ET.
    Compares live features against training data distribution.
    """
    if not DRIFT_MONITOR_AVAILABLE:
        logger.warning('Drift monitor skipped — evidently not installed')
        return

    logger.info('Phase 16: Starting weekly drift check...')
    try:
        results = run_all_tickers_drift()
        alerts  = [r['ticker'] for r in results if r.get('retrain_alert')]
        logger.info(f'Drift check complete: {len(alerts)}/{len(TICKERS)} tickers need retraining')
        if alerts:
            logger.warning(f'RETRAIN ALERT: Significant drift in {alerts}. Run train_regime_models.py')
    except Exception as e:
        logger.error(f'Drift monitor failed: {e}')


# ── Broker execution ───────────────────────────────────────────────────────────
def run_broker_execution():
    """
    Execute ATLAS signals as paper trades via Alpaca.
    Called after sentiment pipeline (~8:33 AM ET).
    Orders are DAY type — fill at market open 9:30 AM ET.
    """
    logger.info('Phase 13: Starting broker execution...')
    try:
        results = execute_all_signals(model_type='xgboost')
        buys    = [r for r in results if r.get('action') == 'BUY']
        sells   = [r for r in results if r.get('action') == 'SELL']
        skipped = [r for r in results if r.get('action') not in ('BUY', 'SELL')]

        logger.info(
            f'Broker execution complete: '
            f'{len(buys)} BUY | {len(sells)} SELL | {len(skipped)} skipped'
        )
        for r in buys + sells:
            logger.info(
                f"  {r.get('action')} {r.get('ticker')} — "
                f"signal={r.get('signal')} "
                f"confidence={r.get('confidence', 'N/A')} "
                f"qty={r.get('qty', 'N/A')}"
            )

        snapshot = save_portfolio_snapshot()
        logger.info(
            f"Portfolio snapshot saved: "
            f"${snapshot['portfolio_value']:,.2f} | "
            f"{snapshot.get('n_positions', 0)} positions"
        )
    except Exception as e:
        logger.error(f'Broker execution failed: {e}')


# ── Scheduled job: full daily run ──────────────────────────────────────────────
def scheduled_daily_job():
    """
    This is what runs at 8:30 AM ET Mon-Fri.

    Calls run_daily_pipeline() from src/data/live_pipeline.py — the version
    that includes the confidence filter (MIN_CONFIDENCE = 0.30).
    Then runs sentiment + broker execution.
    """
    logger.info('=' * 60)
    logger.info(f'ATLAS Scheduled Job — {datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")}')
    logger.info(f'Tickers: {TICKERS}')
    logger.info('=' * 60)

    # ── Step 1: Live data + confidence filter ──────────────────────────────────
    # This calls src/data/live_pipeline.py run_daily_pipeline()
    # which includes the confidence filter (skips low-conviction tickers)
    logger.info('Step 1: Running live data pipeline with confidence filter...')
    try:
        result = run_daily_pipeline()
        ok      = result.get('ok', [])
        skipped = result.get('skipped', [])
        failed  = result.get('fail', [])
        logger.info(f'Pipeline done — Acting on: {ok}')
        logger.info(f'Pipeline done — Skipped  : {[s["ticker"] for s in skipped]}')
        logger.info(f'Pipeline done — Failed   : {failed}')
        if failed:
            send_failure_alert(failed, 'See logs/pipeline.log for details')
    except Exception as e:
        logger.error(f'Live pipeline failed: {e}')

    # ── Step 2: Sentiment ──────────────────────────────────────────────────────
    logger.info('Step 2: Running sentiment pipeline...')
    run_sentiment_pipeline()

    # ── Step 3: Broker execution ───────────────────────────────────────────────
    logger.info('Step 3: Running broker execution...')
    run_broker_execution()

    logger.info('=' * 60)
    logger.info('Scheduled daily job complete.')
    logger.info('=' * 60)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info(f'ATLAS Scheduler starting — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Tickers loaded: {len(TICKERS)} — {TICKERS}')
    logger.info('=' * 60)

    # Initialise databases
    init_db()
    init_sentiment_db()

    # Run full pipeline immediately on startup so data is always fresh
    logger.info('Running startup pipeline (immediate run)...')
    scheduled_daily_job()

    # Run drift check on startup
    if DRIFT_MONITOR_AVAILABLE:
        logger.info('Running startup drift check...')
        run_drift_monitor()

    # ── APScheduler ───────────────────────────────────────────────────────────
    scheduler = BackgroundScheduler(timezone='America/New_York')

    # Daily pipeline — Mon-Fri 8:30 AM ET
    scheduler.add_job(
        scheduled_daily_job,
        trigger=CronTrigger(
            hour=8, minute=30,
            day_of_week='mon-fri',
            timezone='America/New_York'
        ),
        id='daily_pipeline',
        name='ATLAS Daily Pipeline + Sentiment + Broker',
        misfire_grace_time=3600,
    )

    # Weekly drift check — Sunday 9:00 AM ET
    scheduler.add_job(
        run_drift_monitor,
        trigger=CronTrigger(
            hour=9, minute=0,
            day_of_week='sun',
            timezone='America/New_York'
        ),
        id='weekly_drift_check',
        name='ATLAS Weekly Drift Monitor',
        misfire_grace_time=3600,
    )

    scheduler.start()
    logger.info('Scheduler started.')
    logger.info('Pipeline runs at 8:30 AM ET Mon-Fri.')
    logger.info('Drift monitor runs every Sunday at 9:00 AM ET.')
    logger.info('Press Ctrl+C to stop.')

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info('Scheduler stopping...')
        scheduler.shutdown()
        logger.info('Scheduler stopped.')