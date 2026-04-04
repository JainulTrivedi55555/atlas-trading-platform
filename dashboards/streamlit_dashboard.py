import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
from src.data.live_pipeline import TICKERS


def get_regime_badge(regime: str) -> str:
    badges = {
        'bull':    '<span style="background:#0D2818;color:#3FB950;'
                   'padding:3px 10px;border-radius:12px;font-weight:bold;'
                   'border:1px solid #3FB950">🟢 Bull Market</span>',
        'bear':    '<span style="background:#2D0D0D;color:#F85149;'
                   'padding:3px 10px;border-radius:12px;font-weight:bold;'
                   'border:1px solid #F85149">🔴 Bear Market</span>',
        'highvol': '<span style="background:#1F1700;color:#D29922;'
                   'padding:3px 10px;border-radius:12px;font-weight:bold;'
                   'border:1px solid #D29922">🟡 High Volatility</span>',
    }
    return badges.get(regime, '<span style="color:#8B949E">Unknown</span>')

try:
    from src.data.live_cache import get_pipeline_status
    LIVE_ENABLED = True
except ImportError:
    LIVE_ENABLED = False

# Import confidence filter settings from live pipeline
try:
    from src.data.live_pipeline import MIN_CONFIDENCE
    TICKER_THRESHOLDS = {}  # Per-ticker thresholds not used — all use MIN_CONFIDENCE
    FILTER_ENABLED = True
except (ImportError, Exception):
    TICKER_THRESHOLDS = {}
    MIN_CONFIDENCE = 0.30
    FILTER_ENABLED = False


st.set_page_config(
    page_title='ATLAS Trading Dashboard',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

API_URL = 'http://localhost:8000'


def call_api(endpoint: str) -> dict:
    try:
        r = requests.get(f'{API_URL}{endpoint}', timeout=10)
        return r.json()
    except Exception as e:
        return {'error': str(e)}

def call_api_post(endpoint: str, params: dict = None) -> dict:
    try:
        r = requests.post(f'{API_URL}{endpoint}', params=params, timeout=30)
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def show_data_freshness_banner():
    if not LIVE_ENABLED:
        st.info('🔌 Serving historical data (Phase 10 mode)')
        return
    statuses = get_pipeline_status()
    fresh_tickers = [s['ticker'] for s in statuses if s.get('is_fresh')]
    stale_tickers = [s['ticker'] for s in statuses if not s.get('is_fresh')]
    total = len(statuses) or 1
    if len(fresh_tickers) == total:
        latest_date = statuses[0].get('as_of_date', 'unknown')
        st.success(f'🟢 LIVE DATA — All {total} tickers fresh as of {latest_date}')
    elif len(fresh_tickers) > 0:
        st.warning(
            f'🟡 PARTIAL LIVE — {len(fresh_tickers)}/{total} tickers fresh. '
            f'Stale: {", ".join(stale_tickers)}'
        )
    else:
        st.warning('🟠 HISTORICAL MODE — Live pipeline not yet run today.')

def get_ticker_freshness_badge(ticker: str) -> str:
    if not LIVE_ENABLED:
        return '<span style="color:#8B949E">Historical</span>'
    statuses  = get_pipeline_status()
    ticker_st = next((s for s in statuses if s['ticker'] == ticker), None)
    if ticker_st and ticker_st.get('is_fresh'):
        return f'<span style="color:#3FB950;font-weight:bold">🟢 LIVE {ticker_st.get("as_of_date","")}</span>'
    elif ticker_st and 'as_of_date' in ticker_st:
        return f'<span style="color:#D29922">🟡 STALE (last: {ticker_st.get("as_of_date","")})</span>'
    return '<span style="color:#F85149">🔴 NO DATA</span>'


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title('ATLAS Controls')
st.sidebar.markdown('---')

selected_ticker = st.sidebar.selectbox('Select Ticker', TICKERS, index=0)
model_type      = st.sidebar.selectbox('Model Type', ['xgboost', 'lgbm'], index=0)

st.sidebar.markdown('---')
st.sidebar.markdown('**API Status**')
health = call_api('/health')
if 'error' not in health:
    st.sidebar.success('✅ API Online')
else:
    st.sidebar.error('❌ API Offline — run uvicorn first')

# Pipeline sync status
st.sidebar.markdown('---')
st.sidebar.markdown('**⏱️ Pipeline Sync**')
try:
    if LIVE_ENABLED:
        _statuses = get_pipeline_status()
        _fresh    = [s for s in _statuses if s.get('is_fresh')]
        if _fresh:
            st.sidebar.success(f'✅ Last sync: {_fresh[0].get("as_of_date","unknown")}')
            st.sidebar.caption(f'{len(_fresh)}/{len(_statuses)} tickers live')
        else:
            st.sidebar.warning('⚠️ No fresh data yet today')
            st.sidebar.caption('Run: python scheduler.py to start auto-updates')
    else:
        st.sidebar.info('🔌 Historical mode')
except Exception as _e:
    st.sidebar.error(f'Sync check failed: {_e}')

if st.sidebar.button('🗑️ Clear Cache & Rerun'):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
    
# ── Confidence Filter Status in Sidebar ──────────────────────────────────────
st.sidebar.markdown('---')
st.sidebar.markdown('**🎯 Confidence Filter**')
if FILTER_ENABLED:
    st.sidebar.success(f'✅ Active — min confidence: {MIN_CONFIDENCE:.0%}')
    thresh = TICKER_THRESHOLDS.get(selected_ticker, MIN_CONFIDENCE)
    st.sidebar.caption(f'{selected_ticker} threshold: {thresh:.1%}')
else:
    st.sidebar.warning('⚠️ Filter not loaded')


st.sidebar.markdown('---')
st.sidebar.subheader('🔄 Live Pipeline')
if st.sidebar.button('Refresh Now (All Tickers)'):
    with st.spinner('Fetching live data...'):
        try:
            from src.data.live_pipeline import (
                fetch_ohlcv, build_live_features, validate_feature_alignment
            )
            from src.data.live_cache import save_features
            results = []
            for ticker in TICKERS:
                df_raw = fetch_ohlcv(ticker)
                if df_raw is not None:
                    df_feat = build_live_features(df_raw, ticker)
                    if df_feat is not None and validate_feature_alignment(df_feat):
                        save_features(ticker, df_feat, df_feat.index[0].date())
                        results.append(f'✅ {ticker}')
                    else:
                        results.append(f'⚠️ {ticker} (feature error)')
                else:
                    results.append(f'❌ {ticker} (fetch failed)')
                time.sleep(0.3)
            st.sidebar.success('Done!\n' + '  '.join(results))
            st.rerun()
        except Exception as e:
            st.sidebar.error(f'Refresh failed: {e}')


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
st.title('📊 ATLAS — Algorithmic Trading Signal Dashboard')
show_data_freshness_banner()
st.markdown(f'*Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*')
st.markdown('---')

# ── Base Signal ───────────────────────────────────────────────────────────────
st.subheader(f'🎯 Signal — {selected_ticker}')
badge_html = get_ticker_freshness_badge(selected_ticker)
st.markdown(f'**Data Source:** {badge_html}', unsafe_allow_html=True)

sig_data = call_api(f'/signal/{selected_ticker}?model_type={model_type}')

if 'error' not in sig_data:
    col1, col2, col3, col4, col5 = st.columns([1.8, 1.2, 1.2, 1.4, 1.4])
    signal_val     = sig_data['signal']
    signal_display = f"{'🟢' if signal_val == 'BULLISH' else '🔴'} {signal_val}"
    col1.metric('Signal',      signal_display)
    col2.metric('Confidence',  f"{sig_data['confidence']*100:.1f}%")
    col3.metric('Prob Up',     f"{sig_data['prob_up']*100:.1f}%")
    col4.metric('As Of',       sig_data.get('as_of_date', 'N/A'))
    col5.metric('Data Source', '🟢 Live' if sig_data.get('is_live') else '🔌 Historical')

    # ── Confidence Filter Check for this ticker ───────────────────────────
    if FILTER_ENABLED:
        conf_val  = sig_data.get('confidence', 0)
        thresh    = TICKER_THRESHOLDS.get(selected_ticker, MIN_CONFIDENCE)
        passes    = conf_val >= thresh
        gap       = conf_val - thresh
        if passes:
            st.success(
                f'✅ **Confidence Filter: PASS** — {conf_val:.1%} ≥ threshold {thresh:.1%} '
                f'(+{gap:.1%} above threshold) → Signal eligible for execution'
            )
        else:
            st.warning(
                f'⚠️ **Confidence Filter: BLOCKED** — {conf_val:.1%} < threshold {thresh:.1%} '
                f'({abs(gap):.1%} below threshold) → Signal will NOT be sent to broker today'
            )

    if sig_data.get('top_features'):
        st.markdown('**Top 5 Features driving this signal:**')
        feat_df = pd.DataFrame(
            sig_data['top_features'].items(),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        feat_df = feat_df[feat_df['Importance'] > 0]
        fig = px.bar(
            feat_df, x='Importance', y='Feature', orientation='h',
            title=f'Top 5 Feature Importances — {selected_ticker} ({model_type.upper()})',
            color='Importance', color_continuous_scale='Blues',
            text=feat_df['Importance'].apply(
                lambda v: f'{v:.4f}' if v > 0.0001 else '~0'),
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=320, margin=dict(l=20, r=60, t=50, b=20),
            coloraxis_showscale=False,
            xaxis_title='Importance Score', yaxis_title='',
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No feature importance data returned for this ticker/model.')
else:
    st.error(f'Signal error: {sig_data["error"]}')

st.markdown('---')

# ── Regime Signal ─────────────────────────────────────────────────────────────
st.subheader(f'🏛️ Regime Signal — {selected_ticker}')
regime_data = call_api(f'/signal/regime/{selected_ticker}?model_type={model_type}')

if 'error' not in regime_data and 'detail' not in regime_data:
    regime_key = regime_data.get('regime', 'unknown')
    badge      = get_regime_badge(regime_key)
    model_used = regime_data.get('model_used', 'base')
    fallback   = regime_data.get('fallback_used', False)

    st.markdown(
        f'**Detected Regime:** {badge}'
        + (f' <span style="color:#8B949E;font-size:12px"> (fallback to base model)</span>'
           if fallback else ''),
        unsafe_allow_html=True
    )
    st.markdown(f'**Model Used:** `{model_used}`')
    st.markdown('')

    rc1, rc2, rc3, rc4 = st.columns(4)
    r_signal = regime_data['signal']
    rc1.metric('Regime Signal', f"{'🟢' if r_signal == 'BULLISH' else '🔴'} {r_signal}")
    rc2.metric('Confidence',    f"{regime_data['confidence']*100:.1f}%")
    rc3.metric('Regime',        regime_key.upper())
    rc4.metric('As Of',         regime_data.get('as_of_date', 'N/A'))

    base_signal   = sig_data.get('signal', 'N/A') if 'error' not in sig_data else 'N/A'
    regime_signal = regime_data['signal']
    if base_signal != 'N/A' and base_signal != regime_signal:
        st.warning(
            f'⚠️ **Signal Divergence:** Base model says **{base_signal}** but '
            f'Regime model says **{regime_signal}**. '
            f'The regime-aware model is more context-specific — prefer it.'
        )
    elif base_signal == regime_signal:
        st.success(
            f'✅ Both base and regime models agree: **{regime_signal}** — '
            f'High-confidence signal.'
        )
else:
    st.info(
        '🔺 Regime models not yet available. '
        'Run: `python train_regime_models.py`'
    )

st.markdown('---')

# ── Fused Signal Panel ────────────────────────────────────────────────────────
st.subheader(f'💡 Fused Signal — {selected_ticker}')
st.markdown(
    'Combines **price model** (70%) + **FinBERT news sentiment** (30%). '
    'This is the final ATLAS signal.'
)

fused_data = call_api(f'/signal/fused/{selected_ticker}?model_type={model_type}')

if 'error' not in fused_data:
    is_fused = fused_data.get('fused', False)

    if is_fused:
        f1, f2, f3, f4, f5 = st.columns([1.8, 1.2, 1.2, 1.2, 1.4])
        fused_sig = fused_data.get('fused_signal', 'N/A')
        f1.metric('Fused Signal', f"{'🟢' if fused_sig == 'BULLISH' else '🔴'} {fused_sig}")
        f2.metric('Fused Confidence', f"{fused_data.get('fused_confidence', 0)*100:.1f}%")
        f3.metric('Sentiment Score',  f"{fused_data.get('sentiment_score', 0)*100:.1f}%")
        f4.metric('Headlines Scored', str(fused_data.get('n_headlines', 0)))
        sent_label = fused_data.get('sentiment_label', 'neutral')
        sent_emoji = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '🟡'}.get(sent_label, '⚠')
        f5.metric('Sentiment', f'{sent_emoji} {sent_label.upper()}')

        st.caption(f"📉 {fused_data.get('fusion_reason', '')}")

        divergence = fused_data.get('signal_divergence', False)
        price_sig  = fused_data.get('price_signal', 'N/A')
        sent_sig   = fused_data.get('sentiment_signal', 'N/A')
        if divergence:
            st.warning(
                f'⚠️ **Signal Divergence Detected**: Price model says **{price_sig}** '
                f'but news sentiment says **{sent_sig}**. '
                f'Treat the fused signal with lower confidence.'
            )
        else:
            st.success(
                f'✅ Price model and sentiment agree: **{fused_sig}** — '
                f'Higher confidence fused signal.'
            )
    else:
        st.info(
            f'💡 Fused signal unavailable — sentiment pipeline may not have run yet. '
            f'Reason: {fused_data.get("fusion_reason", "unknown")}. '
            f'Price-only signal: **{fused_data.get("signal", "N/A")}**'
        )
else:
    st.info('💡 Fused signal loading... ensure API and sentiment pipeline are running.')

st.markdown('---')

# ── Live Sentiment Feed ───────────────────────────────────────────────────────
st.subheader('📰 Live Sentiment Feed — All Tickers')
st.markdown("FinBERT sentiment scores from today's financial news. Updated daily at 8:30 AM ET.")

sent_all = call_api('/sentiment/all')

if 'error' not in sent_all and sent_all.get('scores'):
    scores = sent_all['scores']
    sent_rows = []
    for s in scores:
        label = s.get('sentiment_label', 'neutral')
        score = s.get('sentiment_score', 0.5)
        emoji = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '🟡'}.get(label, '⚠')
        bar   = '█' * int(score * 10)
        sent_rows.append({
            'Ticker':    s['ticker'],
            'Sentiment': f'{emoji} {label.upper()}',
            'Score':     f'{score:.3f}',
            'Bar':       bar,
            'Headlines': s.get('n_headlines', 0),
            'Positive%': f"{s.get('positive_pct', 0)*100:.0f}%",
            'Negative%': f"{s.get('negative_pct', 0)*100:.0f}%",
            'Date':      s.get('score_date', 'N/A'),
        })
    sent_df = pd.DataFrame(sent_rows)

    def color_sentiment(val):
        if 'BULLISH' in str(val): return 'background-color: #0D2818; color: #3FB950'
        if 'BEARISH' in str(val): return 'background-color: #2D0D0D; color: #F85149'
        return 'background-color: #1F1700; color: #D29922'

    styled_sent = sent_df.style.map(color_sentiment, subset=['Sentiment'])
    st.dataframe(styled_sent, use_container_width=True)

    sent_counts = sent_df['Sentiment'].str.replace(
        r'[🟢🔴🟡⚠] ', '', regex=True
    ).value_counts().reset_index()
    sent_counts.columns = ['Sentiment', 'Count']
    fig_sent = px.bar(
        sent_counts, x='Sentiment', y='Count', color='Sentiment',
        color_discrete_map={'BULLISH': '#2ECC71', 'BEARISH': '#E74C3C', 'NEUTRAL': '#F39C12'},
        title='Sentiment Distribution — All Tickers Today',
        text='Count',
    )
    fig_sent.update_traces(textposition='outside')
    fig_sent.update_layout(height=260, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_sent, use_container_width=True)

    with st.expander('📈 Sentiment vs Price Signal Comparison'):
        fused_all = call_api(f'/signals/fused/all?model_type={model_type}')
        if isinstance(fused_all, list) and len(fused_all) > 0:
            comp_rows = []
            for f in fused_all:
                if 'error' not in f:
                    comp_rows.append({
                        'Ticker':       f.get('ticker', ''),
                        'Price Signal': f.get('price_signal', f.get('signal', 'N/A')),
                        'Sentiment':    str(f.get('sentiment_label') or 'N/A').upper(),
                        'Fused Signal': f.get('fused_signal', f.get('signal', 'N/A')),
                        'Divergence':   '⚠️ YES' if f.get('signal_divergence') else '✅ NO',
                    })
            if comp_rows:
                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(comp_df, use_container_width=True)
else:
    st.info(
        '📰 No sentiment data yet today. Run the sentiment pipeline first:\n\n'
        '```\npython -c "import sys; sys.path.insert(0,\'.\'); '
        'from scheduler import run_sentiment_pipeline; '
        'from src.sentiment.sentiment_cache import init_db; '
        'init_db(); run_sentiment_pipeline()"\n```'
    )

st.markdown('---')

# ── All Tickers Table ─────────────────────────────────────────────────────────
st.subheader(f"All {len(TICKERS)} Tickers — Live Signals")

all_sigs = call_api(f'/signals/regime/all?model_type={model_type}')
if isinstance(all_sigs, dict) and 'error' in all_sigs:
    all_sigs = call_api(f'/signals/all?model_type={model_type}')

if isinstance(all_sigs, list) and len(all_sigs) > 0:
    sigs_data = []
    for s in all_sigs:
        if 'error' not in s:
            ticker_     = s['ticker']
            conf_val    = s['confidence']
            thresh      = TICKER_THRESHOLDS.get(ticker_, MIN_CONFIDENCE) if FILTER_ENABLED else 0.0
            filter_pass = '✅ PASS' if conf_val >= thresh else '⚠️ BLOCKED'
            sigs_data.append({
                'Ticker':     ticker_,
                'Signal':     s['signal'],
                'Regime':     s.get('regime', 'N/A').upper(),
                'Confidence': f"{conf_val*100:.1f}%",
                'Threshold':  f"{thresh*100:.1f}%",
                'Filter':     filter_pass,
                'Prob Up':    f"{s['prob_up']*100:.1f}%",
                'Date':       s['as_of_date'],
                'Source':     '🟢 Live' if s.get('is_live') else '🔌 Historical',
            })
    sigs_df = pd.DataFrame(sigs_data)

    def color_signal(val):
        return ('background-color: #d4edda' if val == 'BULLISH'
                else 'background-color: #f8d7da')

    def color_filter(val):
        if 'PASS'    in str(val): return 'color: #3FB950; font-weight: bold'
        if 'BLOCKED' in str(val): return 'color: #D29922; font-weight: bold'
        return ''

    def color_regime(val):
        if val == 'BULL':    return 'color: #3FB950; font-weight: bold'
        if val == 'BEAR':    return 'color: #F85149; font-weight: bold'
        if val == 'HIGHVOL': return 'color: #D29922; font-weight: bold'
        return ''

    styled = sigs_df.style\
        .map(color_signal, subset=['Signal'])\
        .map(color_regime,  subset=['Regime'])\
        .map(color_filter,  subset=['Filter'])
    st.dataframe(styled, use_container_width=True)

    # Count how many are actually eligible to trade today
    n_pass    = sigs_df['Filter'].str.contains('PASS').sum()
    n_blocked = sigs_df['Filter'].str.contains('BLOCKED').sum()
    fc1, fc2 = st.columns(2)
    fc1.metric('✅ Signals Eligible Today', n_pass)
    fc2.metric('⚠️ Signals Blocked Today', n_blocked)

    signal_counts = sigs_df['Signal'].astype(str).value_counts().reset_index()
    signal_counts.columns = ['Signal', 'Count']
    fig2 = px.bar(
        signal_counts, x='Signal', y='Count', color='Signal',
        color_discrete_map={'BULLISH': '#2ECC71', 'BEARISH': '#E74C3C'},
        title='Signal Distribution — All Tickers', text='Count',
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(height=280, showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning('No signal data — is the API running?')

st.markdown('---')

# ── Risk Metrics ──────────────────────────────────────────────────────────────
st.subheader(f'⚠️ Risk Metrics — {selected_ticker}')
risk_data = call_api(f'/risk/{selected_ticker}')

if 'error' not in risk_data:
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric('Sharpe Ratio',  risk_data.get('sharpe_ratio', 'N/A'))
    r2.metric('Sortino Ratio', risk_data.get('sortino_ratio', 'N/A'))
    r3.metric('Max Drawdown',  f"{risk_data.get('max_drawdown', 'N/A')}%")
    r4.metric('VaR 95%',       f"{risk_data.get('var_95', 'N/A')}%")
    r5.metric('Ann. Return',   f"{risk_data.get('ann_return', 'N/A')}%")
else:
    st.error(f'Risk error: {risk_data["error"]}')

st.markdown('---')

# ── Backtest Equity Curve ─────────────────────────────────────────────────────
st.subheader(f'📋 Backtest Equity Curve — {selected_ticker}')
try:
    from src.backtesting.strategy import run_atlas_strategy, run_buy_hold
    xgb_result = run_atlas_strategy(selected_ticker, model_type)
    bh_result  = run_buy_hold(selected_ticker)
    if xgb_result and bh_result:
        fig3, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            xgb_result['results'].index,
            xgb_result['results']['strategy_equity'],
            label=f'ATLAS {model_type.upper()}',
            color='#2ECC71', linewidth=2
        )
        ax.plot(
            bh_result['results'].index,
            bh_result['results']['strategy_equity'],
            label='Buy-and-Hold',
            color='#E74C3C', linewidth=2, linestyle='--'
        )
        ax.axhline(10000, color='black', alpha=0.3, linestyle=':')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.set_title(f'{selected_ticker} — ATLAS vs Buy-and-Hold (2023–2025)')
        st.pyplot(fig3)
        plt.close(fig3)
except Exception as e:
    st.info(f'Equity curve unavailable: {e}')

st.markdown('---')

# ══════════════════════════════════════════════════════════════════════════════
# ── BACKTEST RESULTS PANEL ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.subheader('📊 Backtest Results — All 25 Tickers')
st.markdown(
    'Results from the held-out test set (2023–2025). '
    'Ensemble XGBoost 50% + LightGBM 50%, adaptive confidence thresholds.'
)

BACKTEST_CSV = ROOT / 'data' / 'backtest_report.csv'

if BACKTEST_CSV.exists():
    bt_df = pd.read_csv(BACKTEST_CSV)

    # Identify key columns flexibly
    def find_col(df, candidates):
        for c in candidates:
            matches = [col for col in df.columns if c.lower() in col.lower()]
            if matches:
                return matches[0]
        return None

    sharpe_col   = find_col(bt_df, ['sharpe'])
    winrate_col  = find_col(bt_df, ['winrate', 'win_rate', 'win rate'])
    prodfact_col = find_col(bt_df, ['profit', 'profact', 'prof_fact'])
    maxdd_col    = find_col(bt_df, ['maxdd', 'max_dd', 'drawdown'])
    atlas_col    = find_col(bt_df, ['atlas%', 'atlas_pct', 'atlas'])
    bh_col       = find_col(bt_df, ['b&h', 'bh%', 'buyhold', 'buy_hold'])
    trades_col   = find_col(bt_df, ['trades'])
    ticker_col   = find_col(bt_df, ['ticker'])

    # Filter to valid stat rows only (trades >= 20)
    if trades_col:
        valid_bt = bt_df[pd.to_numeric(bt_df[trades_col], errors='coerce') >= 20].copy()
    else:
        valid_bt = bt_df.copy()

    # Display summary metrics
    if sharpe_col and winrate_col:

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(
            '📊 Valid Tickers',
            f"{len(valid_bt)}/{len(bt_df)}",
            help='Tickers with ≥20 trades (statistically meaningful)'
        )
        if sharpe_col:
            avg_sharpe = pd.to_numeric(valid_bt[sharpe_col], errors='coerce').mean()
            k2.metric('⚡ Avg Sharpe', f"{avg_sharpe:.3f}", delta="target >0.5")
        if winrate_col:
            avg_wr = pd.to_numeric(valid_bt[winrate_col], errors='coerce').mean()
            k3.metric('🎯 Avg Win Rate', f"{avg_wr:.1f}%", delta="target >52%")
        if prodfact_col:
            avg_pf = pd.to_numeric(valid_bt[prodfact_col], errors='coerce').mean()
            k4.metric('💰 Avg Profit Factor', f"{avg_pf:.2f}", delta="target >1.2")
        if maxdd_col:
            avg_dd = pd.to_numeric(valid_bt[maxdd_col], errors='coerce').mean()
            k5.metric('🛡️ Avg Max Drawdown', f"{avg_dd:.1f}%", delta="target >-25%")

    st.markdown('---')

    # ── Full results table ────────────────────────────────────────────────
    st.markdown('**Full Results Table**')

    # Colour-code the table
    def color_backtest_row(row):
        styles = [''] * len(row)
        if sharpe_col and sharpe_col in row.index:
            v = pd.to_numeric(row[sharpe_col], errors='coerce')
            if pd.notna(v):
                idx = row.index.get_loc(sharpe_col)
                styles[idx] = 'color: #3FB950' if v > 0.5 else (
                    'color: #D29922' if v > 0 else 'color: #F85149')
        if atlas_col and atlas_col in row.index:
            v = pd.to_numeric(row[atlas_col], errors='coerce')
            if pd.notna(v):
                idx = row.index.get_loc(atlas_col)
                styles[idx] = 'color: #3FB950' if v > 0 else 'color: #F85149'
        return styles

    styled_bt = bt_df.style.apply(color_backtest_row, axis=1)
    st.dataframe(styled_bt, use_container_width=True, height=500)

    # ── ATLAS vs B&H bar chart ────────────────────────────────────────────
    if atlas_col and bh_col and ticker_col:
        st.markdown('**ATLAS Return vs Buy & Hold — Per Ticker**')
        chart_df = bt_df[[ticker_col, atlas_col, bh_col]].copy()
        chart_df[atlas_col] = pd.to_numeric(chart_df[atlas_col], errors='coerce')
        chart_df[bh_col]    = pd.to_numeric(chart_df[bh_col],    errors='coerce')
        chart_df = chart_df.dropna()
        chart_df = chart_df.sort_values(atlas_col, ascending=False)

        chart_melt = chart_df.melt(
            id_vars=ticker_col,
            value_vars=[atlas_col, bh_col],
            var_name='Strategy', value_name='Return %'
        )
        chart_melt['Strategy'] = chart_melt['Strategy'].map({
            atlas_col: 'ATLAS', bh_col: 'Buy & Hold'
        })

        fig_bt = px.bar(
            chart_melt,
            x=ticker_col, y='Return %', color='Strategy', barmode='group',
            color_discrete_map={'ATLAS': '#2ECC71', 'Buy & Hold': '#E74C3C'},
            title='ATLAS vs Buy & Hold Return — Test Period 2023–2025',
            text='Return %',
        )
        fig_bt.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_bt.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis_title='Ticker',
            yaxis_title='Return %',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

    # ── Win Rate bar chart ────────────────────────────────────────────────
    if winrate_col and ticker_col:
        st.markdown('**Win Rate per Ticker**')
        wr_df = bt_df[[ticker_col, winrate_col]].copy()
        wr_df[winrate_col] = pd.to_numeric(wr_df[winrate_col], errors='coerce')
        wr_df = wr_df.dropna().sort_values(winrate_col, ascending=False)

        fig_wr = px.bar(
            wr_df, x=ticker_col, y=winrate_col,
            color=winrate_col,
            color_continuous_scale=['#F85149', '#D29922', '#3FB950'],
            range_color=[40, 80],
            title='Win Rate by Ticker (Target: >52%)',
            text=wr_df[winrate_col].apply(lambda v: f'{v:.1f}%'),
        )
        fig_wr.add_hline(
            y=52, line_dash='dot', line_color='#D29922',
            annotation_text='52% target', annotation_position='top right'
        )
        fig_wr.update_traces(textposition='outside')
        fig_wr.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=60, b=20),
            coloraxis_showscale=False,
            xaxis_title='Ticker', yaxis_title='Win Rate %',
        )
        st.plotly_chart(fig_wr, use_container_width=True)

    # ── Confidence filter summary ─────────────────────────────────────────
    st.markdown('**Confidence Filter Status (from last backtest)**')
    filter_rows = []
    for _, row in bt_df.iterrows():
        if ticker_col:
            t = row.get(ticker_col, 'N/A')
            thresh = TICKER_THRESHOLDS.get(t, MIN_CONFIDENCE) if FILTER_ENABLED else MIN_CONFIDENCE
            if trades_col:
                tr = pd.to_numeric(row.get(trades_col, 0), errors='coerce')
                low_trades = pd.notna(tr) and tr < 20
            else:
                low_trades = False
            filter_rows.append({
                'Ticker':    t,
                'Threshold': f'{thresh:.1%}',
                'Status':    '⚠️ Low Trades' if low_trades else '✅ Valid',
            })
    if filter_rows:
        filt_df = pd.DataFrame(filter_rows)
        st.dataframe(filt_df, use_container_width=True, height=300)

    st.caption(
        f'📁 Source: data/backtest_report.csv — '
        f'last modified {datetime.fromtimestamp(BACKTEST_CSV.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}'
    )

else:
    st.warning(
        '📊 No backtest results found. Run the backtest first:\n\n'
        '```\npython backtest_report.py\n```'
    )

st.markdown('---')

# ── Portfolio Allocation ──────────────────────────────────────────────────────
st.subheader('📈 Optimal Portfolio Allocation')
st.markdown(
    'Portfolio weights computed via **PyPortfolioOpt** on 2015–2022 training data. '
    'Three strategies shown: Max Sharpe, Min Volatility, and **ATLAS Model** '
    '(tilts toward today\'s highest-confidence BULLISH tickers).'
)

weights_data = call_api('/portfolio/weights')

if 'error' not in weights_data and not weights_data.get('fallback'):
    ms  = weights_data.get('max_sharpe',  {})
    mv  = weights_data.get('min_vol',     {})
    atl = weights_data.get('atlas_model', {})

    pw1, pw2, pw3 = st.columns(3)
    pw1.metric('🔌 Max Sharpe — Exp. Return',    f"{ms.get('exp_return',  0):.1f}%" if ms.get('exp_return')  else 'N/A')
    pw2.metric('🟢 Min Volatility — Volatility', f"{mv.get('volatility',  0):.1f}%" if mv.get('volatility')  else 'N/A')
    pw3.metric('🟠 ATLAS Model — Sharpe Ratio',  f"{atl.get('sharpe',     0):.3f}"  if atl.get('sharpe')     else 'N/A')

    BLUE_P   = ['#1f77b4','#4e9cd1','#7abde8','#a6d4ef','#2196F3','#64B5F6','#0D47A1','#42A5F5','#1565C0','#90CAF9']
    GREEN_P  = ['#2ECC71','#27AE60','#1E8449','#58D68D','#A9DFBF','#145A32','#82E0AA','#196F3D','#52BE80','#0E6655']
    ORANGE_P = ['#E07B39','#F39C12','#D35400','#F8C471','#FAD7A0','#784212','#E59866','#A04000','#CA6F1E','#FDEBD0']

    def make_pie(weights_dict, title, colors):
        filtered = {k: v for k, v in weights_dict.items() if v > 0.001}
        if not filtered:
            return None
        fig = go.Figure(go.Pie(
            labels=list(filtered.keys()),
            values=[round(v * 100, 1) for v in filtered.values()],
            hole=0.35, textinfo='label+percent', textfont_size=11,
            marker=dict(colors=colors[:len(filtered)]),
        ))
        fig.update_layout(title=dict(text=title, font=dict(size=13)), height=300,
                          margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        return fig

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        f = make_pie(ms.get('weights', {}), '🔌 Max Sharpe', BLUE_P)
        if f: st.plotly_chart(f, use_container_width=True)
    with pc2:
        f = make_pie(mv.get('weights', {}), '🟢 Min Volatility', GREEN_P)
        if f: st.plotly_chart(f, use_container_width=True)
    with pc3:
        f = make_pie(atl.get('weights', {}), '🟠 ATLAS Model', ORANGE_P)
        if f: st.plotly_chart(f, use_container_width=True)

    with st.expander('📏 View Full Weight Table'):
        all_t = sorted(set(
            list(ms.get('weights', {}).keys()) +
            list(mv.get('weights', {}).keys()) +
            list(atl.get('weights', {}).keys())
        ))
        wt_df = pd.DataFrame([{
            'Ticker':      t,
            'Max Sharpe':  f"{ms.get('weights',  {}).get(t, 0)*100:.1f}%",
            'Min Vol':     f"{mv.get('weights',  {}).get(t, 0)*100:.1f}%",
            'ATLAS Model': f"{atl.get('weights', {}).get(t, 0)*100:.1f}%",
        } for t in all_t]).set_index('Ticker')
        st.dataframe(wt_df, use_container_width=True)

    st.caption(
        f"Weights from training period (2015–2022). "
        f"As of: {weights_data.get('as_of', 'N/A')}. "
        f"ATLAS Model uses today's live signal probabilities."
    )

elif weights_data.get('fallback'):
    st.info('🔺 Portfolio optimizer unavailable. Install with: `pip install PyPortfolioOpt --break-system-packages`')
else:
    st.info('📈 Portfolio weights loading... ensure API is running.')

# ── Paper Trading ─────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('💼 Paper Trading — Alpaca Portfolio')
portfolio_data = call_api('/portfolio')
if 'error' not in portfolio_data and 'detail' not in portfolio_data:
    acc = portfolio_data.get('account', {})
    pa1, pa2, pa3, pa4 = st.columns(4)
    pv     = acc.get('portfolio_value', 0)
    cash   = acc.get('cash', 0)
    equity = acc.get('equity', 0)
    pnl    = pv - 100_000
    pa1.metric('Portfolio Value', f"${pv:,.2f}")
    pa2.metric('Cash',            f"${cash:,.2f}")
    pa3.metric('Equity',          f"${equity:,.2f}")
    pa4.metric('Total P&L',       f"${pnl:,.2f}", delta=f"{(pnl/100_000)*100:.2f}%")

positions = portfolio_data.get('positions', [])
st.markdown(f"**Open Positions: {len(positions)}**")

if positions:
    pos_df = pd.DataFrame(positions)
    pos_df = pos_df[['ticker','qty','avg_entry_price','current_price','market_value','unrealized_pl','unrealized_plpc']]
    pos_df.columns = ['Ticker','Qty','Avg Entry','Current Price','Market Value','Unrealized P&L','P&L %']
    pos_df['Unrealized P&L'] = pos_df['Unrealized P&L'].apply(lambda v: f"${v:,.2f}")
    pos_df['P&L %']          = pos_df['P&L %'].apply(lambda v: f"{v:.2f}%")

    def color_pnl(val):
        v = val.replace('$','').replace('%','').replace(',','')
        try: return 'color: #3FB950' if float(v) >= 0 else 'color: #F85149'
        except: return ''

    st.dataframe(pos_df.style.map(color_pnl, subset=['Unrealized P&L','P&L %']), use_container_width=True)
else:
    st.info("No open positions — signals will open positions at next market open.")

history = portfolio_data.get('history', [])
if len(history) >= 2:
    hist_df = pd.DataFrame(history)
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date')
    fig_port = px.line(hist_df, x='date', y='portfolio_value',
                       title='Paper Portfolio Value Over Time',
                       color_discrete_sequence=['#3FB950'])
    fig_port.add_hline(y=100_000, line_dash='dot', line_color='#8B949E',
                       annotation_text='Starting Capital $100k')
    fig_port.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20),
                           xaxis_title='Date', yaxis_title='Portfolio Value ($)')
    st.plotly_chart(fig_port, use_container_width=True)
else:
    st.info("Portfolio chart will appear after 2+ days of trading data.")

st.markdown("**Recent Orders (from Alpaca)**")
orders_data = call_api('/orders/history?limit=20')

if isinstance(orders_data, list) and len(orders_data) > 0:
    ord_df    = pd.DataFrame(orders_data)
    disp_cols = ['ticker','side','filled_qty','filled_price','status','created_at']
    available = [c for c in disp_cols if c in ord_df.columns]
    ord_df    = ord_df[available]
    ord_df.columns = [c.replace('_',' ').title() for c in available]

    def color_side(val):
        if str(val).lower() == 'buy':  return 'color: #3FB950; font-weight: bold'
        if str(val).lower() == 'sell': return 'color: #F85149; font-weight: bold'
        return ''

    st.dataframe(ord_df.style.map(color_side, subset=['Side']), use_container_width=True)
else:
    st.info("No orders yet — orders will appear after first execution.")

st.markdown("---")
if st.button("🚀 Execute Signals Now (Paper Trade)"):
    with st.spinner("Executing signals..."):
        exec_result = call_api_post('/orders/execute')
        if 'error' not in exec_result:
            results = exec_result.get('results', [])
            buys  = [r for r in results if r.get('action') == 'BUY']
            sells = [r for r in results if r.get('action') == 'SELL']
            st.success(f"Execution complete: {len(buys)} BUY, {len(sells)} SELL orders submitted.")
            st.rerun()
        else:
            st.error(f"Execution failed: {exec_result['error']}")

# ── MLOps Panel ───────────────────────────────────────────────────────────────
st.markdown('---')
st.subheader('🔬 MLOps — Experiment Tracker & Drift Monitor')

st.markdown('**MLflow Experiment Runs**')
experiments_data = call_api('/mlops/experiments')

if 'error' not in experiments_data and experiments_data.get('experiments'):
    exps = experiments_data['experiments']
    exp_rows = []
    for e in exps:
        exp_rows.append({
            'Experiment': e.get('experiment', 'N/A'),
            'Latest Run': e.get('latest_run', 'N/A'),
            'Accuracy':   f"{e.get('accuracy', 0):.4f}",
            'F1 Score':   f"{e.get('f1_score', 0):.4f}",
            'Run ID':     e.get('run_id', 'N/A'),
        })
    st.dataframe(pd.DataFrame(exp_rows), use_container_width=True)
    st.caption(f"View full experiment details at: http://localhost:5000 ({len(exps)} experiments logged)")
else:
    st.info('📈 No MLflow runs yet. Retrain your models to start tracking: `python train_regime_models.py`')

st.markdown('---')
st.markdown('**Data Drift Monitor**')
st.markdown('Compares live market features against training data. Updated weekly every Sunday at 9:00 AM ET.')

drift_data = call_api('/mlops/drift/latest')

if 'error' not in drift_data and drift_data.get('drift_reports'):
    reports  = drift_data['drift_reports']
    n_alerts = drift_data.get('n_alerts', 0)

    if n_alerts > 0:
        st.error(f'🚨 RETRAIN ALERT: {n_alerts} ticker(s) have significant data drift: '
                 f'{drift_data.get("retrain_alerts", [])}. Run: python train_regime_models.py')
    else:
        st.success('✅ No drift alerts — models are aligned with current market conditions.')

    drift_rows = []
    for r in reports:
        if 'error' not in r:
            drift_ratio = r.get('drift_ratio', 0)
            alert       = r.get('retrain_alert', False)
            drift_rows.append({
                'Ticker':           r['ticker'],
                'Drift Ratio':      f"{drift_ratio:.1%}",
                'Drifted Features': r.get('n_drifted', 0),
                'Monitored':        r.get('n_monitored', 0),
                'Alert':            '🚨 RETRAIN' if alert else '✅ OK',
                'Report Date':      r.get('report_date', 'N/A'),
            })

    drift_df = pd.DataFrame(drift_rows)

    def color_alert(val):
        if 'RETRAIN' in str(val): return 'background-color:#2D0D0D;color:#F85149;font-weight:bold'
        if 'OK'      in str(val): return 'background-color:#0D2818;color:#3FB950'
        return ''

    st.dataframe(drift_df.style.map(color_alert, subset=['Alert']), use_container_width=True)

    with st.expander('🔍 View Full Drift Details'):
        for r in reports:
            if 'error' not in r and r.get('drifted_features'):
                st.markdown(f"**{r['ticker']}** — drifted features:")
                st.write(r['drifted_features'])
                if r.get('html_report'):
                    st.caption(f"HTML report: {r['html_report']}")
else:
    st.info('📈 No drift reports yet. Drift check runs on startup and every Sunday. Restart scheduler.py to trigger the first check.')

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('---')
st.markdown('*ATLAS — Algorithmic Trading with LLM-Augmented Signal Synthesis*')