# ATLAS — Algorithmic Trading with LLM-Augmented Signal Synthesis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=flat-square&logo=streamlit)
![GCP](https://img.shields.io/badge/Deployed-GCP-orange?style=flat-square&logo=google-cloud)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**A production-grade, end-to-end algorithmic trading platform — built across various development phases.**

[🌐 Live Dashboard](https://atlas-trading-platform.me) · [📂 Repository](https://github.com/JainulTrivedi55555/atlas-trading-platform)

</div>

---

## What is ATLAS?

Most portfolio projects show you can train a model in a Jupyter notebook. ATLAS is different. It's a complete system where data flows in automatically every day, signals are generated, trades are placed, and everything is monitored continuously — exactly how a real trading company would run it.

Every morning at **8:30 AM ET**, ATLAS automatically:
1. Fetches live OHLCV market data for **25 tickers** via yfinance
2. Scores financial news headlines with **FinBERT** (transformer-based sentiment)
3. Fuses price, sentiment, and macro signals into a single **confidence-filtered trading signal**
4. Executes **BUY/SELL orders** on a $100,000 simulated paper portfolio via Alpaca
5. Monitors for **data drift** and logs everything to MLflow

Everything is accessible live at **[atlas-trading-platform.me](https://atlas-trading-platform.me)**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
│  yfinance (OHLCV) · NewsAPI (headlines) · FRED (macro data)    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    FEATURE ENGINEERING                          │
│  54 technical features · 11 sentiment features · 19 macro      │
│  features = 69 total · enforced via build_live_features()       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     SIGNAL GENERATION                           │
│  HMM Regime Detection → Bull / Bear / HighVol                   │
│  XGBoost · LightGBM · PyTorch LSTM                             │
│  Confidence filter (threshold: 0.30)                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    EXECUTION & MONITORING                       │
│  Alpaca Paper Trading · SQLite Cache · MLflow · Evidently AI   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  PRODUCTION INFRASTRUCTURE                      │
│  FastAPI · Streamlit · APScheduler · Nginx · GCP EC2           │
│  GitHub Actions CI/CD · Systemd · SSL (Let's Encrypt)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Best Model AUC | **0.817** (PyTorch LSTM) |
| Total Features | **69** (54 technical + 11 sentiment + 19 macro + 4 modality) |
| Live Tickers | **25** (10 core stocks + 10 growth/value + 5 ETFs) |
| News Articles Scored | **740** with FinBERT |
| ML Models | **3** (XGBoost, LightGBM, PyTorch LSTM) |
| Market Regimes | **4** (Bull, Bear, High Volatility, Sideways via HMM) |
| Paper Portfolio | **$100,000** simulated via Alpaca |
| MLflow Runs Logged | **20+** experiment runs |
| Backtesting Win Rate | **59.9%** average across 25 tickers |
| Development Phases | **16** phases completed |
| Confidence Threshold | **0.30** minimum for signal execution |
| Scheduler Trigger | **8:30 AM ET** daily, Mon–Fri |
| Deployment | **GCP e2-standard-2** with Nginx + SSL |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Backend** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit + Plotly |
| **ML Models** | XGBoost, LightGBM, PyTorch LSTM |
| **Sentiment** | FinBERT (ProsusAI/finbert, ~440MB) |
| **Regime Detection** | Hidden Markov Model (hmmlearn) |
| **RAG System** | LangChain + FAISS over SEC 10-K/10-Q filings |
| **Risk Engine** | PyPortfolioOpt, VaR, CVaR, Sharpe, Sortino |
| **Backtesting** | Custom vectorised engine |
| **Broker** | Alpaca Paper Trading API |
| **Scheduler** | APScheduler (8:30 AM ET daily) |
| **MLOps** | MLflow experiment tracking + Evidently AI drift monitoring |
| **Cache** | SQLite (live features, sentiment, orders) |
| **Data Sources** | yfinance, NewsAPI, FRED |
| **Infrastructure** | GCP Compute Engine, Nginx, Systemd, Let's Encrypt SSL |
| **CI/CD** | GitHub Actions (auto-deploy on push to main) |
| **Environment** | Conda (fintech), Python 3.10 |

---

## Project Structure

```
atlas-trading-platform/
├── app/
│   ├── main.py                    # FastAPI app — all 17+ endpoints
│   └── api/
├── src/
│   ├── data/
│   │   ├── live_pipeline.py       # yfinance fetcher + build_live_features()
│   │   └── live_cache.py          # SQLite cache with staleness detection
│   ├── models/
│   │   ├── regime_detector.py     # HMM regime classifier
│   │   ├── regime_trainer.py      # Per-regime XGBoost + LightGBM training
│   │   └── regime_predictor.py    # Inference-time model selection
│   ├── sentiment/
│   │   ├── news_fetcher.py        # NewsAPI integration
│   │   ├── finbert_scorer.py      # FinBERT sentiment inference
│   │   ├── sentiment_cache.py     # SQLite sentiment cache
│   │   └── signal_fusion.py      # 70% price + 30% sentiment fusion
│   ├── broker/
│   │   ├── alpaca_broker.py       # Alpaca connection + account info
│   │   ├── order_executor.py      # BUY/SELL signal execution
│   │   └── position_tracker.py   # Portfolio snapshot tracking
│   ├── rag/
│   │   ├── sec_collector.py       # SEC 10-K/10-Q filing collector
│   │   ├── chunker.py             # Document chunking
│   │   ├── embedder.py            # FAISS vector store
│   │   ├── retriever.py           # Evidence retrieval
│   │   └── generator.py          # LLM market brief generator
│   ├── risk/
│   │   ├── risk_engine.py         # VaR, CVaR, Sharpe, Sortino, Max Drawdown
│   │   └── portfolio_optimizer.py # PyPortfolioOpt — Max Sharpe, Min Vol
│   ├── backtesting/
│   │   ├── backtest_engine.py     # Custom vectorised backtester
│   │   └── strategy.py           # Ensemble XGBoost 50% + LightGBM 50%
│   ├── mlops/
│   │   ├── mlflow_logger.py       # MLflow experiment tracking
│   │   ├── drift_monitor.py       # Evidently AI drift reports
│   │   ├── drift_detector.py      # PSI-based drift detection
│   │   └── registry.py           # MLflow model registry
│   ├── macro/
│   │   └── macro_features.py     # 19 macro features from FRED
│   └── utils/
│       ├── config.py              # 25 tickers, paths, constants
│       └── logger.py             # Rotating file handler
├── dashboards/
│   └── streamlit_dashboard.py    # Full Streamlit dashboard
├── notebooks/
│   ├── 07_fusion/                 # 3-modal feature fusion experiments
│   ├── 08_rag/                    # RAG pipeline over SEC filings
│   ├── 09_risk/                   # Risk engine analysis
│   └── 10_backtesting/           # Walk-forward backtest
├── deploy/
│   ├── nginx.conf                 # Nginx reverse proxy config
│   ├── atlas-api.service          # Systemd FastAPI service
│   └── atlas-streamlit.service   # Systemd Streamlit service
├── .github/workflows/
│   └── deploy.yml                 # GitHub Actions CI/CD → GCP
├── scheduler.py                   # APScheduler — daily pipeline
├── train_regime_models.py         # Full regime model training script
├── environment.yml                # Conda environment export
└── requirements.txt               # pip requirements for deployment
```

---

## The Data Pipeline

The pipeline is the foundation everything else is built on. Every morning before market open, the scheduler fires and runs three sequential stages.

**Stage 1 — Live Data & Feature Engineering**

The scheduler fetches ~252 days of OHLCV history for all 25 tickers from yfinance. From raw price data, 54 engineered features are computed including RSI, MACD, Bollinger Bands, Stochastic oscillator, Chaikin Money Flow, ATR, volatility ratios, and momentum indicators. The critical constraint is that all features must be computed identically to how they were computed during model training — even one feature in the wrong position produces silently wrong predictions. `build_live_features()` enforces this as a hard constraint, always outputting exactly a (1, N) matrix validated against expected shape.

Features are then stored in a SQLite cache (`atlas_live.db`) with a staleness threshold of 26 hours. The cache reduces API response latency from 2-3 seconds to milliseconds.

**Stage 2 — Sentiment Pipeline**

After the live data stage, the scheduler fetches financial news headlines from NewsAPI for each ticker and scores them with FinBERT (`ProsusAI/finbert`) — a BERT-based transformer fine-tuned on financial text. FinBERT scores each headline with positive, negative, and neutral probabilities. 11 sentiment features are engineered from these scores including rolling averages, sentiment momentum, and sentiment volatility. Results are cached in `atlas_sentiment.db`.

**Stage 3 — Broker Execution**

After sentiment scoring, the order executor reads the regime-aware signals for all 25 tickers, applies the 0.30 confidence threshold filter, and submits market orders to Alpaca's paper trading API. Position sizing uses a fixed fraction of the $100K portfolio per signal. The executor is market-hours aware — orders submitted before 9:30 AM ET are queued as DAY orders and fill at the opening price.

---

## The Machine Learning Stack

**Three model families** trained with Independent Cross-Validation (walk-forward, no data leakage):

- **XGBoost** — tuned with Optuna (50 trials), gradient boosting on 69 features
- **LightGBM** — tuned with Optuna (50 trials), faster tree boosting
- **PyTorch LSTM** — 60-day lookback window, treats feature sequence as time series

**Hidden Markov Model Regime Detection** classifies the market into Bull, Bear, High Volatility, and Sideways states based on price returns and volatility patterns. A separate model is trained for each regime per ticker — the correct model is selected automatically at inference time. This is critical because features that predict direction in a Bull market are completely different from those that matter in a Bear market.

**SHAP Analysis** confirmed genuine feature informativeness:
- Chaikin Money Flow ranked #1 for tree models
- RSI ranked #1 for the LSTM
- Bollinger Band Width and Stochastic K appeared consistently across all three model families

**An honest finding about sentiment:** When FinBERT sentiment features were first added to XGBoost and LightGBM, AUC dropped slightly. The reason was a price proxy problem — news is often written after price moves, so sentiment was correlated with technical indicators already in the model. Feature fusion needs to be done carefully. The macro features added genuine value because they capture information orthogonal to price.

---

## Signal Generation & Fusion

The final signal fuses two towers:

```
Final Signal = (0.70 × Price Model Prob_Up) + (0.30 × FinBERT Sentiment Score)
```

If the fused probability exceeds 0.50 → **BULLISH**. Below 0.50 → **BEARISH**. Signals below the 0.30 confidence threshold are suppressed entirely — the system holds rather than acting on weak predictions. This significantly reduces false positives.

---

## Risk Engine & Backtesting

The risk engine computes portfolio-level metrics on the held-out test set (2023–2025):

- Value at Risk (VaR 95%), Conditional VaR (CVaR)
- Sharpe Ratio, Sortino Ratio, Max Drawdown
- Portfolio optimization via PyPortfolioOpt — Max Sharpe, Min Volatility, and ATLAS Model strategies

The custom vectorised backtester runs the ensemble strategy (XGBoost 50% + LightGBM 50%) across all 25 tickers on data never seen during training:

| Metric | Result |
|--------|--------|
| Average Win Rate | 59.9% |
| Average Profit Factor | 1.67 |
| Average Max Drawdown | -8.9% |
| Tickers with ≥20 trades | 20/25 |

Context: 2023–2025 was an extreme bull market (NVDA +333%, TSLA +167%). Any strategy that sits in cash loses to buy-and-hold in this environment. The 59.9% win rate and 1.67 profit factor confirm the model makes correct directional decisions — it just misses some bull runs by design.

---

## RAG System

ATLAS includes a Retrieval Augmented Generation system over SEC filings:

- SEC 10-K and 10-Q filings collected via EDGAR API
- Documents chunked and embedded using sentence-transformers
- FAISS vector store for fast similarity search
- LangChain orchestration for retrieval and generation
- Generates grounded market briefs with evidence citations from actual filings

---

## MLOps Infrastructure

**MLflow** tracks every model training run:
- Hyperparameters, AUC scores, feature importance data, model artifacts
- Compare 20+ runs side by side in the MLflow UI
- Roll back to any previous version if a new model underperforms

**Evidently AI** runs weekly data drift monitoring:
- Compares live feature distributions against training data distributions
- Flags when features drift significantly (PSI-based detection)
- Automatically alerts when it's time to retrain
- Runs every Sunday at 9:00 AM ET via APScheduler

---

## Production Deployment

ATLAS runs 24/7 on **Google Cloud Platform** (e2-standard-2, Ubuntu 22.04):

```
Internet → GCP Static IP :80
         → Nginx Reverse Proxy
         → Streamlit Dashboard (:8501)   → https://atlas-trading-platform.me
         → FastAPI Backend (:8000)       → https://atlas-trading-platform.me/api
8:30 AM ET → Systemd Timer → scheduler.py (data + sentiment + broker)
git push main → GitHub Actions → SSH to GCP → git pull → restart services
```

**Systemd services** ensure auto-restart after every reboot:
- `atlas-api.service` — FastAPI + Uvicorn (2 workers)
- `atlas-streamlit.service` — Streamlit dashboard
- `atlas-scheduler.timer` — Daily pipeline trigger

**GitHub Actions CI/CD** — every push to main automatically deploys to GCP in ~30 seconds.

**SSL** via Let's Encrypt (auto-renewing) — `https://atlas-trading-platform.me`

---

## Running Locally

```bash
# Clone and set up environment
git clone https://github.com/JainulTrivedi55555/atlas-trading-platform.git
cd atlas-trading-platform
conda env create -f environment.yml
conda activate fintech

# Configure API keys
cp .env.example .env
# Add: ALPACA_API_KEY, ALPACA_SECRET_KEY, NEWSAPI_KEY

# Terminal 1 — FastAPI backend (start first)
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run dashboards/streamlit_dashboard.py

# Terminal 3 — Scheduler (start last)
python scheduler.py
```

**First run:** You need to train the models before the API serves live signals:
```bash
python fix_features_and_retrain.py   # regenerate splits + retrain base models
python train_regime_models.py        # train regime-specific models
python backtest_report.py            # generate backtest results
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/signal/{ticker}` | GET | Live signal with confidence + regime |
| `/signal/regime/{ticker}` | GET | Regime-aware signal |
| `/signals/all` | GET | All 25 tickers in one call |
| `/signal/fused/{ticker}` | GET | Price + sentiment fused signal |
| `/sentiment/{ticker}` | GET | FinBERT sentiment score |
| `/sentiment/all` | GET | All tickers sentiment |
| `/portfolio` | GET | Live paper portfolio |
| `/orders/history` | GET | Order history |
| `/status/pipeline` | GET | Last refresh time per ticker |
| `/mlops/experiments` | GET | MLflow experiment runs |
| `/mlops/drift/latest` | GET | Latest drift report summary |
| `/health` | GET | System health check |

---

## Ticker Universe

| Category | Tickers |
|----------|---------|
| Tech | AAPL, MSFT, GOOGL, AMZN, META, NVDA, AMD, CRM, NFLX, ORCL |
| Financial | JPM, GS, BAC |
| Growth/Value | TSLA, UBER, WMT, JNJ, XOM, LLY, V |
| ETFs (context signals) | SPY, QQQ, IWM, GLD, TLT |

The ETFs are not just standalone signals — they serve as risk-on/risk-off context. When TLT and GLD are rising together, that signals a risk-off environment which modulates how aggressively the system acts on individual stock signals.

---

## What I Learned

Building ATLAS across all phases produced several non-obvious insights that only come from actually shipping a system:

**Feature fusion must be deliberate.** Adding more data modalities does not automatically improve AUC. FinBERT sentiment features hurt tree model performance because news sentiment is a lagging indicator — it correlates with price moves that the technical indicators already capture. Macro features added genuine value because they are orthogonal to price.

**Production bugs are invisible in notebooks.** A Windows path bug in the MLflow logger wrote artifacts to a Linux temp path that was silently wiped on reboot. A relative path SQLite connection worked locally but failed on GCP. A corrupted HuggingFace model cache broke FinBERT with no error message. These are the kinds of issues that separate a real system from a tutorial.

**Feature alignment is the most critical constraint.** The single most dangerous bug in an ML trading system is a silent feature mismatch — where the live feature vector has different columns or ordering than the training data. `build_live_features()` is the most important function in the entire codebase because it enforces this contract on every inference call.

**Independent Cross-Validation is non-negotiable for time series.** Standard k-fold shuffles future data into the training window. Walk-forward validation always validates on data that comes after the training window. This is the difference between a model that looks good in backtesting and one that actually works.

---

## What I Would Do Differently

1. **Feature store** — Use Feast to decouple feature computation from model serving. Currently `build_live_features()` is tightly coupled to the API layer.
2. **Docker** — Containerize all four processes so deployment becomes a single `docker-compose up` command instead of four terminals.
3. **Time-series database** — Replace SQLite with TimescaleDB for historical signal tracking. Currently signals are generated fresh each day but not persisted historically.
4. **Async inference** — Make the signal endpoint fully async to handle concurrent requests without blocking on model inference.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by **Jainul Trivedi** · [GitHub](https://github.com/JainulTrivedi55555) · [Live Demo](https://atlas-trading-platform.me)

*ATLAS is a paper trading system for educational and portfolio purposes. It does not constitute financial advice and uses simulated money only.*

</div>