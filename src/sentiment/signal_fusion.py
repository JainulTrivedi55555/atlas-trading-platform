"""
Signal Fusion
0.78
0.48
0.539
0.528
0.508
Combines price model signals with FinBERT sentiment scores.
BEARISH (high confidence)
BULLISH (weak — caution)
BULLISH (news overriding)
BULLISH (barely — low conf)
Fusion formula: fused = PRICE_WEIGHT * price_prob_up + SENTIMENT_WEIGHT * sentiment_score
"""

import logging
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.sentiment.sentiment_cache import load_sentiment, load_all_sentiment

logger = logging.getLogger('atlas.signal_fusion')

# Fusion weights — configurable
PRICE_WEIGHT     = 0.70   # Weight given to price model probability
SENTIMENT_WEIGHT = 0.30   # Weight given to FinBERT sentiment score

# Must sum to 1.0
TICKERS = ['AAPL','MSFT','GOOGL','AMZN','META',
           'JPM','GS','BAC','NVDA','TSLA']


def fuse_signal(price_signal: dict, sentiment: dict | None) -> dict:
    """
    Fuse a price model signal with a sentiment score.

    Args:
        price_signal: Dict from get_signal() or get_regime_signal()
                      Must contain: ticker, signal, prob_up, confidence, as_of_date
        sentiment:    Dict from load_sentiment() or None if unavailable

    Returns:
        Fused signal dict with all original fields plus fusion metadata
    """

    ticker        = price_signal.get('ticker', 'UNKNOWN')
    price_prob_up = price_signal.get('prob_up', 0.5)

    # If no sentiment available, return price signal unchanged with a flag
    if sentiment is None or sentiment.get('n_headlines', 0) == 0:
        result = dict(price_signal)

        result.update({
            'fused':              False,
            'fusion_reason':      'No sentiment data available — using price signal only',
            'sentiment_score':    None,
            'sentiment_label':    None,
            'fused_prob_up':      price_prob_up,
            'fused_signal':       price_signal.get('signal', 'BEARISH'),
            'fused_confidence':   price_signal.get('confidence', 0.5),
        })

        return result

    sentiment_score = sentiment.get('sentiment_score', 0.5)

    # Core fusion formula
    fused_prob_up = (PRICE_WEIGHT * price_prob_up) + (SENTIMENT_WEIGHT * sentiment_score)
    fused_prob_up = round(fused_prob_up, 4)

    fused_signal     = 'BULLISH' if fused_prob_up >= 0.50 else 'BEARISH'
    fused_confidence = abs(fused_prob_up - 0.50) * 2

    # Detect signal divergence
    price_signal_str = price_signal.get('signal', 'BEARISH')
    sent_signal      = 'BULLISH' if sentiment_score >= 0.50 else 'BEARISH'

    divergence = price_signal_str != sent_signal

    result = dict(price_signal)

    result.update({
        'fused':              True,
        'price_prob_up':      round(price_prob_up, 4),
        'price_signal':       price_signal_str,
        'price_weight':       PRICE_WEIGHT,
        'sentiment_score':    round(sentiment_score, 4),
        'sentiment_label':    sentiment.get('sentiment_label', 'neutral'),
        'sentiment_signal':   sent_signal,
        'sentiment_weight':   SENTIMENT_WEIGHT,
        'n_headlines':        sentiment.get('n_headlines', 0),
        'fused_prob_up':      fused_prob_up,
        'fused_prob_down':    round(1.0 - fused_prob_up, 4),
        'fused_signal':       fused_signal,
        'fused_confidence':   round(fused_confidence, 4),
        'signal_divergence':  divergence,
        'fusion_reason': (
            f'Price({PRICE_WEIGHT}) x {price_prob_up:.3f} + '
            f'Sentiment({SENTIMENT_WEIGHT}) x {sentiment_score:.3f} = {fused_prob_up:.3f}'
        ),
    })

    return result


def get_fused_signal(ticker: str, price_signal: dict,
                     score_date: date = None) -> dict:
    """
    Load cached sentiment and fuse with price signal for one ticker.

    Args:
        ticker:       Stock ticker
        price_signal: From get_signal() or get_regime_signal()
        score_date:   Date to use for sentiment lookup (default: today)

    Returns:
        Fused signal dict
    """

    sentiment = load_sentiment(ticker, score_date)

    return fuse_signal(price_signal, sentiment)


def get_all_fused_signals(model_type: str = 'xgboost') -> list:
    """
    Get fused signals for all 10 tickers.

    Loads price signals from regime predictor + sentiment from cache.

    Returns: List of fused signal dicts
    """

    try:
        from src.models.regime_predictor import get_all_regime_signals
        from app.atlas_engine import get_live_features_for_ticker

        price_signals = get_all_regime_signals(model_type)

    except Exception as e:
        logger.error(f'Could not load price signals: {e}')
        return []

    results = []

    for ps in price_signals:

        if 'error' in ps:
            results.append(ps)
            continue

        ticker  = ps.get('ticker', '')
        fused   = get_fused_signal(ticker, ps)

        results.append(fused)

    return results


if __name__ == '__main__':

    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing signal fusion...")

    # Simulate a price signal
    mock_price = {
        'ticker': 'AAPL',
        'signal': 'BULLISH',
        'prob_up': 0.65,
        'prob_down': 0.35,
        'confidence': 0.65,
        'as_of_date': '2026-03-07',
    }

    # Simulate a sentiment result
    mock_sentiment = {
        'sentiment_score': 0.82,
        'sentiment_label': 'bullish',
        'n_headlines': 8,
        'positive_pct': 0.75,
        'negative_pct': 0.125,
    }

    fused = fuse_signal(mock_price, mock_sentiment)

    print(f"Price signal:  {mock_price['signal']} ({mock_price['prob_up']:.3f})")
    print(f"Sentiment:     {mock_sentiment['sentiment_label']} ({mock_sentiment['sentiment_score']:.3f})")
    print(f"Fused signal:  {fused['fused_signal']} ({fused['fused_prob_up']:.3f})")
    print(f"Formula:       {fused['fusion_reason']}")
    print(f"Divergence:    {fused['signal_divergence']}")