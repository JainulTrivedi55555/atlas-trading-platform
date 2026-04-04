"""
FinBERT Sentiment Scorer
Scores financial news headlines using ProsusAI/finbert.

KEY FIX: Model is loaded ONCE at module level (global singleton).
Previous version reloaded the model for every single ticker, causing
the DLL initialization error on Windows (c10.dll fails on repeated loads).
"""
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger('atlas.finbert')

# ── Load FinBERT model ONCE at module import ──────────────────────────────────
# This is the critical fix: loading torch/transformers repeatedly in the same
# process causes WinError 1114 (DLL init failure) on Windows.
_PIPELINE = None

def _get_pipeline():
    """Lazy-load FinBERT pipeline — loads once, reused for all tickers."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    logger.info('Loading FinBERT model: ProsusAI/finbert')
    logger.info('First run will download ~440MB — subsequent runs are instant.')

    try:
        from transformers import pipeline
        _PIPELINE = pipeline(
            'text-classification',
            model='ProsusAI/finbert',
            tokenizer='ProsusAI/finbert',
            top_k=None,          # return all 3 label scores
            truncation=True,
            max_length=512,
        )
        logger.info('FinBERT model loaded successfully.')
        return _PIPELINE
    except Exception as e:
        logger.error(f'Failed to load FinBERT model: {e}')
        return None


def score_headlines(headlines: list[str]) -> list[dict]:
    """
    Score a list of headlines using FinBERT.

    Returns list of dicts with keys: label, positive, negative, neutral
    Falls back to neutral (0.5) if model unavailable.
    """
    if not headlines:
        return []

    pipe = _get_pipeline()
    if pipe is None:
        logger.warning('FinBERT unavailable — returning neutral scores')
        return [{'label': 'neutral', 'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                for _ in headlines]

    results = []
    try:
        # Score all headlines in one batch — much faster than one-by-one
        raw_outputs = pipe(headlines, batch_size=8)

        for output in raw_outputs:
            # output is a list of {label, score} for all 3 classes
            scores = {item['label'].lower(): item['score'] for item in output}
            pos = scores.get('positive', 0.0)
            neg = scores.get('negative', 0.0)
            neu = scores.get('neutral',  0.0)

            # Label = highest scoring class
            label = max(scores, key=scores.get)
            results.append({
                'label':    label,
                'positive': pos,
                'negative': neg,
                'neutral':  neu,
            })

    except Exception as e:
        logger.error(f'Batch scoring failed: {e}')
        # Return neutral fallback for each headline
        results = [{'label': 'neutral', 'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                   for _ in headlines]

    return results


def aggregate_sentiment(headlines: list[str]) -> dict:
    """
    Aggregate FinBERT scores across all headlines for one ticker.

    Returns:
        sentiment_score:  float 0-1 (0=bearish, 0.5=neutral, 1=bullish)
        sentiment_label:  'bullish' | 'bearish' | 'neutral'
        positive_pct:     fraction of headlines that are positive
        negative_pct:     fraction of headlines that are negative
        neutral_pct:      fraction of headlines that are neutral
        n_headlines:      number of headlines scored
    """
    if not headlines:
        return {
            'sentiment_score':  0.5,
            'sentiment_label':  'neutral',
            'positive_pct':     0.0,
            'negative_pct':     0.0,
            'neutral_pct':      1.0,
            'n_headlines':      0,
        }

    scored = score_headlines(headlines)
    n      = len(scored)

    # Count labels
    pos_count = sum(1 for s in scored if s['label'] == 'positive')
    neg_count = sum(1 for s in scored if s['label'] == 'negative')
    neu_count = sum(1 for s in scored if s['label'] == 'neutral')

    pos_pct = pos_count / n
    neg_pct = neg_count / n
    neu_pct = neu_count / n

    # Average raw scores
    avg_pos = sum(s['positive'] for s in scored) / n
    avg_neg = sum(s['negative'] for s in scored) / n

    # Sentiment score: 0 = fully bearish, 0.5 = neutral, 1 = fully bullish
    sentiment_score = 0.5 + (avg_pos - avg_neg) / 2

    # Label based on dominant sentiment
    if pos_pct > 0.5:
        label = 'bullish'
    elif neg_pct > 0.5:
        label = 'bearish'
    elif pos_pct > neg_pct + 0.1:
        label = 'bullish'
    elif neg_pct > pos_pct + 0.1:
        label = 'bearish'
    else:
        label = 'neutral'

    return {
        'sentiment_score': round(sentiment_score, 4),
        'sentiment_label': label,
        'positive_pct':    round(pos_pct, 4),
        'negative_pct':    round(neg_pct, 4),
        'neutral_pct':     round(neu_pct, 4),
        'n_headlines':     n,
    }