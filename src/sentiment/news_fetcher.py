"""
News Fetcher
Fetches recent financial news headlines for each ticker from NewsAPI.
Returns a list of clean headline strings ready for FinBERT scoring.
"""
import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger('atlas.news_fetcher')

# Ticker to full company name mapping — NewsAPI searches by name, not symbol
TICKER_NAMES = {
    'AAPL':  'Apple',
    'MSFT':  'Microsoft',
    'GOOGL': 'Google Alphabet',
    'AMZN':  'Amazon',
    'META':  'Meta Platforms Facebook',
    'JPM':   'JPMorgan Chase',
    'GS':    'Goldman Sachs',
    'BAC':   'Bank of America',
    'NVDA':  'NVIDIA',
    'TSLA':  'Tesla',
}

MAX_HEADLINES  = 10   # Articles per ticker per fetch
LOOKBACK_DAYS  = 2    # How many days back to search

def get_newsapi_client():
    """Initialise NewsAPI client using key from .env"""
    from newsapi import NewsApiClient
    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        raise ValueError(
            'NEWSAPI_KEY not found in .env — '
            'get a free key at https://newsapi.org/register'
        )
    return NewsApiClient(api_key=api_key)

def fetch_headlines(ticker: str, max_articles: int = MAX_HEADLINES) -> list[str]:
    """
    Fetch recent news headlines for a ticker.
    Returns list of headline strings — empty list if API fails.
    """
    company_name = TICKER_NAMES.get(ticker.upper(), ticker)
    from_date = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime('%Y-%m-%d')
    to_date   = datetime.now().strftime('%Y-%m-%d')
    
    try:
        client   = get_newsapi_client()
        response = client.get_everything(
            q=company_name,
            from_param=from_date,
            to=to_date,
            language='en',
            sort_by='relevancy',
            page_size=max_articles,
        )
        
        if response['status'] != 'ok':
            logger.warning(f'{ticker}: NewsAPI status={response["status"]}')
            return []
            
        articles   = response.get('articles', [])
        headlines  = []
        for article in articles:
            title       = article.get('title', '') or ''
            description = article.get('description', '') or ''
            # Prefer title + description combined for richer context
            text = f"{title}. {description}".strip('. ')
            if text and len(text) > 20:   # Skip empty/stub articles
                headlines.append(text[:512])  # FinBERT max input length
                
        logger.info(f'{ticker}: Fetched {len(headlines)} headlines')
        return headlines
    except Exception as e:
        logger.error(f'{ticker}: News fetch failed — {e}')
        return []

def fetch_all_tickers(tickers: list = None) -> dict:
    """
    Fetch headlines for all tickers.
    Results dict: {ticker: [headline1, headline2, ...]}
    """
    if tickers is None:
        tickers = list(TICKER_NAMES.keys())
        
    results = {}
    for i, ticker in enumerate(tickers):
        results[ticker] = fetch_headlines(ticker)
        if i < len(tickers) - 1:
            time.sleep(0.5)   # 500ms between requests
            
    total = sum(len(v) for v in results.values())
    logger.info(f'News fetch complete: {total} headlines across {len(tickers)} tickers')
    return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('Testing NewsAPI fetch...')
    headlines = fetch_headlines('AAPL', max_articles=5)
    print(f'AAPL headlines: {len(headlines)}')
    for i, h in enumerate(headlines, 1):
        print(f'  {i}. {h[:100]}...')