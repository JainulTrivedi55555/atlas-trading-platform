import requests, pandas as pd, time
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import NEWS_API_KEY, TICKERS, RAW_DIR

def fetch_news(ticker, days_back=30):
    """Fetch news for one ticker. Free tier: last 30 days only."""
    end = datetime.now()
    start = end - timedelta(days=days_back)
    params = {
        'q':        f'{ticker} stock',
        'from':     start.strftime('%Y-%m-%d'),
        'to':       end.strftime('%Y-%m-%d'),
        'language': 'en', 
        'sortBy': 'publishedAt',
        'pageSize': 100,  
        'apiKey': NEWS_API_KEY,
    }
    
    r = requests.get('https://newsapi.org/v2/everything', params=params)
    
    if r.status_code != 200:
        print(f'Error {ticker}: {r.json().get("message")}')
        return []
    
    return [
        {
            'ticker': ticker,
            'published_at': a.get('publishedAt'),
            'title':        a.get('title', ''),
            'description':  a.get('description', ''),
            'source':       a.get('source', {}).get('name', ''),
            'url':          a.get('url', '')
        }
        for a in r.json().get('articles', [])
    ]

def download_all_news():
    save_dir = RAW_DIR / 'news'
    save_dir.mkdir(parents=True, exist_ok=True)
    all_articles = []
    
    for ticker in TICKERS:
        arts = fetch_news(ticker)
        all_articles.extend(arts)
        print(f'  {ticker}: {len(arts)} articles')
        time.sleep(1)   # Respect rate limits
        
    df = pd.DataFrame(all_articles)
    if not df.empty:
        df['published_at'] = pd.to_datetime(df['published_at'])
        df.sort_values('published_at', ascending=False).to_csv(
            save_dir / 'news_raw.csv', index=False
        )
    
    print(f'Total: {len(df)} articles saved')
    return df

if __name__ == '__main__':
    news = download_all_news()
    if not news.empty:
        print(news[['ticker','published_at','title']].head(10))