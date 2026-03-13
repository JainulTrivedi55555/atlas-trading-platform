import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR, RAW_DIR  

MODEL_NAME = 'ProsusAI/finbert'
BATCH_SIZE = 16 


NEWS_PATH = RAW_DIR / 'news' / 'news_raw.csv'


def load_finbert():
    """Load FinBERT model and tokenizer."""
    print('Loading FinBERT...')
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    device = torch.device('cpu')
    model  = model.to(device)
    print('FinBERT loaded on CPU')
    return tokenizer, model, device


def score_batch(texts, tokenizer, model, device):
    """Score a batch of texts. Returns list of dicts."""
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs  = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    labels = ['positive', 'negative', 'neutral']
    results = []
    for p in probs:
        pred = labels[p.argmax()]
        results.append({
            'sentiment':       pred,
            'positive_score':  float(p[0]),
            'negative_score':  float(p[1]),
            'neutral_score':   float(p[2]),
            'sentiment_score': float(p[0] - p[1]),  # Range: -1 to +1
            'confidence':      float(p.max())
        })
    return results


def score_all_articles():
    """Score all news articles with FinBERT."""

    # FIXED: Load from correct path
    print(f'Loading news from: {NEWS_PATH}')
    if not NEWS_PATH.exists():
        raise FileNotFoundError(
            f'News file not found at {NEWS_PATH}. '
            f'Check your data/raw/news/ folder.'
        )

    news_df = pd.read_csv(NEWS_PATH)
    print(f'Loaded {len(news_df)} articles')
    print(f'Columns: {news_df.columns.tolist()}')

    # FIXED: Auto-detect text column from your actual CSV
    text_col = None
    for col in ['headline', 'title', 'text', 'content',
                 'description', 'summary', 'body']:
        if col in news_df.columns:
            text_col = col
            break
    if text_col is None:
        # Fall back to first string column
        for col in news_df.columns:
            if news_df[col].dtype == object:
                text_col = col
                break
    print(f'Using text column: {text_col}')

    # FIXED: Auto-detect ticker column
    ticker_col = None
    for col in ['ticker', 'symbol', 'stock', 'company']:
        if col in news_df.columns:
            ticker_col = col
            break
    if ticker_col:
        print(f'Using ticker column: {ticker_col}')
        print(f'Tickers found: {news_df[ticker_col].unique().tolist()}')
    else:
        print('No ticker column found — scoring all articles together')

    # FIXED: Auto-detect date column
    date_col = None
    for col in ['date', 'published_at', 'datetime',
                 'timestamp', 'published', 'created_at']:
        if col in news_df.columns:
            date_col = col
            break
    if date_col:
        print(f'Using date column: {date_col}')

    # Clean text
    news_df[text_col] = news_df[text_col].fillna('').astype(str)
    news_df = news_df[news_df[text_col].str.len() > 10].reset_index(drop=True)
    print(f'Articles after cleaning: {len(news_df)}')

    # Load FinBERT
    tokenizer, model, device = load_finbert()

    # Score in batches
    all_results = []
    total = len(news_df)
    print(f'\nScoring {total} articles in batches of {BATCH_SIZE}...')

    for i in range(0, total, BATCH_SIZE):
        batch  = news_df[text_col].iloc[i:i+BATCH_SIZE].tolist()
        scores = score_batch(batch, tokenizer, model, device)
        all_results.extend(scores)
        if (i // BATCH_SIZE) % 5 == 0:
            pct = min(i + BATCH_SIZE, total)
            print(f'  Processed {pct}/{total} articles '
                  f'({100*pct/total:.0f}%)')

    # Add scores to dataframe
    scores_df = pd.DataFrame(all_results)
    result_df = pd.concat(
        [news_df.reset_index(drop=True), scores_df], axis=1
    )

    # Save to processed folder
    save_path = PROCESSED_DIR / 'sentiment_scores.csv'
    result_df.to_csv(save_path, index=False)
    print(f'\nSaved to {save_path}')

    # Summary
    print(f'\nSentiment Distribution:')
    print(result_df['sentiment'].value_counts())
    print(f'\nMean sentiment score: {result_df["sentiment_score"].mean():.4f}')
    print(f'Positive articles:    {(result_df["sentiment"]=="positive").sum()}')
    print(f'Negative articles:    {(result_df["sentiment"]=="negative").sum()}')
    print(f'Neutral articles:     {(result_df["sentiment"]=="neutral").sum()}')

    return result_df


if __name__ == '__main__':
    df = score_all_articles()
    print('\nFinBERT scoring complete!')
    print(f'Output shape: {df.shape}')
    print(f'Columns: {df.columns.tolist()}')