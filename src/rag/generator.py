from src.rag.retriever import retrieve
from pathlib import Path
import sys
import re
sys.path.append(str(Path(__file__).parent.parent.parent))


def generate_market_brief(ticker: str,
                           query: str,
                           model_signal: str = None,
                           vectorstore=None) -> str:
    """
    Generate a market intelligence brief for a ticker.
    ticker:       stock ticker e.g. 'AAPL'
    query:        what to look for e.g. 'revenue risks'
    model_signal: optional model prediction e.g. 'BEARISH'
    Returns:      formatted market brief string
    """
    results = retrieve(query, ticker=ticker, k=5,
                        vectorstore=vectorstore)

    if not results:
        return f'No relevant filings found for {ticker}.'

    # Extract all text from retrieved chunks
    all_text = ' '.join(doc.page_content for doc, _ in results)
    sentences = re.split(r'(?<=[.!?]) +', all_text)

    # Score sentences by keyword relevance
    query_words = set(query.lower().split())
    scored = []
    for sent in sentences:
        if len(sent) < 40 or len(sent) > 500:
            continue
        words   = set(sent.lower().split())
        overlap = len(words & query_words)
        scored.append((overlap, sent))

    scored.sort(reverse=True)

    # Deduplicate sentences
    seen = set()
    top_sentences = []
    for _, sent in scored:
        sent_clean = sent.strip()
        key = sent_clean[:60]
        if key not in seen:
            seen.add(key)
            top_sentences.append(sent_clean)
        if len(top_sentences) >= 5:
            break

    # Fallback if no sentences scored
    if not top_sentences:
        top_sentences = [s.strip() for s in sentences[:5]
                         if len(s.strip()) > 40]

    # Get source info filtered by ticker
    sources = []
    for doc, score in results[:3]:
        meta = doc.metadata
        if meta.get('ticker') == ticker.upper():
            sources.append(
                f"{meta['ticker']} {meta['form_type']} "
                f"({meta['date']}) "
                f"[similarity: {1-score:.2f}]"
            )

    # Build brief
    brief_lines = [
        f'ATLAS MARKET BRIEF — {ticker}',
        f'Query: {query}',
        '=' * 50,
    ]

    if model_signal:
        brief_lines += [
            f'Model Signal: {model_signal}',
            '-' * 50,
        ]

    brief_lines += ['Key Findings from SEC Filings:', '']
    for i, sent in enumerate(top_sentences, 1):
        brief_lines.append(f'{i}. {sent}')
        brief_lines.append('')

    brief_lines += ['-' * 50, 'Sources:']
    if sources:
        brief_lines.extend(f'  - {s}' for s in sources)
    else:
        brief_lines.append(f'  - {ticker} filings retrieved')

    return '\n'.join(brief_lines)


def generate_all_briefs(tickers: list,
                         queries: dict,
                         vectorstore=None) -> dict:
    """Generate market briefs for multiple tickers."""
    from src.rag.embedder import load_faiss_index
    if vectorstore is None:
        vectorstore = load_faiss_index()
    briefs = {}
    for ticker in tickers:
        query = queries.get(ticker, 'revenue growth risks competition')
        briefs[ticker] = generate_market_brief(
            ticker, query, vectorstore=vectorstore
        )
        print(f'Generated brief for {ticker}')
    return briefs


if __name__ == '__main__':
    brief = generate_market_brief(
        ticker='AAPL',
        query='iPhone demand risks competition China',
        model_signal='BEARISH — Model AUC 0.72 predicts down day'
    )
    print(brief)