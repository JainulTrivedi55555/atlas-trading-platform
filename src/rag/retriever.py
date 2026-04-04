from src.rag.embedder import load_faiss_index
from src.utils.config import TOP_K_RETRIEVAL, TICKERS
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

def retrieve(query: str,
              ticker: str = None,
              k: int = None,
              vectorstore=None) -> list:
    """
    Retrieve top-K relevant chunks for a query.
    ticker: optional filter to search only one company's filings
    Returns: list of (Document, score) tuples
    """
    if k is None:
        k = TOP_K_RETRIEVAL
    
    if vectorstore is None:
        vectorstore = load_faiss_index()
    
    if ticker:
        # Filter by ticker using metadata
        # We over-fetch (k*3) because the initial vector search isn't pre-filtered
        results = vectorstore.similarity_search_with_score(
            query, k=k*3
        )
        # Filter the results by the specific ticker
        results = [(doc, score) for doc, score in results 
                   if doc.metadata.get('ticker') == ticker.upper()][:k]
    else:
        results = vectorstore.similarity_search_with_score(query, k=k)
        
    return results

def format_results(results: list) -> str:
    """Format retrieval results for display."""
    output = []
    for i, (doc, score) in enumerate(results):
        meta = doc.metadata
        # Note: Score in FAISS L2 is distance (lower is better), 
        # so 1-score is a proxy for similarity percentage.
        output.append(
            f'--- Result {i+1} ---\n'
            f'Ticker: {meta["ticker"]} | '
            f'Form: {meta["form_type"]} | '
            f'Date: {meta["date"]} | '
            f'Similarity: {1-score:.3f}\n'
            f'{doc.page_content[:500]}...'
        )
    return '\n\n'.join(output)

def retrieve_for_ticker(ticker: str, query: str, k: int = 5) -> str:
    """Convenience function to retrieve for a specific ticker."""
    results = retrieve(query, ticker=ticker, k=k)
    if not results:
        return f'No results found for {ticker}: {query}'
    
    return format_results(results)

if __name__ == '__main__':
    # Test retrieval
    test_queries = [
        ('AAPL', 'iPhone demand risks and competition'),
        ('NVDA', 'data center revenue growth'),
        ('JPM',  'credit loss provisions and loan defaults'),
        ('META', 'advertising revenue and user growth'),
        ('TSLA', 'production capacity and delivery targets'),
    ]
    
    # Load index once for all queries
    vs = load_faiss_index()
    
    for ticker, query in test_queries:
        print(f'\n{"="*60}')
        print(f'QUERY: [{ticker}] {query}')
        print(f'{"="*60}')
        results = retrieve(query, ticker=ticker, k=3, vectorstore=vs)
        print(format_results(results))